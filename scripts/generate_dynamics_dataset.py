"""
Öğrenilmiş Dinamik Model (JEPA) için Sentetik Veri Üretimi
============================================================

Bu script, `src/worldmodel/learned_dynamics/` altındaki JEPA (Joint Embedding
Predictive Architecture -- "birleşik-gömme öngörü mimarisi") modelinin eğitim
verisini üretir.

NE ÜRETİYORUZ: (durum, aksiyon, sonraki-durum) üçlüleri. CircAdapt (gerçek
kalp mekaniği motoru, bkz. integrate_drug_with_circadapt.py) hiçbir veriden
öğrenmiyor -- burada onu bir "veri fabrikası" gibi tekrar tekrar koşturup,
onun ürettiği fizik-tabanlı sonuçlardan bir sinir ağının öğreneceği bir
veri kümesi biriktiriyoruz. Yani JEPA modeli GERÇEK bir kalbi değil,
CircAdapt'in DAVRANIŞINI taklit etmeyi öğrenecek -- doğruluk tavanı
CircAdapt'in kendisiyle sınırlı.

MVP KAPSAM SINIRI (bilinçli, plan onaylı):
  - Tek ilaç sınıfı: beta-bloker (configs/drugs_verified.yaml > esmolol).
  - Tek-adımlı geçiş: "ilaçsız durum -> ilaç sonrası durum" (steady-state
    öncesi/sonrası), CircAdapt'in kendisi de zaten sabit-durum (`stable=True`)
    çözüyor -- saatler süren bir zaman-serisi rollout YOK.
  - known_av_block_degree HER ZAMAN None: "first"/"second" dereceler için
    zaten sayısal bir büyüklük modellenmiyor (bkz. Patient.known_av_block_degree
    docstring'i), "third" ise CircAdapt'i hiç çalıştırmadan atlanıyor (aşağıya
    bakın) -- rastgele örneklemede bu alana yer vermek, öğrenilecek hiçbir
    sinyali olmayan bir eksen eklemek olurdu.

Kayıt formatı: .npz (numpy zaten proje bağımlılığı, yeni bir paket
gerektirmiyor) + insan-okunur manifest.csv (pandas ile). Projede bundan önce
toplu simülasyon verisi saklama emsali yoktu -- bu, o kalıbı ilk kez kuruyor.
"""

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

from circadapt.error import CircAdaptException

from worldmodel.patient import Patient, load_verified_drugs
from worldmodel.clinical_metrics import ejection_fraction, cardiac_output
from worldmodel.pd import AV_BLOCK_THRESHOLD_MULTIPLIER
from integrate_drug_with_circadapt import (
    run_baseline, run_with_drug, lv_pressure_volume,
    compute_drug_effect, cumulative_av_conduction_multiplier,
)

# esmolol'ün referans yükleme dozu 0.5 mg/kg (FDA etiketi, bkz.
# configs/drugs_verified.yaml > esmolol.calibration_notes). MVP doz aralığı
# bunun etrafında, klinik olarak makul bir bant: 0.1-1.0 mg/kg.
ESMOLOL_DOSE_RANGE_MG_PER_KG = (0.1, 1.0)

# Kronik komorbidite (hastanın ilaçtan/elektrolitten bağımsız temel kalp/damar
# durumu, bkz. Patient.comorbidity) -- ağırlıklı örnekleme: çoğu sentetik
# hasta sağlıklı, azınlık bir komorbiditeye sahip (gerçekçi popülasyon
# dağılımını kabaca yansıtmak için).
COMORBIDITY_CHOICES = [None, "heart_failure", "hypertension"]
COMORBIDITY_WEIGHTS = [0.7, 0.15, 0.15]

TRAJECTORY_N_POINTS = 100


def sample_synthetic_patient(rng: np.random.Generator, patient_id: int) -> Patient:
    """
    Patient uzayından (bkz. src/worldmodel/patient.py) klinik olarak makul
    bir nokta örnekler. Aralıklar, configs/patients.yaml'daki 6 örnek hastanın
    ve calibrate_circadapt_to_patient() docstring'indeki "Streamlit slider
    sınırı 50-110 bpm" gibi elle doğrulanmış sayısal kararlılık notlarının
    etrafında seçildi -- CircAdapt'in bilinen kararsızlık bölgelerinden
    (örn. çok uç nabız değerleri) kaçınmak için.
    """
    weight_kg = float(rng.uniform(45, 110))
    height_cm = float(rng.uniform(150, 195))
    age = int(rng.uniform(25, 85))
    return Patient(
        name=f"synthetic_{patient_id:05d}",
        weight_kg=weight_kg,
        height_cm=height_cm,
        age=age,
        blood_type="bilinmiyor",
        baseline_hr=float(rng.uniform(50, 105)),
        baseline_sbp=float(rng.uniform(95, 175)),
        baseline_dbp=float(rng.uniform(60, 105)),
        baseline_spo2=float(rng.uniform(92, 100)),
        renal_function=float(rng.uniform(0.3, 1.0)),
        hepatic_function=float(rng.uniform(0.3, 1.0)),
        # Normal aralık 3.5-5.0 mEq/L (bkz. Patient docstring'i) -- üst sınırı
        # 6.5'e kadar açıyoruz (belirgin hiperkalemi dahil, hasta_c_hiperkalemi
        # ile aynı büyüklük mertebesi) ki model AV-iletim-yavaşlaması sinyalini
        # de görsün; AV bloğu tetikleyecek kadar uç durumlar zaten aşağıdaki
        # ön-kontrolle (cumulative_av_conduction_multiplier) elenip atlanıyor.
        potassium_mEqL=float(rng.uniform(3.0, 6.5)),
        calcium_mgdL=float(rng.uniform(7.0, 11.5)),
        comorbidity=rng.choice(COMORBIDITY_CHOICES, p=COMORBIDITY_WEIGHTS),
        known_av_block_degree=None,
    )


def resample_trajectory(t: np.ndarray, series: np.ndarray, n_points: int = TRAJECTORY_N_POINTS) -> np.ndarray:
    """
    CircAdapt'in ürettiği (değişken uzunlukta) zaman dizisini, sabit
    uzunlukta bir vektöre yeniden örnekler -- sinir ağının sabit boyutlu
    girdiye ihtiyacı var, CircAdapt'in çözücüsü ise her koşumda farklı
    sayıda zaman noktası üretebiliyor. Zamanı [0, 1] aralığına normalize
    edip (kendi süresinin fraksiyonu olarak) enterpole ediyoruz -- bu
    yüzden kalp hızı farklı hastalarda farklı olsa bile (dolayısıyla
    döngü süresi farklı olsa bile), her hastanın kendi döngüsü aynı
    sayıda noktayla temsil ediliyor.
    """
    t = np.asarray(t, dtype=float)
    series = np.asarray(series, dtype=float)
    span = t[-1] - t[0]
    if span <= 0:
        raise ValueError("resample_trajectory: zaman dizisi monoton artmıyor")
    t_norm = (t - t[0]) / span
    grid = np.linspace(0.0, 1.0, n_points)
    return np.interp(grid, t_norm, series)


def generate_records(n_patients: int, n_doses: int, seed: int = 0, verbose: bool = True) -> tuple[list[dict], dict]:
    """
    n_patients hasta örnekler, her biri için n_doses farklı esmolol dozunda
    (baseline, ilaçlı) CircAdapt çifti koşturur. Başarısız/atlanan koşumlar
    sessizce yutulmuyor -- sayaçlarla birlikte raporlanıyor (bkz. modül
    docstring'i, `run_stable`'ın kendi CircAdaptException/ModelCrashed
    davranışı hakkındaki not).

    Dönüş: (kayıt listesi, özet istatistikler sözlüğü).
    """
    rng = np.random.default_rng(seed)
    verified = load_verified_drugs(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "drugs_verified.yaml")
    )
    esmolol_base = verified["esmolol"]["drug"]

    records = []
    n_baseline_crashed = 0
    n_dose_crashed = 0
    n_dose_av_block_skipped = 0
    n_dose_success = 0

    for i in range(n_patients):
        patient = sample_synthetic_patient(rng, i)

        try:
            baseline_model = run_baseline(patient)
        except CircAdaptException:
            n_baseline_crashed += 1
            continue

        t_base, p_base, v_base = lv_pressure_volume(baseline_model)
        hr_base_model = 60.0 / baseline_model["General"]["t_cycle"]
        p_base_r = resample_trajectory(t_base, p_base)
        v_base_r = resample_trajectory(t_base, v_base)
        edv_base, esv_base = float(v_base_r.max()), float(v_base_r.min())
        ef_base = ejection_fraction(edv_base, esv_base)
        co_base = cardiac_output(edv_base, esv_base, hr_base_model)

        doses = np.linspace(*ESMOLOL_DOSE_RANGE_MG_PER_KG, n_doses)
        # Hasta-bazlı split ataması: aynı hastanın TÜM dozları aynı split'te
        # kalmalı (sızıntıyı önlemek için) -- indekse göre deterministik.
        split = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test")

        for dose_mg_per_kg in doses:
            drug = dataclasses.replace(
                esmolol_base,
                dose_mg_per_kg=float(dose_mg_per_kg),
                dose_mg=float(dose_mg_per_kg) * patient.weight_kg,
            )
            drug_effect = compute_drug_effect(patient, drug)

            av_multiplier = cumulative_av_conduction_multiplier(patient, [drug], [drug_effect])
            if av_multiplier >= AV_BLOCK_THRESHOLD_MULTIPLIER:
                n_dose_av_block_skipped += 1
                continue

            try:
                drug_model = run_with_drug(patient, drug, drug_effect)
            except CircAdaptException:
                n_dose_crashed += 1
                continue

            t_drug, p_drug, v_drug = lv_pressure_volume(drug_model)
            hr_drug_model = 60.0 / drug_model["General"]["t_cycle"]
            p_drug_r = resample_trajectory(t_drug, p_drug)
            v_drug_r = resample_trajectory(t_drug, v_drug)
            edv_drug, esv_drug = float(v_drug_r.max()), float(v_drug_r.min())
            ef_drug = ejection_fraction(edv_drug, esv_drug)
            co_drug = cardiac_output(edv_drug, esv_drug, hr_drug_model)

            records.append({
                "patient_id": i,
                "split": split,
                "dose_mg_per_kg": float(dose_mg_per_kg),
                "age": patient.age, "weight_kg": patient.weight_kg, "height_cm": patient.height_cm,
                "baseline_hr": patient.baseline_hr, "baseline_sbp": patient.baseline_sbp,
                "baseline_dbp": patient.baseline_dbp, "baseline_spo2": patient.baseline_spo2,
                "renal_function": patient.renal_function, "hepatic_function": patient.hepatic_function,
                "potassium_mEqL": patient.potassium_mEqL, "calcium_mgdL": patient.calcium_mgdL,
                "comorbidity": patient.comorbidity,
                "traj_base_p": p_base_r, "traj_base_v": v_base_r,
                "traj_drug_p": p_drug_r, "traj_drug_v": v_drug_r,
                "hr_base": hr_base_model, "hr_drug": hr_drug_model,
                "ef_base": ef_base, "ef_drug": ef_drug,
                "co_base": co_base, "co_drug": co_drug,
                "edv_base": edv_base, "esv_base": esv_base,
                "edv_drug": edv_drug, "esv_drug": esv_drug,
            })
            n_dose_success += 1

        if verbose and (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{n_patients} hasta işlendi "
                  f"(başarılı doz: {n_dose_success}, çöken doz: {n_dose_crashed}, "
                  f"AV-bloğu atlanan: {n_dose_av_block_skipped})")

    summary = {
        "n_patients_requested": n_patients,
        "n_baseline_crashed": n_baseline_crashed,
        "n_dose_success": n_dose_success,
        "n_dose_crashed": n_dose_crashed,
        "n_dose_av_block_skipped": n_dose_av_block_skipped,
    }
    return records, summary


def save_dataset(records: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    manifest_rows = [{k: v for k, v in r.items() if not k.startswith("traj_")} for r in records]
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    arrays = {
        "patient_id": np.array([r["patient_id"] for r in records], dtype=np.int64),
        "split": np.array([r["split"] for r in records]),
        "dose_mg_per_kg": np.array([r["dose_mg_per_kg"] for r in records], dtype=np.float32),
        "age": np.array([r["age"] for r in records], dtype=np.float32),
        "weight_kg": np.array([r["weight_kg"] for r in records], dtype=np.float32),
        "height_cm": np.array([r["height_cm"] for r in records], dtype=np.float32),
        "baseline_hr": np.array([r["baseline_hr"] for r in records], dtype=np.float32),
        "baseline_sbp": np.array([r["baseline_sbp"] for r in records], dtype=np.float32),
        "baseline_dbp": np.array([r["baseline_dbp"] for r in records], dtype=np.float32),
        "baseline_spo2": np.array([r["baseline_spo2"] for r in records], dtype=np.float32),
        "renal_function": np.array([r["renal_function"] for r in records], dtype=np.float32),
        "hepatic_function": np.array([r["hepatic_function"] for r in records], dtype=np.float32),
        "potassium_mEqL": np.array([r["potassium_mEqL"] for r in records], dtype=np.float32),
        "calcium_mgdL": np.array([r["calcium_mgdL"] for r in records], dtype=np.float32),
        "comorbidity": np.array([r["comorbidity"] if r["comorbidity"] is not None else "none" for r in records]),
        "traj_base_p": np.stack([r["traj_base_p"] for r in records]).astype(np.float32),
        "traj_base_v": np.stack([r["traj_base_v"] for r in records]).astype(np.float32),
        "traj_drug_p": np.stack([r["traj_drug_p"] for r in records]).astype(np.float32),
        "traj_drug_v": np.stack([r["traj_drug_v"] for r in records]).astype(np.float32),
        "hr_base": np.array([r["hr_base"] for r in records], dtype=np.float32),
        "hr_drug": np.array([r["hr_drug"] for r in records], dtype=np.float32),
        "ef_base": np.array([r["ef_base"] for r in records], dtype=np.float32),
        "ef_drug": np.array([r["ef_drug"] for r in records], dtype=np.float32),
        "co_base": np.array([r["co_base"] for r in records], dtype=np.float32),
        "co_drug": np.array([r["co_drug"] for r in records], dtype=np.float32),
        "edv_base": np.array([r["edv_base"] for r in records], dtype=np.float32),
        "esv_base": np.array([r["esv_base"] for r in records], dtype=np.float32),
        "edv_drug": np.array([r["edv_drug"] for r in records], dtype=np.float32),
        "esv_drug": np.array([r["esv_drug"] for r in records], dtype=np.float32),
    }
    np.savez_compressed(os.path.join(out_dir, "dataset.npz"), **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=300)
    parser.add_argument("--n-doses", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dynamics_dataset"))
    args = parser.parse_args()

    print(f"Sentetik veri üretimi başlıyor: {args.n_patients} hasta x {args.n_doses} doz "
          f"(hedef {args.n_patients * args.n_doses} koşum)...")
    t0 = time.time()
    records, summary = generate_records(args.n_patients, args.n_doses, seed=args.seed)
    elapsed = time.time() - t0

    save_dataset(records, args.out_dir)

    print(f"\nTamamlandı ({elapsed:.1f} sn).")
    print(f"  Talep edilen hasta: {summary['n_patients_requested']}")
    print(f"  Baseline çöktü (hasta tamamen atlandı): {summary['n_baseline_crashed']}")
    print(f"  Başarılı (hasta, doz) çifti: {summary['n_dose_success']}")
    print(f"  İlaçlı koşum çöktü: {summary['n_dose_crashed']}")
    print(f"  AV-bloğu riski nedeniyle atlandı: {summary['n_dose_av_block_skipped']}")
    print(f"  Kaydedildi: {args.out_dir}")


if __name__ == "__main__":
    main()
