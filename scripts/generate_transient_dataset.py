"""
Zaman-İçi (Transient) Otoregresif Veri Üretimi (Faz 2)
==========================================================

`transient_integration.py::run_transient_trajectory()`'yi çok sayıda
sentetik hastada çalıştırıp, kalbin atım-atım (kare-kare) hareketini
`data/transient_dataset/dataset.npz`'e kaydeder. Tek-adımlı MVP'nin
(`scripts/generate_dynamics_dataset.py`) SATIR=trajectory şemasından
FARKLI olarak, burada SATIR=KARE -- her (hasta,doz) çifti (`trajectory_id`)
birden fazla satıra (kareye) yayılır.

MALİYET UYARISI (Faz 0 benchmark'ında ÖLÇÜLDÜ, tahmin değil): her
trajectory ~96 saniye sürüyor (1 `stable=True` + 30 `stable=False` adım,
adım başına ~150-190 atım simüle ediliyor -- `stable=False`'ın
`stable=True`'dan ~5.6x daha pahalı olduğu ortaya çıktı). Bu yüzden
varsayılan ölçek tek-adımlı MVP'den (300x5) KÜÇÜK: 24 hasta x 2 doz = 48
trajectory ~77 dakika.

PARALELLEŞTİRME (10-20x veri büyütme çalışması): CircAdapt koşumları
process-güvenli olduğu izole bir deneyle doğrulandı (paylaşılan durum
YOK, her process kendi DLL örneğini yüklüyor). Ama hasta ÖRNEKLEMESİ
(`sample_synthetic_patient`) TEK, sıralı ilerleyen bir `rng` nesnesine
bağlı -- bu adımı PARALELLEŞTİRMİYORUZ (paralel/sırasız tüketim, aynı
seed'in aynı hastaları üretme garantisini BOZARDI). Bu yüzden akış iki
aşamaya ayrılıyor: (1) TÜM hastalar ANA process'te, sırayla, ucuza
örneklenir (CircAdapt çağrılmaz, sadece rng çekişi) -- (2) SADECE pahalı
CircAdapt simülasyonu (`run_transient_trajectory`) worker process'lere
dağıtılır, her worker tam bir `Patient`+`Drug` nesnesi alır.
"""

import argparse
import dataclasses
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

from generate_dynamics_dataset import (
    sample_synthetic_patient, resample_trajectory, ESMOLOL_DOSE_RANGE_MG_PER_KG,
    TRAJECTORY_N_POINTS,
)
from transient_integration import run_transient_trajectory, DEFAULT_WINDOW_MIN, DEFAULT_FRAME_INTERVAL_MIN
from worldmodel.patient import load_verified_drugs
from worldmodel.clinical_metrics import ejection_fraction, cardiac_output

DEFAULT_N_WORKERS = 6  # fiziksel çekirdek sayısı -- CPU-bound iş için
                        # hyperthreading'den fayda beklenmiyor (izole
                        # deneyde doğrulandı, mantıksal 12'nin YARISI).

# Karar 2 (kullanıcı onaylı): i<130 hastalar MEVCUT i%10 kuralını AYNEN
# korur (13 test hastası, index [9,19,...,129] -- rapor edilen R²
# sonuçlarının dayandığı test kümesi DEĞİŞMEMELİ). i>=130 hastalar HER
# ZAMAN "train" -- yeni eklenen hiçbir hasta val/test'e sızmaz, mevcut
# değerlendirme kümesiyle önceki/sonraki karşılaştırma anlamlı kalır.
SPLIT_BOUNDARY_PATIENT_INDEX = 130


def determine_split(patient_index: int) -> str:
    """Saf fonksiyon, CircAdapt/rastgelelik İÇERMEZ -- bu yüzden
    `tests/test_generate_transient_dataset.py`'de CircAdapt hiç
    çağrılmadan izole test edilebiliyor (bkz. o dosya)."""
    if patient_index >= SPLIT_BOUNDARY_PATIENT_INDEX:
        return "train"
    if patient_index % 10 < 8:
        return "train"
    elif patient_index % 10 == 8:
        return "val"
    return "test"


def _frames_to_records(frames: list[dict], truncated: bool, trajectory_id: int, patient_id: int,
                        split: str, dose_mg_per_kg: float, patient) -> list[dict]:
    records = []
    for frame in frames:
        p_r = resample_trajectory(frame["t"], frame["p"], n_points=TRAJECTORY_N_POINTS)
        v_r = resample_trajectory(frame["t"], frame["v"], n_points=TRAJECTORY_N_POINTS)
        edv, esv = float(v_r.max()), float(v_r.min())
        ef = ejection_fraction(edv, esv)
        co = cardiac_output(edv, esv, frame["current_hr"])
        records.append({
            "trajectory_id": trajectory_id,
            "frame_idx": frame["frame_idx"],
            "patient_id": patient_id,
            "split": split,
            "dose_mg_per_kg": float(dose_mg_per_kg),
            "elapsed_min": frame["elapsed_min"],
            "conc_mg_L": frame["conc_mg_L"],
            "current_hr": frame["current_hr"],
            "truncated": truncated,
            "age": patient.age, "weight_kg": patient.weight_kg, "height_cm": patient.height_cm,
            "baseline_hr": patient.baseline_hr, "baseline_sbp": patient.baseline_sbp,
            "baseline_dbp": patient.baseline_dbp, "baseline_spo2": patient.baseline_spo2,
            "renal_function": patient.renal_function, "hepatic_function": patient.hepatic_function,
            "potassium_mEqL": patient.potassium_mEqL, "calcium_mgdL": patient.calcium_mgdL,
            "comorbidity": patient.comorbidity,
            "traj_p": p_r, "traj_v": v_r,
            "ef": ef, "co": co, "edv": edv, "esv": esv,
        })
    return records


def generate_transient_records(n_patients: int, n_doses: int, seed: int = 0,
                                window_min: float = DEFAULT_WINDOW_MIN,
                                frame_interval_min: float = DEFAULT_FRAME_INTERVAL_MIN,
                                verbose: bool = True) -> tuple[list[dict], dict]:
    """SIRALI (tek process) referans -- paralel sürümle karşılaştırma/
    benchmark için korunuyor, davranışı DEĞİŞMEDİ (sadece split ataması
    artık `determine_split()` üzerinden, mantık AYNI)."""
    rng = np.random.default_rng(seed)
    verified = load_verified_drugs(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "drugs_verified.yaml")
    )
    esmolol_base = verified["esmolol"]["drug"]

    records = []
    trajectory_id = 0
    n_trajectories_truncated = 0
    n_trajectories_total = 0

    for i in range(n_patients):
        patient = sample_synthetic_patient(rng, i)
        split = determine_split(i)
        doses = np.linspace(*ESMOLOL_DOSE_RANGE_MG_PER_KG, n_doses)

        for dose_mg_per_kg in doses:
            drug = dataclasses.replace(
                esmolol_base, dose_mg_per_kg=float(dose_mg_per_kg),
                dose_mg=float(dose_mg_per_kg) * patient.weight_kg,
            )

            t0 = time.time()
            result = run_transient_trajectory(patient, drug, window_min=window_min,
                                               frame_interval_min=frame_interval_min)
            elapsed = time.time() - t0
            n_trajectories_total += 1
            if result.truncated:
                n_trajectories_truncated += 1

            records.extend(_frames_to_records(
                result.frames, result.truncated, trajectory_id, i, split, dose_mg_per_kg, patient,
            ))
            trajectory_id += 1

            if verbose:
                print(f"  hasta={i} doz={dose_mg_per_kg:.3f} -> {len(result.frames)} kare, "
                      f"truncated={result.truncated}, {elapsed:.1f}sn")

    summary = {
        "n_trajectories_total": n_trajectories_total,
        "n_trajectories_truncated": n_trajectories_truncated,
        "n_frames_total": len(records),
    }
    return records, summary


def _simulate_trajectory_task(task: tuple) -> tuple[int, int, str, float, list[dict], bool]:
    """Worker process'te çalışır -- SADECE pahalı CircAdapt simülasyonunu
    yapar, hiç rastgelelik/örnekleme İÇERMEZ (o zaten ana process'te,
    task oluşturulurken bitmiş oluyor). `multiprocessing` spawn modunda
    (Windows varsayılanı) bu modülün TÜM üst-seviye importları (circadapt,
    numpy, worldmodel...) worker başına bir kez otomatik tekrar çalışır --
    bu ADIM 0'da gözlemlenen, kabul edilen bir maliyet, azaltılmaya
    çalışılmadı."""
    trajectory_id, patient_id, split, dose_mg_per_kg, patient, drug, window_min, frame_interval_min = task
    result = run_transient_trajectory(patient, drug, window_min=window_min, frame_interval_min=frame_interval_min)
    records = _frames_to_records(result.frames, result.truncated, trajectory_id, patient_id,
                                  split, dose_mg_per_kg, patient)
    return trajectory_id, patient_id, split, dose_mg_per_kg, records, result.truncated


def generate_transient_records_parallel(n_patients: int, n_doses: int, seed: int = 0,
                                         window_min: float = DEFAULT_WINDOW_MIN,
                                         frame_interval_min: float = DEFAULT_FRAME_INTERVAL_MIN,
                                         n_workers: int = DEFAULT_N_WORKERS,
                                         verbose: bool = True,
                                         progress_log_path: str | None = None,
                                         progress_every: int = 100) -> tuple[list[dict], dict]:
    """
    PARALEL üretim. `generate_transient_records()` ile BİREBİR AYNI
    kayıtları (aynı trajectory_id/split/patient_id ataması) üretir --
    TEK fark, CircAdapt simülasyonlarının `n_workers` process'e
    dağıtılması. Bunu sağlayan şey: hasta örnekleme (adım 1, aşağıda)
    TEK, sıralı `rng` ile ANA process'te bitiyor -- worker'lara sadece
    zaten-tamamlanmış `Patient`/`Drug` nesneleri (ve önceden atanmış
    `trajectory_id`/`split`) gidiyor, worker'lar hiç rastgelelik
    üretmiyor.
    """
    rng = np.random.default_rng(seed)
    verified = load_verified_drugs(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "drugs_verified.yaml")
    )
    esmolol_base = verified["esmolol"]["drug"]

    # --- Adım 1: TÜM hastaları ve görev listesini ANA process'te,
    # SIRAYLA üret -- ucuz (sadece rng çekişi), CircAdapt çağrılmıyor.
    tasks = []
    trajectory_id = 0
    for i in range(n_patients):
        patient = sample_synthetic_patient(rng, i)
        split = determine_split(i)
        doses = np.linspace(*ESMOLOL_DOSE_RANGE_MG_PER_KG, n_doses)
        for dose_mg_per_kg in doses:
            drug = dataclasses.replace(
                esmolol_base, dose_mg_per_kg=float(dose_mg_per_kg),
                dose_mg=float(dose_mg_per_kg) * patient.weight_kg,
            )
            tasks.append((trajectory_id, i, split, float(dose_mg_per_kg), patient, drug,
                          window_min, frame_interval_min))
            trajectory_id += 1

    # --- Adım 2: SADECE pahalı simülasyonu worker'lara dağıt.
    records = []
    n_trajectories_truncated = 0
    n_completed = 0
    log_file = open(progress_log_path, "a", encoding="utf-8") if progress_log_path else None
    t_start = time.time()
    try:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for traj_id, patient_id, split, dose, traj_records, truncated in pool.imap_unordered(
                _simulate_trajectory_task, tasks
            ):
                records.extend(traj_records)
                n_completed += 1
                if truncated:
                    n_trajectories_truncated += 1
                if verbose:
                    print(f"  [{n_completed}/{len(tasks)}] trajectory_id={traj_id} hasta={patient_id} "
                          f"doz={dose:.3f} -> {len(traj_records)} kare, truncated={truncated}")
                if log_file and n_completed % progress_every == 0:
                    elapsed_min = (time.time() - t_start) / 60.0
                    rate_per_min = n_completed / elapsed_min if elapsed_min > 0 else 0.0
                    remaining = len(tasks) - n_completed
                    eta_min = remaining / rate_per_min if rate_per_min > 0 else float("nan")
                    line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"{n_completed}/{len(tasks)} trajectory tamamlandı "
                            f"(geçen süre {elapsed_min:.1f} dk, tahmini kalan süre {eta_min:.1f} dk)\n")
                    log_file.write(line)
                    log_file.flush()
                    if verbose:
                        print(line, end="")
    finally:
        if log_file:
            elapsed_min = (time.time() - t_start) / 60.0
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"TAMAMLANDI: {n_completed}/{len(tasks)} trajectory ({elapsed_min:.1f} dk)\n")
            log_file.close()

    summary = {
        "n_trajectories_total": len(tasks),
        "n_trajectories_truncated": n_trajectories_truncated,
        "n_frames_total": len(records),
    }
    return records, summary


def save_transient_dataset(records: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    manifest_rows = [{k: v for k, v in r.items() if not k.startswith("traj_")} for r in records]
    pd.DataFrame(manifest_rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    scalar_int_fields = ["trajectory_id", "frame_idx", "patient_id"]
    scalar_float_fields = [
        "dose_mg_per_kg", "elapsed_min", "conc_mg_L", "current_hr",
        "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp", "baseline_dbp",
        "baseline_spo2", "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
        "ef", "co", "edv", "esv",
    ]
    arrays = {f: np.array([r[f] for r in records], dtype=np.int64) for f in scalar_int_fields}
    arrays.update({f: np.array([r[f] for r in records], dtype=np.float32) for f in scalar_float_fields})
    arrays["split"] = np.array([r["split"] for r in records])
    arrays["truncated"] = np.array([r["truncated"] for r in records], dtype=bool)
    arrays["comorbidity"] = np.array([r["comorbidity"] if r["comorbidity"] is not None else "none" for r in records])
    arrays["traj_p"] = np.stack([r["traj_p"] for r in records]).astype(np.float32)
    arrays["traj_v"] = np.stack([r["traj_v"] for r in records]).astype(np.float32)

    np.savez_compressed(os.path.join(out_dir, "dataset.npz"), **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=24)
    parser.add_argument("--n-doses", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-min", type=float, default=DEFAULT_WINDOW_MIN)
    parser.add_argument("--frame-interval-min", type=float, default=DEFAULT_FRAME_INTERVAL_MIN)
    parser.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "transient_dataset"))
    parser.add_argument("--sequential", action="store_true",
                         help="Paralel yerine tek-process sırayla çalıştır (benchmark/referans için).")
    parser.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS,
                         help=f"Paralel modda process sayısı (varsayılan {DEFAULT_N_WORKERS}, fiziksel çekirdek sayısı).")
    parser.add_argument("--progress-log", type=str, default=None,
                         help="Verilirse, ilerleme her --progress-every trajectory'de bir bu dosyaya yazılır (terminale değil).")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    mode = "sıralı (tek process)" if args.sequential else f"paralel ({args.n_workers} process)"
    print(f"Transient veri uretimi basliyor [{mode}]: {args.n_patients} hasta x {args.n_doses} doz "
          f"(hedef {args.n_patients * args.n_doses} trajectory)...")
    t0 = time.time()
    if args.sequential:
        records, summary = generate_transient_records(
            args.n_patients, args.n_doses, seed=args.seed,
            window_min=args.window_min, frame_interval_min=args.frame_interval_min,
        )
    else:
        records, summary = generate_transient_records_parallel(
            args.n_patients, args.n_doses, seed=args.seed,
            window_min=args.window_min, frame_interval_min=args.frame_interval_min,
            n_workers=args.n_workers, progress_log_path=args.progress_log,
            progress_every=args.progress_every,
        )
    elapsed = time.time() - t0

    save_transient_dataset(records, args.out_dir)

    print(f"\nTamamlandi ({elapsed / 60.0:.1f} dakika).")
    print(f"  Toplam trajectory: {summary['n_trajectories_total']}")
    print(f"  Kesilen (truncated) trajectory: {summary['n_trajectories_truncated']}")
    print(f"  Toplam kare (satir): {summary['n_frames_total']}")
    print(f"  Kaydedildi: {args.out_dir}")


if __name__ == "__main__":
    main()
