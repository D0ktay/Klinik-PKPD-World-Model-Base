"""
Transient CircAdapt Sürücüsü Ön-Ölçümü (Faz 0)
==================================================

`transient_integration.py::run_transient_trajectory()`'yi büyük ölçekte
kullanmadan ÖNCE: (1) çağrı-başına süreyi (stable=True başlangıç ile
stable=False adımları AYRI AYRI) ölç, (2) "mutlak hedef" mantığının
gerçekten biriktirmediğini SAĞLAMA yap -- trajectory'nin SON karesi
(t=75dk, esmolol konsantrasyonu pik değerin ~%1'ine düşmüş), AYNI
trajectory'nin KENDİ frame_idx=0'ına (ilaçsız kararlı-durum) YAKIN olmalı.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from generate_dynamics_dataset import sample_synthetic_patient, ESMOLOL_DOSE_RANGE_MG_PER_KG
from transient_integration import run_transient_trajectory, DEFAULT_FRAME_INTERVAL_MIN
from worldmodel.patient import load_verified_drugs
import dataclasses


def main(n_patients: int = 2):
    rng = np.random.default_rng(123)
    verified = load_verified_drugs(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "drugs_verified.yaml")
    )
    esmolol_base = verified["esmolol"]["drug"]
    mid_dose = float(np.mean(ESMOLOL_DOSE_RANGE_MG_PER_KG))

    baseline_call_times = []  # tek stable=True cagrisi (run_baseline icinde)
    step_call_times = []      # her stable=False adimi

    for i in range(n_patients):
        patient = sample_synthetic_patient(rng, i)
        drug = dataclasses.replace(
            esmolol_base, dose_mg_per_kg=mid_dose, dose_mg=mid_dose * patient.weight_kg,
        )

        # baseline (stable=True) suresini ayri olcmek icin run_transient_trajectory'nin
        # disinda bir kez run_baseline cagirip atiyoruz -- asil trajectory cagrisi
        # zaten kendi ici run_baseline'i da yapiyor, o yuzden STEP surelerini
        # trajectory cagrisinin toplam suresinden cikararak tahmin edecegiz.
        from integrate_drug_with_circadapt import run_baseline
        t0 = time.time()
        run_baseline(patient)
        baseline_call_times.append(time.time() - t0)

        t0 = time.time()
        result = run_transient_trajectory(patient, drug)
        total_time = time.time() - t0
        n_steps = len(result.frames) - 1  # frame_idx=0 haric, gerceklesen stable=False adim sayisi
        if n_steps > 0:
            # toplam sureden trajectory'nin KENDI ic run_baseline cagrisinin
            # (yaklasik olarak baseline_call_times'in son olcumune esit) suresini
            # cikarip kalanini n_steps'e bolerek adim-basina yaklasik sure buluyoruz.
            approx_step_time = (total_time - baseline_call_times[-1]) / n_steps
            step_call_times.append(approx_step_time)

        print(f"Hasta {i}: {len(result.frames)} kare uretildi "
              f"(beklenen {round(75.0 / DEFAULT_FRAME_INTERVAL_MIN) + 1}), "
              f"truncated={result.truncated}, toplam_sure={total_time:.2f}sn")

        # --- Saglama: son kare, frame_idx=0'a YAKIN olmali (ilac etkisi ~%1'e dusmus) ---
        frame0 = result.frames[0]
        frame_last = result.frames[-1]
        edv0 = float(np.max(frame0["v"]))
        edv_last = float(np.max(frame_last["v"]))
        pct_diff = abs(edv_last - edv0) / edv0 * 100
        print(f"  Saglama: EDV(frame0)={edv0:.3f} mL, EDV(son_kare)={edv_last:.3f} mL, "
              f"fark=%{pct_diff:.2f} (kucuk olmali, ilac neredeyse elimine olmus)")
        print(f"  HR(frame0)={frame0['current_hr']:.2f} bpm, HR(son_kare)={frame_last['current_hr']:.2f} bpm, "
              f"conc(son_kare)={frame_last['conc_mg_L']:.5f} mg/L")

    baseline_arr = np.array(baseline_call_times)
    step_arr = np.array(step_call_times)
    print("\n--- Ozet ---")
    print(f"stable=True (baseline) cagri suresi: ort {baseline_arr.mean():.3f} sn (n={len(baseline_arr)})")
    if len(step_arr):
        print(f"stable=False (adim) cagri suresi:   ort {step_arr.mean():.4f} sn (n={len(step_arr)})")
        per_trajectory = baseline_arr.mean() + 30 * step_arr.mean()
        print(f"\nTrajectory basina tahmini sure (1 baseline + 30 adim): {per_trajectory:.2f} sn")
        for n_pat, n_dose in [(40, 3), (60, 3)]:
            total_min = per_trajectory * n_pat * n_dose / 60.0
            print(f"  {n_pat} hasta x {n_dose} doz tahmini toplam sure: ~{total_min:.1f} dakika")


if __name__ == "__main__":
    main()
