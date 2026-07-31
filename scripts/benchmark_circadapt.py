"""
CircAdapt Koşum Maliyeti Ön-Ölçümü (Faz 0)
=============================================

generate_dynamics_dataset.py ile büyük ölçekli sentetik veri üretmeden ÖNCE
çalıştırılmalı. CircAdapt'in tek koşum süresi (bkz. run_stable() -- sabit
atım sayısıyla değil, `stable=True` ile kendi kendine yakınsıyor, yani süresi
hasta/ilaç parametrelerine göre DEĞİŞKEN) kod tabanında hiçbir yerde ölçülmüş
değil -- bu script küçük bir örneklemle ortalama süreyi VE çökme/atlanma
oranını ölçüp, tam ölçekli üretimin ne kadar süreceğini tahmin eder.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circadapt.error import CircAdaptException

from generate_dynamics_dataset import (
    sample_synthetic_patient, ESMOLOL_DOSE_RANGE_MG_PER_KG,
)
from worldmodel.patient import load_verified_drugs
from worldmodel.pd import AV_BLOCK_THRESHOLD_MULTIPLIER
from integrate_drug_with_circadapt import (
    run_baseline, run_with_drug, compute_drug_effect, cumulative_av_conduction_multiplier,
)

import dataclasses
import numpy as np


def main(n_patients: int = 20):
    rng = np.random.default_rng(42)
    verified = load_verified_drugs(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "drugs_verified.yaml")
    )
    esmolol_base = verified["esmolol"]["drug"]

    baseline_times, drug_times = [], []
    n_baseline_crash, n_drug_crash, n_av_skip = 0, 0, 0

    print(f"{n_patients} hasta üzerinde ölçüm yapılıyor "
          f"(her hasta için 1 baseline + 1 orta-doz ilaçlı koşum)...\n")

    for i in range(n_patients):
        patient = sample_synthetic_patient(rng, i)

        t0 = time.time()
        try:
            run_baseline(patient)
            baseline_times.append(time.time() - t0)
        except CircAdaptException:
            n_baseline_crash += 1
            continue

        mid_dose = float(np.mean(ESMOLOL_DOSE_RANGE_MG_PER_KG))
        drug = dataclasses.replace(
            esmolol_base, dose_mg_per_kg=mid_dose, dose_mg=mid_dose * patient.weight_kg,
        )
        drug_effect = compute_drug_effect(patient, drug)
        if cumulative_av_conduction_multiplier(patient, [drug], [drug_effect]) >= AV_BLOCK_THRESHOLD_MULTIPLIER:
            n_av_skip += 1
            continue

        t0 = time.time()
        try:
            run_with_drug(patient, drug, drug_effect)
            drug_times.append(time.time() - t0)
        except CircAdaptException:
            n_drug_crash += 1

    baseline_times = np.array(baseline_times)
    drug_times = np.array(drug_times)
    avg_per_pair = (baseline_times.mean() if len(baseline_times) else 0) + \
                   (drug_times.mean() if len(drug_times) else 0)

    print("--- Sonuç ---")
    print(f"Baseline koşum: ort {baseline_times.mean():.3f} sn (n={len(baseline_times)}), "
          f"çöken: {n_baseline_crash}")
    print(f"İlaçlı koşum:   ort {drug_times.mean():.3f} sn (n={len(drug_times)}), "
          f"çöken: {n_drug_crash}, AV-bloğu nedeniyle atlanan: {n_av_skip}")
    print(f"\n(hasta, doz) çifti başına tahmini süre: {avg_per_pair:.3f} sn")
    for n_patients_target, n_doses_target in [(300, 5), (500, 5)]:
        est_seconds = avg_per_pair * n_patients_target * n_doses_target
        print(f"  {n_patients_target} hasta x {n_doses_target} doz tahmini toplam süre: "
              f"~{est_seconds / 60:.1f} dakika")

    crash_rate = (n_baseline_crash + n_drug_crash) / max(n_patients, 1)
    print(f"\nÇökme oranı: %{crash_rate * 100:.1f} -- "
          f"generate_dynamics_dataset.py'nin --n-patients parametresini buna göre büyütün.")


if __name__ == "__main__":
    main()
