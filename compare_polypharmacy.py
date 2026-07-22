"""
Polifarmasi Karşılaştırması — Tehlikeli Bir İlaç Kombinasyonu
=================================================================

Esmolol (beta-bloker) ve digoksin (pozitif inotrop, ama AYNI
ZAMANDA negatif kronotrop -- AV düğümünü yavaşlatır) İKİ FARKLI
mekanizmadan gelse de, İKİSİ DE nabzı düşürür. Bu script, ikisinin
tek başına ve BİRLİKTE verildiğinde nabız/bradikardi riskini hem
istatistiksel PK/PD modelinde hem gerçek CircAdapt kalp mekaniğinde
karşılaştırır -- gerçek klinikte de bilinen bir "dikkatli kombine
edilmesi gereken ilaçlar" örneğidir (ikisi de AV iletimini yavaşlatır).
"""

import os
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from worldmodel.simulation import run_monte_carlo, run_polypharmacy_simulation, summarize
from integrate_drug_with_circadapt import run_comparison, run_polypharmacy_comparison


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))
    patient = patients["hasta_a"]

    # Referans (38mg) yerine daha ölçülü bir esmolol dozu -- etkileşim daha görünür.
    esmolol = replace(drugs["beta_bloker"], dose_mg=20.0, dose_mg_per_kg=None)
    digoxin = drugs["digoxin"]

    print(f"Hasta: {patient.name}")
    print(f"İlaç 1: {esmolol.display_name} ({esmolol.dose_mg:.1f} mg)")
    print(f"İlaç 2: {digoxin.display_name} ({digoxin.dose_mg:.2f} mg)\n")

    # --- İstatistiksel PK/PD ---
    print("=== İstatistiksel PK/PD (Monte Carlo) ===")
    solo_esmolol = summarize(run_monte_carlo(patient, esmolol, n_realizations=2000, seed=1))
    solo_digoxin = summarize(run_monte_carlo(patient, digoxin, n_realizations=2000, seed=1))
    combo_result = run_polypharmacy_simulation(patient, [esmolol, digoxin], n_realizations=2000, seed=1)
    combo = summarize(combo_result)

    print(f"Esmolol tek başına:  ortalama en düşük nabız={solo_esmolol['mean_min_hr']:.1f} bpm, "
          f"bradikardi riski=%{solo_esmolol['pct_bradycardia_risk']:.1f}")
    print(f"Digoksin tek başına: ortalama en düşük nabız={solo_digoxin['mean_min_hr']:.1f} bpm, "
          f"bradikardi riski=%{solo_digoxin['pct_bradycardia_risk']:.1f}")
    print(f"BİRLİKTE:            ortalama en düşük nabız={combo['mean_min_hr']:.1f} bpm, "
          f"bradikardi riski=%{combo['pct_bradycardia_risk']:.1f}")

    # --- CircAdapt (gerçek kalp mekaniği) ---
    print("\n=== CircAdapt (gerçek kalp mekaniği) ===")
    r_esmolol = run_comparison(patient, esmolol)
    r_digoxin = run_comparison(patient, digoxin)
    r_combo = run_polypharmacy_comparison(patient, [esmolol, digoxin])

    print(f"Esmolol tek başına:  {r_esmolol['hr_base']:.1f} -> {r_esmolol['hr_drug_model']:.1f} bpm")
    print(f"Digoksin tek başına: {r_digoxin['hr_base']:.1f} -> {r_digoxin['hr_drug_model']:.1f} bpm")
    print(f"BİRLİKTE:            {r_combo['hr_base']:.1f} -> {r_combo['hr_drug_model']:.1f} bpm")

    if r_combo["hr_drug_model"] < min(r_esmolol["hr_drug_model"], r_digoxin["hr_drug_model"]):
        print("\n-> Kombinasyon, İKİ İLACIN TEK BAŞINA ürettiğinden DAHA DÜŞÜK bir nabza yol açıyor --"
              " gerçek bir 'tehlikeli kombinasyon' davranışı, hem istatistiksel hem mekanik modelde tutarlı.")

    # --- Grafik ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Baseline", "Esmolol\ntek başına", "Digoksin\ntek başına", "Birlikte"]
    hr_values_stat = [patient.baseline_hr, solo_esmolol["mean_min_hr"], solo_digoxin["mean_min_hr"], combo["mean_min_hr"]]
    hr_values_circadapt = [r_esmolol["hr_base"], r_esmolol["hr_drug_model"], r_digoxin["hr_drug_model"], r_combo["hr_drug_model"]]
    colors = ["steelblue", "seagreen", "darkorange", "crimson"]

    axes[0].bar(range(len(labels)), hr_values_stat, color=colors)
    axes[0].axhline(50, color="black", linestyle="--", alpha=0.5, label="Bradikardi eşiği (50 bpm)")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Ortalama en düşük nabız (bpm)")
    axes[0].set_title("İstatistiksel PK/PD")
    axes[0].legend()

    axes[1].bar(range(len(labels)), hr_values_circadapt, color=colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Nabız (bpm)")
    axes[1].set_title("CircAdapt (gerçek kalp mekaniği)")

    plt.tight_layout()
    out_path = os.path.join(base, "compare_polypharmacy_sonucu.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nBAŞARILI! Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
