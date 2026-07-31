"""
N-İlaç Polifarmasi Karşılaştırması (ADIM 7.2)
================================================

compare_polypharmacy.py'nin (Esmolol+Digoksin, N=2'ye özel) N ilaca
genellenmiş versiyonu -- N=5 ilaçlık (karma yönlü: 3 negatif kronotrop +
2 pozitif kronotrop) bir kombinasyonu hem istatistiksel PK/PD (additive +
Loewe) hem gerçek CircAdapt kalp mekaniği tarafında çalıştırıp karşılaştırır.
Amaç: uçtan uca N-ilaç zincirinin, Streamlit dışında da (örn. otomatik bir
betik/CI bağlamında) doğru çalıştığını göstermek.
"""

import os
import sys
from dataclasses import replace

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from worldmodel.simulation import run_polypharmacy_simulation, run_polypharmacy_simulation_loewe, summarize
from integrate_drug_with_circadapt import run_comparison, run_polypharmacy_comparison


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))
    patient = patients["hasta_a"]

    # N=5, karma yönlü: esmolol/digoxin/nicardipine nabzı düşürür,
    # dobutamine/nitroprussid tarafı farklı yönde etki eder (nicardipine
    # zaten refleks taşikardi ile HAFİF artırıcı) -- ADR-4'ün gruplama+fark
    # mantığını gerçek bir N=5 senaryosunda egzersiz eder.
    combo = [
        replace(drugs["beta_bloker"], dose_mg=15.0, dose_mg_per_kg=None),
        drugs["digoxin"],
        replace(drugs["nicardipine"], dose_mg=1.0, dose_mg_per_kg=None),
        replace(drugs["dobutamine"], dose_mg=3.0, dose_mg_per_kg=None),
        drugs["vazodilator"],
    ]

    print(f"Hasta: {patient.name}")
    for i, d in enumerate(combo):
        print(f"İlaç {i+1}: {d.display_name} ({d.dose_mg:.2f} mg) [{d.drug_class}], emax_hr={d.emax_hr}")

    print("\n=== İstatistiksel PK/PD (Monte Carlo, N=5) ===")
    additive = summarize(run_polypharmacy_simulation(patient, combo, n_realizations=500, seed=1))
    print(f"Additive (toplamsal):  ort. en düşük nabız={additive['mean_min_hr']:.1f} bpm, "
          f"bradikardi riski=%{additive['pct_bradycardia_risk']:.1f}")
    try:
        loewe = summarize(run_polypharmacy_simulation_loewe(patient, combo, n_realizations=500, seed=1))
        print(f"Loewe (gruplama+fark): ort. en düşük nabız={loewe['mean_min_hr']:.1f} bpm, "
              f"bradikardi riski=%{loewe['pct_bradycardia_risk']:.1f}")
    except ValueError as e:
        loewe = None
        print(f"Loewe hesaplanamadı: {e}")

    print("\n=== CircAdapt (gerçek kalp mekaniği, N=5) ===")
    r_combo = run_polypharmacy_comparison(patient, combo)
    if r_combo.get("instability_triggered") or r_combo.get("av_block_triggered"):
        print(f"UYARI: sayısal kararlılık sınırı aşıldı "
              f"(unstable_parameter={r_combo.get('unstable_parameter')}, "
              f"av_block_triggered={r_combo.get('av_block_triggered')}) -- "
              f"CircAdapt hiç çalıştırılmadı, hr_drug_model={r_combo['hr_drug_model']:.1f} bpm "
              "(sadece istatistiksel tahmin).")
    else:
        print(f"Baseline -> Kombine: {r_combo['hr_base']:.1f} -> {r_combo['hr_drug_model']:.1f} bpm")

    # --- Grafik ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Baseline", "Additive\n(N=5)"] + (["Loewe\n(N=5)"] if loewe else [])
    hr_values_stat = [patient.baseline_hr, additive["mean_min_hr"]] + ([loewe["mean_min_hr"]] if loewe else [])
    colors = ["steelblue", "seagreen", "darkorange"][:len(labels)]

    axes[0].bar(range(len(labels)), hr_values_stat, color=colors)
    axes[0].axhline(50, color="black", linestyle="--", alpha=0.5, label="Bradikardi eşiği (50 bpm)")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Ortalama en düşük nabız (bpm)")
    axes[0].set_title("İstatistiksel PK/PD (N=5, karma yönlü)")
    axes[0].legend()

    circadapt_labels = ["Baseline", "CircAdapt\n(N=5 birlikte)"]
    circadapt_values = [r_combo["hr_base"], r_combo["hr_drug_model"]]
    axes[1].bar(range(len(circadapt_labels)), circadapt_values, color=["steelblue", "crimson"])
    axes[1].set_xticks(range(len(circadapt_labels)))
    axes[1].set_xticklabels(circadapt_labels)
    axes[1].set_ylabel("Nabız (bpm)")
    axes[1].set_title("CircAdapt (gerçek kalp mekaniği, N=5)")

    plt.tight_layout()
    out_path = os.path.join(base, "compare_n_drug_polypharmacy_sonucu.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nBAŞARILI! Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
