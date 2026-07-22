"""
Komorbidite Karşılaştırması — Aynı Doz, Farklı Kalp, Farklı Sonuç
=====================================================================

Faz 12: Şu ana kadar herkes "sağlıklı bazal" bir kalple başlıyordu.
Gerçek hastanelerde kalp yetmezliği, hipertansiyon gibi önceden var olan
durumlar ilaç tepkisini KÖKTEN değiştirir -- gerçek klinikte de tam bu
yüzden dozlar hastaya göre uyarlanır.

Bu script, AYNI esmolol dozunu üç farklı kalp üzerinde çalıştırır:
sağlıklı, sistolik kalp yetmezliği, kronik hipertansif.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from integrate_drug_with_circadapt import run_comparison

PATIENT_KEYS = ["hasta_a", "hasta_e_kalp_yetmezligi", "hasta_f_hipertansif"]
COLORS = {"hasta_a": "steelblue", "hasta_e_kalp_yetmezligi": "crimson", "hasta_f_hipertansif": "darkorange"}


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))
    drug = drugs["beta_bloker"]  # esmolol -- aynı doz, üç farklı hasta

    print(f"İlaç: {drug.display_name} ({drug.dose_mg_per_kg} mg/kg -- AYNI DOZ, ÜÇ FARKLI HASTA)\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    results = {}

    print(f"{'Hasta':<45} {'Baseline LV pik':<18} {'İlaçlı LV pik':<18} {'Baseline LVEDV':<18} {'İlaçlı LVEDV'}")
    for key in PATIENT_KEYS:
        patient = patients[key]
        r = run_comparison(patient, drug)
        results[key] = r
        print(f"{patient.name:<45} {r['p_base'].max():<18.1f} {r['p_drug'].max():<18.1f} "
              f"{r['v_base'].max():<18.1f} {r['v_drug'].max():.1f}")

        axes[0].plot(r["t_drug"], r["p_drug"], color=COLORS[key],
                     label=f"{patient.name}\n({r['hr_drug_model']:.0f} bpm)")
        axes[1].plot(r["v_drug"], r["p_drug"], color=COLORS[key], label=patient.name)

    print(f"\nAynı esmolol dozu, hastanın komorbiditesine göre TAMAMEN FARKLI hemodinamik sonuçlar üretiyor:")
    hf = results["hasta_e_kalp_yetmezligi"]
    htn = results["hasta_f_hipertansif"]
    healthy = results["hasta_a"]
    print(f"  - Sağlıklı hastada LVEDV artışı: {healthy['v_drug'].max() - healthy['v_base'].max():+.1f} mL")
    print(f"  - Kalp yetmezliği hastasında LVEDV artışı: {hf['v_drug'].max() - hf['v_base'].max():+.1f} mL "
          f"(zaten dilate olan bir ventrikül daha da geriliyor -- akut dekompansasyon riski)")
    print(f"  - Hipertansif hastada LV pik basıncı: {htn['p_drug'].max():.1f} mmHg "
          f"(yüksek ardyüke karşı hâlâ yüksek basınç gerekiyor, beta-bloker bunu tam telafi edemiyor)")

    axes[0].set_xlabel("Zaman (ms)")
    axes[0].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[0].set_title(f"LV Basıncı — Aynı Doz, Farklı Hasta ({drug.display_name})")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Sol karıncık hacmi (mL)")
    axes[1].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[1].set_title("PV Loop — Aynı Doz, Farklı Hasta")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(base, "compare_comorbidity_effects_sonucu.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nBAŞARILI! Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
