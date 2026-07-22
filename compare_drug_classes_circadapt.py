"""
Üç İlaç Sınıfının CircAdapt Üzerindeki Etkisini Karşılaştırma
===============================================================

integrate_drug_with_circadapt.py'daki apply_drug_effect_to_circadapt
fonksiyonu artık ilaç sınıfına göre (beta_blocker / vasodilator /
positive_inotrope) FARKLI bir CircAdapt mekanizmasına bağlanıyor. Bu
script üçünü aynı hasta üzerinde sırayla çalıştırıp yan yana
karşılaştırır: LV basıncı, LVEDV ve kalp hızı değişimi.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from integrate_drug_with_circadapt import run_comparison, run_baseline, lv_pressure_volume

DRUG_KEYS = ["beta_bloker", "nicardipine", "dobutamine"]
COLORS = {"beta_bloker": "crimson", "nicardipine": "seagreen", "dobutamine": "darkorange"}


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))
    patient = patients["hasta_a"]

    print(f"Hasta: {patient.name}\n")

    baseline_model = run_baseline(patient)
    t_base, p_base, v_base = lv_pressure_volume(baseline_model)
    hr_base = 60.0 / baseline_model["General"]["t_cycle"]
    lvedv_base = v_base.max()
    p_peak_base = p_base.max()

    results = {}
    for key in DRUG_KEYS:
        drug = drugs[key]
        print(f"--- {drug.display_name} (drug_class={drug.drug_class}) çalıştırılıyor ---")
        r = run_comparison(patient, drug)
        results[key] = r
        print(f"  Nabız: {hr_base:.1f} -> {r['hr_drug_model']:.1f} bpm")
        print(f"  LV pik basıncı: {p_peak_base:.1f} -> {r['p_drug'].max():.1f} mmHg")
        print(f"  LVEDV: {lvedv_base:.1f} -> {r['v_drug'].max():.1f} mL\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) LV basıncı - zaman
    axes[0].plot(t_base, p_base, color="steelblue", linewidth=2, label=f"Baseline ({hr_base:.0f} bpm)")
    for key in DRUG_KEYS:
        r = results[key]
        drug = drugs[key]
        axes[0].plot(r["t_drug"], r["p_drug"], color=COLORS[key],
                     label=f"{drug.display_name} ({r['hr_drug_model']:.0f} bpm)")
    axes[0].set_xlabel("Zaman (ms)")
    axes[0].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[0].set_title("LV Basıncı")
    axes[0].legend(fontsize=8)

    # 2) LVEDV bar chart
    labels = ["Baseline"] + [drugs[k].display_name for k in DRUG_KEYS]
    lvedv_values = [lvedv_base] + [results[k]["v_drug"].max() for k in DRUG_KEYS]
    bar_colors = ["steelblue"] + [COLORS[k] for k in DRUG_KEYS]
    axes[1].bar(range(len(labels)), lvedv_values, color=bar_colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(["Baseline", "Beta-\nbloker", "Vazo-\ndilatör", "Pozitif\ninotrop"])
    axes[1].set_ylabel("LVEDV (mL)")
    axes[1].set_title("Diyastol Sonu Hacim")

    # 3) Kalp hızı bar chart
    hr_values = [hr_base] + [results[k]["hr_drug_model"] for k in DRUG_KEYS]
    axes[2].bar(range(len(labels)), hr_values, color=bar_colors)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(["Baseline", "Beta-\nbloker", "Vazo-\ndilatör", "Pozitif\ninotrop"])
    axes[2].set_ylabel("Kalp hızı (bpm)")
    axes[2].set_title("Kalp Hızı")

    plt.tight_layout()
    out_path = os.path.join(base, "compare_drug_classes_sonucu.png")
    plt.savefig(out_path, dpi=150)
    print(f"BAŞARILI! Grafik kaydedildi: {out_path}\n")

    # --- Beklenmedik ama fizyolojik olarak tutarlı bulgular ---
    print("=" * 70)
    print("NOTLAR (mülakat/sunum malzemesi):")
    print("=" * 70)

    dob = results["dobutamine"]
    if dob["p_drug"].max() <= p_peak_base and dob["v_drug"].max() < lvedv_base:
        print(
            "- Dobutamin (pozitif inotrop) kontraktiliteyi ARTIRIYOR, ama LV pik\n"
            "  basıncı baseline'a göre neredeyse değişmiyor/hafif düşüyor. Sebep:\n"
            "  eşlik eden taşikardi (nabız {:.0f}->{:.0f} bpm) diyastolik dolum\n"
            "  süresini kısaltıyor, LVEDV düşüyor ({:.1f}->{:.1f} mL) ve azalan\n"
            "  önyük, artan kontraktilitenin basınç üzerindeki etkisini büyük\n"
            "  ölçüde dengeliyor. 'Daha güçlü kasılma = daha yüksek basınç' gibi\n"
            "  basit bir ilişki değil -- hız ve önyük burada birbirine kenetli.".format(
                hr_base, dob["hr_drug_model"], lvedv_base, dob["v_drug"].max()
            )
        )

    bb = results["beta_bloker"]
    if bb["p_drug"].max() > p_peak_base:
        print(
            "- Beta-bloker (esmolol) kontraktiliteyi AZALTIYOR, ama LV pik basıncı\n"
            "  yine de artıyor ({:.1f}->{:.1f} mmHg). Sebep: nabız çok yavaşladığı\n"
            "  için ({:.0f}->{:.0f} bpm) diyastolik dolum süresi uzuyor, LVEDV\n"
            "  belirgin şekilde artıyor ({:.1f}->{:.1f} mL) ve Frank-Starling\n"
            "  mekanizması azalan kontraktiliteyi telafi ediyor.".format(
                p_peak_base, bb["p_drug"].max(), hr_base, bb["hr_drug_model"],
                lvedv_base, bb["v_drug"].max()
            )
        )

    nic = results["nicardipine"]
    print(
        "- Nikardipin (vazodilatör) kontraktiliteye hiç dokunmuyor (sadece\n"
        "  ArtVen sistemik direncini düşürüyor), yine de LV pik basıncı belirgin\n"
        "  şekilde düşüyor ({:.1f}->{:.1f} mmHg) -- saf bir afterload etkisi,\n"
        "  esmolol/dobutaminin kontraktilite/hız etkileriyle karıştırılmamalı.".format(
            p_peak_base, nic["p_drug"].max()
        )
    )


if __name__ == "__main__":
    main()
