"""
Elektrolit/Lab Verisinin Kalp Üzerindeki Etkisi
============================================================

Potasyum ve kalsiyum, "sadece kilo" ötesinde gerçek hasta verisinin
simülasyona katıldığı yeni bir boyut: ilaçtan TAMAMEN bağımsız olarak,
hastanın kendi lab değerleri kalbin elektriksel/mekanik davranışını
doğrudan etkiliyor.

  - Hiperkalemi (K>5.0 mEq/L) -> AV iletim gecikmesi (Timings.c_tau_av1) artar
  - Hipokalsemi/hiperkalsemi (Ca normal aralığın dışında) -> kontraktilite
    (Patch.Sf_act) azalır/artar

Bu script, aynı "ilaçsız baseline" kalbi üç farklı hasta lab profiliyle
karşılaştırır: normal, hiperkalemik, hipokalsemik.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients
from integrate_drug_with_circadapt import run_baseline, lv_pressure_volume

PATIENT_KEYS = ["hasta_a", "hasta_c_hiperkalemi", "hasta_d_hipokalsemi"]
COLORS = {"hasta_a": "steelblue", "hasta_c_hiperkalemi": "crimson", "hasta_d_hipokalsemi": "darkorange"}


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    print(f"{'Hasta':<35} {'K (mEq/L)':<12} {'Ca (mg/dL)':<12} {'AV gecikmesi (ms)':<20} {'LV pik (mmHg)'}")
    for key in PATIENT_KEYS:
        patient = patients[key]
        model = run_baseline(patient)
        t, p, v = lv_pressure_volume(model)
        tau_av_ms = model["Timings"]["tau_av"][0] * 1000

        print(f"{patient.name:<35} {patient.potassium_mEqL:<12.2f} {patient.calcium_mgdL:<12.2f} "
              f"{tau_av_ms:<20.1f} {p.max():.1f}")

        axes[0].plot(t, p, color=COLORS[key], label=f"{patient.name}\n(AV gecikmesi={tau_av_ms:.0f}ms)")
        axes[1].plot(v, p, color=COLORS[key], label=patient.name)

    axes[0].set_xlabel("Zaman (ms)")
    axes[0].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[0].set_title("LV Basıncı — Elektrolit Durumuna Göre (ilaçsız)")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Sol karıncık hacmi (mL)")
    axes[1].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[1].set_title("PV Loop — Elektrolit Durumuna Göre")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(base, "compare_electrolyte_effects_sonucu.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nBAŞARILI! Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
