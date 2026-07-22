"""
HR vs SBP Etki Zamanlaması Ayrışması (Keo Modeli)
====================================================

Faz 5: nabız (HR) ve tansiyon (SBP) etkisi artık aynı, gecikmesiz
plazma konsantrasyonundan değil, kendi keo (etki bölgesi denge hızı)
değerleriyle filtrelenmiş AYRI eğrilerden hesaplanıyor. Bu script,
esmolol için bu iki eğrinin (plazma konsantrasyonuna göre) farklı
zamanlarda tepe yaptığını görsel olarak gösterir.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from worldmodel.pk import plasma_concentration
from worldmodel.pd import effect_compartment_concentration, emax_effect


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))

    patient = patients["hasta_a"]
    drug = drugs["beta_bloker"]

    t = np.linspace(0, 0.5, 1000)  # ilk 30 dakika -- esmolol çok hızlı
    conc = plasma_concentration(
        t, drug.dose_mg, drug.ka, drug.ke_mean, patient.weight_kg, drug.vd_per_kg,
        dose_mg_per_kg=drug.dose_mg_per_kg,
    )

    ce_hr = effect_compartment_concentration(conc, drug.keo_hr, t)
    ce_sbp = effect_compartment_concentration(conc, drug.keo_sbp, t)

    effect_hr = emax_effect(ce_hr, drug.ec50)
    effect_sbp = emax_effect(ce_sbp, drug.ec50)

    t_peak_conc = t[np.argmax(conc)]
    t_peak_hr = t[np.argmax(effect_hr)]
    t_peak_sbp = t[np.argmax(effect_sbp)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t * 60, conc, color="gray", linestyle="--", label="Plazma konsantrasyonu")
    axes[0].plot(t * 60, ce_hr, color="steelblue", label=f"Etki bölgesi (HR, keo={drug.keo_hr:.0f}/sa)")
    axes[0].plot(t * 60, ce_sbp, color="crimson", label=f"Etki bölgesi (SBP, keo={drug.keo_sbp:.0f}/sa)")
    axes[0].axvline(t_peak_conc * 60, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Zaman (dakika)")
    axes[0].set_ylabel("Konsantrasyon (mg/L)")
    axes[0].set_title("Plazma vs Etki Bölgesi Konsantrasyonu")
    axes[0].legend(fontsize=8)

    axes[1].plot(t * 60, effect_hr, color="steelblue", label=f"Nabız etkisi (pik t={t_peak_hr*60:.1f} dk)")
    axes[1].plot(t * 60, effect_sbp, color="crimson", label=f"Tansiyon etkisi (pik t={t_peak_sbp*60:.1f} dk)")
    axes[1].axvline(t_peak_hr * 60, color="steelblue", linestyle=":", alpha=0.5)
    axes[1].axvline(t_peak_sbp * 60, color="crimson", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Zaman (dakika)")
    axes[1].set_ylabel("Etki oranı (0-1)")
    axes[1].set_title(f"{drug.display_name} — Nabız vs Tansiyon Etki Zamanlaması")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(base, "keo_effect_separation_sonucu.png")
    plt.savefig(out_path, dpi=150)

    print(f"BAŞARILI! Grafik kaydedildi: {out_path}")
    print(f"Plazma pik zamanı:   {t_peak_conc*60:.2f} dk")
    print(f"Nabız etkisi piki:   {t_peak_hr*60:.2f} dk (keo_hr={drug.keo_hr}/sa, daha hızlı denge)")
    print(f"Tansiyon etkisi piki:{t_peak_sbp*60:.2f} dk (keo_sbp={drug.keo_sbp}/sa, daha yavaş denge)")


if __name__ == "__main__":
    main()
