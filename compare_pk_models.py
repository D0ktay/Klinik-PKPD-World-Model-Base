"""
Tek-Kompartmanlı vs İki-Kompartmanlı PK Modeli Karşılaştırması
=================================================================

Esmolol için ikisi de aynı literatür yarı ömürlerinden (dağılım ~2 dk,
eliminasyon ~9 dk) türetilmiş, ama farklı matematiksel formülasyonlar
kullanan iki modelin ürettiği plazma konsantrasyon eğrilerini üst üste
çizer.

Fark neden var: tek-kompartmanlı model bir ORAL/absorpsiyon fazı (ka)
varsayar (t=0'da C=0, sonra yükselir), iki-kompartmanlı model gerçek bir
IV BOLUS varsayar (t=0'da C=dose/Vc ile pik yapar, sonra iki farklı hızda
-- önce hızlı dağılım, sonra yavaş eliminasyon -- düşer). Esmolol
klinikte IV bolus olarak verildiği için ikinci model fizyolojik olarak
daha doğru bir temsildir; birincisi (ka ile) sadece pratik bir
yaklaşıklıktı (bkz. README.md > Veri Kaynakları).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from worldmodel.pk import plasma_concentration, plasma_concentration_two_compartment


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))

    patient = patients["hasta_a"]
    drug = drugs["beta_bloker"]

    t = np.linspace(0, 1.0, 500)  # ilk 1 saat -- esmolol çok kısa etkili

    conc_1c = plasma_concentration(
        t, drug.dose_mg, drug.ka, drug.ke_mean, patient.weight_kg, drug.vd_per_kg,
        dose_mg_per_kg=drug.dose_mg_per_kg,
    )
    conc_2c = plasma_concentration_two_compartment(
        t, drug.dose_mg, drug.k10, drug.k12, drug.k21, drug.vd_central_per_kg,
        dose_mg_per_kg=drug.dose_mg_per_kg, weight_kg=patient.weight_kg,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 60, conc_1c, label="Tek-kompartmanlı (oral/absorpsiyon yaklaşıklığı, ka)",
            color="steelblue")
    ax.plot(t * 60, conc_2c, label="İki-kompartmanlı (gerçek IV bolus)",
            color="crimson")
    ax.set_xlabel("Zaman (dakika)")
    ax.set_ylabel("Plazma konsantrasyonu (mg/L)")
    ax.set_title(f"{drug.display_name} — PK Modeli Karşılaştırması ({patient.name})")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(base, "compare_pk_models_sonucu.png")
    plt.savefig(out_path, dpi=150)

    print(f"BAŞARILI! Grafik kaydedildi: {out_path}")
    print(f"Tek-kompartmanlı  pik: {conc_1c.max():.4f} mg/L, t={t[np.argmax(conc_1c)]*60:.2f} dk")
    print(f"İki-kompartmanlı  pik: {conc_2c.max():.4f} mg/L, t={t[np.argmax(conc_2c)]*60:.2f} dk (t=0, IV bolus)")


if __name__ == "__main__":
    main()
