"""Görselleştirme yardımcıları — matplotlib ile 'spaghetti plot' üretir."""

import matplotlib.pyplot as plt
from .simulation import SimulationResult
from .patient import Patient, Drug


def plot_results(result: SimulationResult, patient: Patient, drug: Drug,
                  save_path: str | None = None, label: str | None = None):
    """
    label: başlıkta "{ilaç adı} {doz}mg" yerine kullanılacak metin --
        polifarmasi (N ilaç) senaryosunda tek bir Drug nesnesi başlığı
        yeterince temsil etmediği için (bkz. streamlit_app.py), çağıran
        birden fazla ilacın isimlerini birleştirip buradan verebilir.
        None ise (tek ilaç, mevcut/eski davranış) drug.display_name ve
        drug.dose_mg'den otomatik üretilir.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_runs = result.hr_runs.shape[0]
    for i in range(n_runs):
        axes[0].plot(result.t, result.hr_runs[i], color="steelblue", alpha=0.05)
        axes[1].plot(result.t, result.sbp_runs[i], color="indianred", alpha=0.05)

    axes[0].plot(result.t, result.hr_runs.mean(axis=0), color="navy", linewidth=2, label="Ortalama tepki")
    axes[1].plot(result.t, result.sbp_runs.mean(axis=0), color="darkred", linewidth=2, label="Ortalama tepki")

    axes[0].axhline(patient.baseline_hr, color="gray", linestyle="--", label="Bazal nabız")
    axes[1].axhline(patient.baseline_sbp, color="gray", linestyle="--", label="Bazal tansiyon")

    drug_label = label if label is not None else f"{drug.display_name} {drug.dose_mg}mg"
    axes[0].set_title(f"Nabız Tepkisi — {patient.name} ({patient.weight_kg}kg, {patient.height_cm}cm)\n"
                       f"{n_runs} sanal deneme, {drug_label}")
    axes[0].set_xlabel("Zaman (saat)")
    axes[0].set_ylabel("Nabız (bpm)")
    axes[0].legend()

    axes[1].set_title("Sistolik Tansiyon Tepkisi")
    axes[1].set_xlabel("Zaman (saat)")
    axes[1].set_ylabel("Tansiyon (mmHg)")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grafik kaydedildi: {save_path}")
    return fig
