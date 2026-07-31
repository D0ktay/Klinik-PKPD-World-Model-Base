"""6 kosumun tumu icin diagnostics topla, 260 vs 1560 karsilastirmasini
(ortalama+-std, 3 tohum uzerinden) uret. logs/adim4_sonuclar.json'a yazar."""
import json
import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from adim4_degerlendirme import full_diagnostics, example_patient_hr_table
from worldmodel.learned_dynamics.state_repr import SCALAR_TARGET_FIELDS

RUNS = [
    ("260data", 0, "data/transient_dataset_large_backup_260", "models/dynamics_jepa_transient_1560run_260data_seed0"),
    ("260data", 1, "data/transient_dataset_large_backup_260", "models/dynamics_jepa_transient_1560run_260data_seed1"),
    ("260data", 2, "data/transient_dataset_large_backup_260", "models/dynamics_jepa_transient_1560run_260data_seed2"),
    ("1560data", 0, "data/transient_dataset_large", "models/dynamics_jepa_transient_1560run_1560data_seed0"),
    ("1560data", 1, "data/transient_dataset_large", "models/dynamics_jepa_transient_1560run_1560data_seed1"),
    ("1560data", 2, "data/transient_dataset_large", "models/dynamics_jepa_transient_1560run_1560data_seed2"),
]

all_results = {}
for data_label, seed, data_dir, model_dir in RUNS:
    key = f"{data_label}_seed{seed}"
    print(f"Isleniyor: {key}...", flush=True)
    all_results[key] = full_diagnostics(data_dir, model_dir, key)

# eski (orijinal, tek referans) model de dahil et -- karsilastirma icin capa
all_results["260data_orijinal_referans"] = full_diagnostics(
    "data/transient_dataset_large_backup_260",
    "models/dynamics_jepa_transient_large_backup_260data",
    "260data_orijinal_referans")


def agg_stat(runs, path_fn):
    vals = [path_fn(all_results[r]) for r in runs]
    return {"ortalama": float(np.mean(vals)), "std": float(np.std(vals)), "degerler": vals}


summary = {"per_run": all_results, "karsilastirma": {}}
for field in SCALAR_TARGET_FIELDS:
    summary["karsilastirma"][field] = {
        "pooled_r2_260": agg_stat(["260data_seed0", "260data_seed1", "260data_seed2"],
                                    lambda r: r["pooled_r2"][field]),
        "pooled_r2_1560": agg_stat(["1560data_seed0", "1560data_seed1", "1560data_seed2"],
                                     lambda r: r["pooled_r2"][field]),
        "model_mae_260": agg_stat(["260data_seed0", "260data_seed1", "260data_seed2"],
                                    lambda r: r["model_mae_havuzlanmis"][field]),
        "model_mae_1560": agg_stat(["1560data_seed0", "1560data_seed1", "1560data_seed2"],
                                     lambda r: r["model_mae_havuzlanmis"][field]),
        "persistence_baseline_mae_260": all_results["260data_seed0"]["persistence_baseline_mae"][field],
        "persistence_baseline_mae_1560": all_results["1560data_seed0"]["persistence_baseline_mae"][field],
        "var_oran_260": agg_stat(["260data_seed0", "260data_seed1", "260data_seed2"],
                                   lambda r: r["variance_ratio"][field]["ortalama_oran"]),
        "var_oran_1560": agg_stat(["1560data_seed0", "1560data_seed1", "1560data_seed2"],
                                    lambda r: r["variance_ratio"][field]["ortalama_oran"]),
        "yon_hatasi_oran_260": agg_stat(["260data_seed0", "260data_seed1", "260data_seed2"],
                                          lambda r: r["direction_error"][field]["oran"]),
        "yon_hatasi_oran_1560": agg_stat(["1560data_seed0", "1560data_seed1", "1560data_seed2"],
                                           lambda r: r["direction_error"][field]["oran"]),
    }

# ornek hasta HR tablolari -- test hastalarindan ilk 3'u (9, 19, 29)
example_patients = [9, 19, 29]
summary["ornek_hasta_hr"] = {
    "260data_seed0": example_patient_hr_table("data/transient_dataset_large_backup_260",
                                               "models/dynamics_jepa_transient_1560run_260data_seed0",
                                               example_patients),
    "1560data_seed0": example_patient_hr_table("data/transient_dataset_large",
                                                "models/dynamics_jepa_transient_1560run_1560data_seed0",
                                                example_patients),
    "260data_orijinal_referans": example_patient_hr_table("data/transient_dataset_large_backup_260",
                                                            "models/dynamics_jepa_transient_large_backup_260data",
                                                            example_patients),
}

with open("logs/adim4_sonuclar.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n=== KARSILASTIRMA OZETI (pooled R2, ortalama +- std / 3 tohum) ===")
for field in SCALAR_TARGET_FIELDS:
    c = summary["karsilastirma"][field]
    print(f"{field:5s}  260-veri: {c['pooled_r2_260']['ortalama']:.3f} +- {c['pooled_r2_260']['std']:.3f}"
          f"   |   1560-veri: {c['pooled_r2_1560']['ortalama']:.3f} +- {c['pooled_r2_1560']['std']:.3f}")

print("\nYazildi: logs/adim4_sonuclar.json")
