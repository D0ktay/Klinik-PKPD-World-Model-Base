"""hidden_dim=256 (3 tohum) sonuclarini, hidden_dim=128 'sadece-3a' (finetune
YOK) tabaniyla karsilastir -- finetune_karsilastirma_sonuclar.json'daki
'pre' sonuclari zaten TAM OLARAK bu taban (3a + taze decoder, finetune yok).
"""
import json
import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from adim4_degerlendirme import full_diagnostics
from worldmodel.learned_dynamics.state_repr import SCALAR_TARGET_FIELDS

DATA_DIR = "data/transient_dataset_large"
SEEDS = [0, 1, 2]

h256_results = {}
for seed in SEEDS:
    model_dir = f"models/dynamics_jepa_transient_1560run_h256_seed{seed}"
    h256_results[f"seed{seed}"] = full_diagnostics(DATA_DIR, model_dir, f"h256_seed{seed}")

with open("logs/hidden256_diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(h256_results, f, indent=2, ensure_ascii=False)

# taban: hidden=128, finetune YOK (3a+taze decoder) -- finetune_karsilastirma'nin 'pre' sonuclari
baseline = json.load(open("logs/finetune_karsilastirma_sonuclar.json", encoding="utf-8"))


def agg_h256(field, key, subkey=None):
    vals = []
    for seed in SEEDS:
        v = h256_results[f"seed{seed}"][key] if subkey is None else h256_results[f"seed{seed}"][key][subkey]
        vals.append(v[field] if isinstance(v, dict) else v)
    return float(np.mean(vals)), float(np.std(vals))


def agg_baseline(field, key):
    vals = []
    for seed in SEEDS:
        vals.append(baseline[f"1560data_seed{seed}"]["pre"][key][field])
    return float(np.mean(vals)), float(np.std(vals))


print(f"{'metrik':6s} {'hidden128 R2':>18s} {'hidden256 R2':>18s} {'fark':>10s}")
for field in SCALAR_TARGET_FIELDS:
    b_m, b_s = agg_baseline(field, "pooled_r2")
    h_m, h_s = agg_h256(field, "pooled_r2")
    print(f"{field:6s} {b_m:7.3f}+-{b_s:.3f}      {h_m:7.3f}+-{h_s:.3f}      {h_m-b_m:+.3f}")

print()
print(f"{'metrik':6s} {'hidden128 MAE':>18s} {'hidden256 MAE':>18s} {'fark':>10s}")
for field in SCALAR_TARGET_FIELDS:
    b_m, b_s = agg_baseline(field, "model_mae_havuzlanmis")
    h_m, h_s = agg_h256(field, "model_mae_havuzlanmis")
    print(f"{field:6s} {b_m:9.4f}+-{b_s:.4f}   {h_m:9.4f}+-{h_s:.4f}   {h_m-b_m:+.4f}")

print()
print("Persistence baseline MAE (hedef bu -- gecmek istiyoruz):")
for field in SCALAR_TARGET_FIELDS:
    pb = h256_results["seed0"]["persistence_baseline_mae"][field]
    print(f"  {field}: {pb:.4f}")
