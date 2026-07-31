"""Onerilen sonraki adim (SUMMARY_1560.md Bolum 11): 1560-veride decoder/
encoder/predictor kapasitesini hidden_dim 128->256'ya cikarip 3 tohumla
tekrar egit, degerlendir. finetune_rollout BILINCLI OLARAK ATLANIYOR --
Bolum 10'da 1560-veride katkisiz/hafif zararli oldugu KANITLANDI.
Mimari TASARIMI degismiyor (hala delta-tahmin, EMA+stop-gradient, ayni
215-boyutlu state) -- sadece hidden_dim capraz-katman genisligi degisiyor.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from log_utils import LineFlushLogger

PY = os.path.join(".venv", "Scripts", "python.exe")
os.environ["PYTHONPATH"] = os.path.abspath("src")

DATA_DIR = "data/transient_dataset_large"
SEEDS = [0, 1, 2]
JEPA_EPOCHS = 200
DECODER_EPOCHS = 60
HIDDEN_DIM = 256

log = LineFlushLogger("logs/hidden256_deney.log")


def run(cmd, label):
    log.log(f"BASLIYOR: {label}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = "\n".join(proc.stdout.strip().splitlines()[-4:])
    log.log(f"BITTI ({dt:.1f}s, exit={proc.returncode}): {label}\n{tail}")
    if proc.returncode != 0:
        log.log(f"HATA: {proc.stderr[-2000:]}")
    return proc.returncode == 0, dt


results = {}
for seed in SEEDS:
    run_id = f"h256_seed{seed}"
    out_dir = f"models/dynamics_jepa_transient_1560run_{run_id}"
    log.log(f"===== {run_id} basliyor =====")

    ok, dt_a = run([PY, "-m", "worldmodel.learned_dynamics.train_jepa_transient",
                     "--data", DATA_DIR, "--out-dir", out_dir,
                     "--epochs", str(JEPA_EPOCHS), "--hidden-dim", str(HIDDEN_DIM),
                     "--seed", str(seed)], f"{run_id}::jepa")
    if not ok:
        results[run_id] = {"status": "jepa_basarisiz"}
        continue

    ok, dt_c = run([PY, "-m", "worldmodel.learned_dynamics.train_decoder_transient",
                     "--data", DATA_DIR, "--model-dir", out_dir,
                     "--epochs", str(DECODER_EPOCHS), "--seed", str(seed)], f"{run_id}::decoder")
    if not ok:
        results[run_id] = {"status": "decoder_basarisiz", "dt_a": dt_a}
        continue

    results[run_id] = {"status": "tamam", "dt_a": dt_a, "dt_c": dt_c, "out_dir": out_dir}
    log.log(f"===== {run_id} TAMAMLANDI =====")
    with open("logs/hidden256_deney_sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

log.log("TUM HIDDEN256 KOSUMLARI BITTI")
with open("logs/hidden256_deney_sonuclar.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log.close()
