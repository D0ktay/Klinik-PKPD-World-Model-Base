"""ADIM 3 -- 3 tohum x (eski 260-veri, yeni 1560-veri) tam egitim.
Gecici surucu betigi, proje kaynagina DAHIL DEGIL. logs/retrain_1560_*.log'a
satir-bazli flush'li, zaman damgali yazar."""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "scripts")
from log_utils import LineFlushLogger

PY = os.path.join(".venv", "Scripts", "python.exe")
ROOT = os.path.abspath(".")
os.environ["PYTHONPATH"] = os.path.join(ROOT, "src")

DATASETS = [
    ("260data", "data/transient_dataset_large_backup_260"),
    ("1560data", "data/transient_dataset_large"),
]
SEEDS = [0, 1, 2]

JEPA_EPOCHS = 200
FINETUNE_EPOCHS = 300
FINETUNE_HORIZON = 16
DECODER_EPOCHS = 60

log = LineFlushLogger("logs/retrain_1560_run.log")


def run(cmd, label):
    log.log(f"BASLIYOR: {label} :: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
    log.log(f"BITTI ({dt:.1f}s, exit={proc.returncode}): {label}\n--- son satirlar ---\n{tail}")
    if proc.returncode != 0:
        log.log(f"HATA STDERR ({label}):\n{proc.stderr[-3000:]}")
    return proc.returncode == 0, dt


results = {}
t_start_all = time.time()

for data_label, data_dir in DATASETS:
    for seed in SEEDS:
        run_id = f"{data_label}_seed{seed}"
        out_dir = f"models/dynamics_jepa_transient_1560run_{run_id}"
        log.log(f"===== KOSUM BASLIYOR: {run_id} (data={data_dir}, out={out_dir}) =====")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        ok, dt_a = run([PY, "-m", "worldmodel.learned_dynamics.train_jepa_transient",
                         "--data", data_dir, "--out-dir", out_dir,
                         "--epochs", str(JEPA_EPOCHS), "--seed", str(seed)], f"{run_id}::3a_train_jepa")
        if not ok:
            results[run_id] = {"status": "3a_basarisiz"}
            continue

        # 3a sonrasi snapshot -- finetune_rollout yerinde eziyor, geri donulebilir olsun
        snapshot_dir = out_dir + "_stage3a_snapshot"
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        shutil.copytree(out_dir, snapshot_dir)
        log.log(f"{run_id}: 3a snapshot alindi -> {snapshot_dir}")

        ok, dt_b = run([PY, "-m", "worldmodel.learned_dynamics.finetune_rollout",
                         "--data", data_dir, "--model-dir", out_dir,
                         "--epochs", str(FINETUNE_EPOCHS), "--rollout-horizon", str(FINETUNE_HORIZON)],
                        f"{run_id}::3b_finetune_rollout")
        finetune_ok = ok

        ok, dt_c = run([PY, "-m", "worldmodel.learned_dynamics.train_decoder_transient",
                         "--data", data_dir, "--model-dir", out_dir,
                         "--epochs", str(DECODER_EPOCHS), "--seed", str(seed)], f"{run_id}::3c_train_decoder")
        if not ok:
            results[run_id] = {"status": "3c_basarisiz", "dt_a": dt_a, "dt_b": dt_b, "finetune_ok": finetune_ok}
            continue

        results[run_id] = {"status": "tamam", "dt_a": dt_a, "dt_b": dt_b, "dt_c": dt_c,
                            "finetune_ok": finetune_ok, "out_dir": out_dir, "data_dir": data_dir}
        log.log(f"===== KOSUM TAMAMLANDI: {run_id} (3a={dt_a:.0f}s 3b={dt_b:.0f}s 3c={dt_c:.0f}s) =====")

        with open("logs/retrain_1560_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

total_dt = time.time() - t_start_all
log.log(f"TUM KOSUMLAR BITTI. Toplam sure: {total_dt/60:.1f} dk")
with open("logs/retrain_1560_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log.close()
