"""Finetune_rollout'un katkisini olc: stage3a-snapshot (ince-ayar ONCESI,
kopya dizine decoder egitilerek) vs final (ince-ayar SONRASI) checkpoint'i,
AYNI protokolle (rollout_evaluate.py) karsilastir. Orijinal snapshot/final
checkpoint dosyalarina DOKUNULMAZ -- sadece KOPYA dizinlere yazilir."""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from log_utils import LineFlushLogger
from adim4_degerlendirme import full_diagnostics
from worldmodel.learned_dynamics.state_repr import SCALAR_TARGET_FIELDS

PY = os.path.join(".venv", "Scripts", "python.exe")
os.environ["PYTHONPATH"] = os.path.abspath("src")

log = LineFlushLogger("logs/finetune_karsilastirma.log")

RUNS = [
    ("260data", 0, "data/transient_dataset_large_backup_260"),
    ("260data", 1, "data/transient_dataset_large_backup_260"),
    ("260data", 2, "data/transient_dataset_large_backup_260"),
    ("1560data", 0, "data/transient_dataset_large"),
    ("1560data", 1, "data/transient_dataset_large"),
    ("1560data", 2, "data/transient_dataset_large"),
]

DECODER_EPOCHS = 60  # 3c ile AYNI -- adil karsilastirma icin

results = {}

for data_label, seed, data_dir in RUNS:
    run_id = f"{data_label}_seed{seed}"
    snapshot_src = f"models/dynamics_jepa_transient_1560run_{run_id}_stage3a_snapshot"
    final_dir = f"models/dynamics_jepa_transient_1560run_{run_id}"
    eval_copy = f"models/_finetune_eval_{run_id}_pre"  # KOPYA, gecici

    log.log(f"===== {run_id}: stage3a-snapshot -> KOPYA dizine decoder egitimi =====")
    if os.path.exists(eval_copy):
        shutil.rmtree(eval_copy)
    shutil.copytree(snapshot_src, eval_copy)  # orijinal snapshot'a DOKUNULMADI, kopyalandi

    t0 = time.time()
    proc = subprocess.run(
        [PY, "-m", "worldmodel.learned_dynamics.train_decoder_transient",
         "--data", data_dir, "--model-dir", eval_copy,
         "--epochs", str(DECODER_EPOCHS), "--seed", str(seed)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    ok = proc.returncode == 0
    log.log(f"{run_id}: decoder egitimi {'basarili' if ok else 'BASARISIZ'} ({dt:.1f}s)")
    if not ok:
        log.log(f"HATA: {proc.stderr[-2000:]}")
        results[run_id] = {"status": "decoder_egitimi_basarisiz"}
        continue

    log.log(f"{run_id}: ONCE (pre-finetune) degerlendirme...")
    pre = full_diagnostics(data_dir, eval_copy, f"{run_id}_pre_finetune")

    log.log(f"{run_id}: SONRA (post-finetune, final) degerlendirme...")
    post = full_diagnostics(data_dir, final_dir, f"{run_id}_post_finetune")

    results[run_id] = {"status": "tamam", "pre": pre, "post": post}

    with open("logs/finetune_karsilastirma_sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # kopya dizini temizle -- sadece gecici degerlendirme icindi
    shutil.rmtree(eval_copy)
    log.log(f"{run_id}: TAMAMLANDI, gecici kopya temizlendi")

log.log("TUM KARSILASTIRMALAR BITTI")
log.close()
print("Yazildi: logs/finetune_karsilastirma_sonuclar.json")
