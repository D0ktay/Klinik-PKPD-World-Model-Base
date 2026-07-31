"""ADIM 1 -- On-dogrulamalar (kod yazmadan once). Gecici, proje kaynagina
DAHIL DEGIL. logs/retrain_1560_*.log'a sonuc yazar."""
import sys
import numpy as np

sys.path.insert(0, "src")

NPZ_PATH = "data/transient_dataset_large/dataset.npz"
EXPECTED_TEST = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129]
EXPECTED_VAL = [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128]

lines = []


def log(msg):
    print(msg)
    lines.append(msg)


d = np.load(NPZ_PATH, allow_pickle=True)
n_rows = len(d["patient_id"])
log(f"Toplam satir: {n_rows}")

# 1.2a NaN/Inf taramasi -- tum sayisal alanlar
numeric_fields = [
    "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp", "baseline_dbp",
    "baseline_spo2", "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
    "ef", "co", "edv", "esv", "current_hr", "conc_mg_L", "frame_idx", "trajectory_id", "patient_id",
]
nan_report = {}
for f in numeric_fields:
    arr = np.asarray(d[f], dtype=np.float64)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan or n_inf:
        nan_report[f] = (n_nan, n_inf)
if nan_report:
    log(f"NaN/Inf BULUNDU: {nan_report}")
else:
    log("NaN/Inf taramasi: TEMIZ (skaler alanlarda hic NaN/Inf yok)")

# traj_p / traj_v icin de tara (2D dizi olabilir)
for f in ["traj_p", "traj_v"]:
    arr = np.stack(d[f]).astype(np.float64)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    log(f"{f}: shape={arr.shape}  NaN={n_nan}  Inf={n_inf}")
    all_100 = all(len(row) == 100 for row in d[f])
    log(f"{f}: tum satirlar 100 nokta mi: {all_100}")

# conc_mg_L makul aralikta mi
conc = np.asarray(d["conc_mg_L"], dtype=np.float64)
log(f"conc_mg_L: min={conc.min():.4f} max={conc.max():.4f} negatif_sayisi={(conc<0).sum()}")

# 1.2d split dogrulama (bir kez daha)
patient_split = {}
for pid, sp in zip(d["patient_id"], d["split"]):
    patient_split[int(pid)] = str(sp)
test_ok = all(patient_split.get(i) == "test" for i in EXPECTED_TEST)
val_ok = all(patient_split.get(i) == "val" for i in EXPECTED_VAL)
new_ok = all(sp == "train" for pid, sp in patient_split.items() if pid >= 130)
log(f"13 test hastasi hala 'test' mi: {test_ok}")
log(f"13 val hastasi hala 'val' mi: {val_ok}")
log(f"i>=130 hepsi 'train' mi: {new_ok}")
log(f"SPLIT DOGRULAMA: {'GECTI' if (test_ok and val_ok and new_ok) else 'BASARISIZ'}")

# 1.2 truncated trajectory sayisi
import collections
tid_trunc = {}
for tid, tr in zip(d["trajectory_id"], d["truncated"]):
    tid_trunc[int(tid)] = bool(tr) or tid_trunc.get(int(tid), False)
n_trunc = sum(tid_trunc.values())
log(f"Kesilen (truncated) trajectory sayisi: {n_trunc} / {len(tid_trunc)}")

frames_per_traj = collections.Counter()
for tid in d["trajectory_id"]:
    frames_per_traj[int(tid)] += 1
dist = collections.Counter(frames_per_traj.values())
log(f"Trajectory basina kare sayisi dagilimi: {dict(dist)}")

with open("logs/adim1_dogrulama_sonucu.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nYazildi: logs/adim1_dogrulama_sonucu.txt")
