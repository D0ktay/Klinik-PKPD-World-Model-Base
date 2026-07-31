"""ADIM 4 -- Degerlendirme ve teshisler. Gecici surucu betigi, proje
kaynagina DAHIL DEGIL. Egitilmis model+veri kombinasyonlari icin:
  - R2 (havuzlanmis + son adim) -- rollout_evaluate icinden
  - persistence baseline MAE karsilastirmasi
  - ortalamaya-kacis teshisi: var(tahmin)/var(gercek), hasta-basina dagilim,
    yon hatasi sayimi
  - ornek hasta HR tablolari
Sonuclari logs/adim4_sonuclar.json'a yazar (SUMMARY raporu bunu okuyacak).
"""
import json
import sys
import numpy as np

sys.path.insert(0, "src")

from worldmodel.learned_dynamics.rollout_evaluate import rollout_evaluate, r_squared
from worldmodel.learned_dynamics.state_repr import SCALAR_TARGET_FIELDS


def persistence_baseline_mae(step0_true: dict) -> dict:
    """'Hicbir sey tahmin etme, t=0 degerini kopyala' -- ayni test setinde."""
    out = {}
    for field in SCALAR_TARGET_FIELDS:
        arr = step0_true[field]  # step, patient_id, traj_idx, frame0_true, step_true
        mae = float(np.mean(np.abs(arr[:, 3] - arr[:, 4])))
        out[field] = mae
    return out


def variance_ratio(records: dict) -> dict:
    """Her adimda var(tahmin)/var(gercek) -- hastalar arasi varyans. Adimlar
    uzerinden ortalama + son adim degeri raporlanir."""
    out = {}
    for field in SCALAR_TARGET_FIELDS:
        arr = records[field]  # step, patient_id, traj_idx, true, pred
        steps = sorted(set(arr[:, 0].astype(int)))
        ratios = []
        for step in steps:
            mask = arr[:, 0].astype(int) == step
            true_v, pred_v = arr[mask, 3], arr[mask, 4]
            var_true = np.var(true_v)
            var_pred = np.var(pred_v)
            ratio = float(var_pred / var_true) if var_true > 1e-9 else float("nan")
            ratios.append(ratio)
        last_step = max(steps)
        mask = arr[:, 0].astype(int) == last_step
        true_v, pred_v = arr[mask, 3], arr[mask, 4]
        var_true_last = np.var(true_v)
        ratio_last = float(np.var(pred_v) / var_true_last) if var_true_last > 1e-9 else float("nan")
        out[field] = {"ortalama_oran": float(np.nanmean(ratios)), "son_adim_oran": ratio_last}
    return out


def per_patient_distribution(records: dict) -> dict:
    """Hasta basina (patient_id) MAE ve R2 -- min/medyan/maks/std. Havuzlanmis
    R2'nin gizledigi 'baz hastalar cok iyi baz hastalar cok kotu' durumunu
    ortaya cikarir."""
    out = {}
    for field in SCALAR_TARGET_FIELDS:
        arr = records[field]
        patient_ids = np.unique(arr[:, 1].astype(int))
        maes, r2s = [], []
        for pid in patient_ids:
            mask = arr[:, 1].astype(int) == pid
            true_v, pred_v = arr[mask, 3], arr[mask, 4]
            maes.append(float(np.mean(np.abs(true_v - pred_v))))
            r2s.append(r_squared(true_v, pred_v))
        maes, r2s = np.array(maes), np.array(r2s)
        out[field] = {
            "n_hasta": len(patient_ids),
            "mae_min": float(np.min(maes)), "mae_medyan": float(np.median(maes)),
            "mae_maks": float(np.max(maes)), "mae_std": float(np.std(maes)),
            "r2_min": float(np.nanmin(r2s)), "r2_medyan": float(np.nanmedian(r2s)),
            "r2_maks": float(np.nanmax(r2s)), "r2_std": float(np.nanstd(r2s)),
        }
    return out


def direction_error_count(records: dict, step0_true: dict, eps: float = 1e-3) -> dict:
    """Her trajectory icin, GERCEK t=0 baslangicindan son adima GERCEK
    degisim yonu ile TAHMIN edilen degisim yonu ZIT mi -- sayim ve oran.
    (Rollout t=0'da GERCEK embedding'den basladigi icin 'tahmin' t=0 degeri
    de gercek t=0 degeridir -- degisim ondan itibaren olculur.)"""
    out = {}
    for field in SCALAR_TARGET_FIELDS:
        arr = records[field]
        s0 = step0_true[field]
        frame0_by_traj = {int(r[2]): float(r[3]) for r in s0}
        traj_ids = np.unique(arr[:, 2].astype(int))
        n_wrong, n_total, n_trivial = 0, 0, 0
        for tid in traj_ids:
            mask = arr[:, 2].astype(int) == tid
            sub = arr[mask]
            sub = sub[np.argsort(sub[:, 0])]
            frame0_true = frame0_by_traj.get(int(tid))
            if frame0_true is None:
                continue
            last_true, last_pred = sub[-1, 3], sub[-1, 4]
            true_delta = last_true - frame0_true
            pred_delta = last_pred - frame0_true
            if abs(true_delta) < eps:
                n_trivial += 1
                continue
            n_total += 1
            if np.sign(true_delta) != np.sign(pred_delta):
                n_wrong += 1
        out[field] = {"yanlis_yon": n_wrong, "toplam_anlamli": n_total,
                       "trivial_atlanan": n_trivial,
                       "oran": (n_wrong / n_total) if n_total > 0 else float("nan")}
    return out


def example_patient_hr_table(data_dir: str, model_dir: str, patient_ids: list, steps=(0, 4, 8, 12, 16)):
    """Belirtilen hastalar icin HR gercek-vs-tahmin tablosu (t=0,10,20,30,40 dk)."""
    res = rollout_evaluate(data_dir, model_dir, max_steps=16, verbose=False)
    records = res["records"]["hr"]
    step0 = res["step0_true"]["hr"]
    out = {}
    for pid in patient_ids:
        rows = []
        # step 0 gercek deger step0_true'dan (herhangi bir step satirinin frame0_true'su)
        mask0 = step0[:, 1].astype(int) == pid
        if not mask0.any():
            continue
        frame0_true = float(step0[mask0, 3][0])
        rows.append({"step": 0, "elapsed_min": 0.0, "true": frame0_true, "pred": frame0_true})
        for step in steps:
            if step == 0:
                continue
            mask = (records[:, 0].astype(int) == step) & (records[:, 1].astype(int) == pid)
            if not mask.any():
                continue
            true_v = float(records[mask, 3][0])
            pred_v = float(records[mask, 4][0])
            rows.append({"step": step, "elapsed_min": step * 2.5, "true": true_v, "pred": pred_v})
        out[pid] = rows
    return out


def full_diagnostics(data_dir: str, model_dir: str, label: str) -> dict:
    res = rollout_evaluate(data_dir, model_dir, max_steps=16, verbose=False)
    records, step0_true = res["records"], res["step0_true"]

    pooled_r2 = {f: r_squared(records[f][:, 3], records[f][:, 4]) for f in SCALAR_TARGET_FIELDS}
    last_step = max(set(records[SCALAR_TARGET_FIELDS[0]][:, 0].astype(int)))
    last_r2 = {}
    for f in SCALAR_TARGET_FIELDS:
        mask = records[f][:, 0].astype(int) == last_step
        last_r2[f] = r_squared(records[f][mask, 3], records[f][mask, 4])

    model_mae = {f: float(np.mean(np.abs(records[f][:, 3] - records[f][:, 4]))) for f in SCALAR_TARGET_FIELDS}

    return {
        "label": label, "n_trajectories": res["n_trajectories"],
        "pooled_r2": pooled_r2, "last_step_r2": last_r2,
        "model_mae_havuzlanmis": model_mae,
        "persistence_baseline_mae": persistence_baseline_mae(step0_true),
        "variance_ratio": variance_ratio(records),
        "per_patient_distribution": per_patient_distribution(records),
        "direction_error": direction_error_count(records, step0_true),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args()
    result = full_diagnostics(args.data, args.model_dir, args.label)
    print(json.dumps(result, indent=2, ensure_ascii=False))
