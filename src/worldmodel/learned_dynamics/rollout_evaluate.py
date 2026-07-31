"""
Otoregresif Rollout Değerlendirmesi (Faz 6)
===============================================

Modelin GERÇEKTEN "world model" gibi davranıp davranmadığının asıl kanıtı
burada -- tek-adım val_loss'unda DEĞİL. Her test trajectory'si için:

  1. SADECE frame_idx=0'ın (ilaçsız kararlı-durum) GERÇEK embedding'inden başla.
  2. Her adımda Predictor'ı [MEVCUT embedding, GERÇEK ilaç konsantrasyonu(t)]
     ile çağır -- konsantrasyon eğrisi bilinen/dışsal bir girdi (gerçek
     hastada da ilaç ne zaman ne kadar verildiği bilinir), ama STATE
     modelin KENDİ ürettiği tahmin -- gerçek veri ARAYA HİÇ KARIŞMIYOR
     ("teacher forcing" değil, gerçek otoregresif rollout).
  3. Her adımda DecoderHead ile tahmini embedding'i EF/CO/HR/EDV/ESV'ye çevir,
     CircAdapt'in GERÇEKTEN ürettiği değerle karşılaştır.
  4. Hatanın adım sayısına göre (compounding error) nasıl büyüdüğünü raporla.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from worldmodel.learned_dynamics.dataset import load_norm_stats_or_raise
from worldmodel.learned_dynamics.model import Encoder, Predictor, DecoderHead, get_device, predict_next_embedding
from worldmodel.learned_dynamics.state_repr import (
    SCALAR_TARGET_FIELDS, build_patient_covariate_vector, build_state_vector,
)

_PATIENT_FIELD_NAMES = [
    "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp", "baseline_dbp",
    "baseline_spo2", "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
]


def _load_test_trajectories(npz_path: str) -> list[dict]:
    """npz'deki test satırlarını trajectory_id'ye göre gruplayıp frame_idx'e
    göre sıralanmış bir liste-of-dict olarak döndürür (rollout için tam
    sekans gerekiyor, TransientDynamicsDataset'in düzleştirdiği ÇİFTLER
    değil)."""
    data = np.load(npz_path, allow_pickle=True)
    # bkz. transient_dataset.py -- NpzFile lazy erisimi dongude YAPMA, once materyalize et.
    needed_fields = _PATIENT_FIELD_NAMES + [
        "comorbidity", "trajectory_id", "frame_idx", "patient_id", "conc_mg_L", "current_hr",
        "traj_p", "traj_v", "split",
    ] + [f for f in SCALAR_TARGET_FIELDS if f != "hr"]
    arrays = {f: data[f] for f in set(needed_fields)}

    mask = arrays["split"] == "test"
    row_indices = np.where(mask)[0]

    trajectories: dict[int, list[dict]] = {}
    for row_idx in row_indices:
        traj_id = int(arrays["trajectory_id"][row_idx])
        patient_row = {f: arrays[f][row_idx] for f in _PATIENT_FIELD_NAMES}
        patient_row["comorbidity"] = str(arrays["comorbidity"][row_idx])
        frame = {
            "frame_idx": int(arrays["frame_idx"][row_idx]),
            "patient_id": int(arrays["patient_id"][row_idx]),
            "conc_mg_L": float(arrays["conc_mg_L"][row_idx]),
            "current_hr": float(arrays["current_hr"][row_idx]),
            "traj_p": arrays["traj_p"][row_idx], "traj_v": arrays["traj_v"][row_idx],
            "patient_covariates": build_patient_covariate_vector(patient_row),
            "scalars": {f: float(arrays[f][row_idx]) if f != "hr" else float(arrays["current_hr"][row_idx])
                        for f in SCALAR_TARGET_FIELDS},
        }
        trajectories.setdefault(traj_id, []).append(frame)

    for traj_id in trajectories:
        trajectories[traj_id].sort(key=lambda fr: fr["frame_idx"])

    return list(trajectories.values())


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")


def rollout_evaluate(data_dir: str, model_dir: str, max_steps: int = 30, verbose: bool = True):
    """Otoregresif rollout'u çalıştırır. Konsola MAE+R² tablosu basar (verbose=True
    ise) VE her (trajectory, step, field) kaydını ham haliyle döndürür -- bu ham
    veri hem "eski R² sayılarını yeniden üret" doğrulaması hem de ADIM 4'teki
    ortalamaya-kaçış teşhisleri (varyans oranı, hasta-başına dağılım, yön hatası)
    için kullanılıyor."""
    device = get_device()
    norm_stats = load_norm_stats_or_raise(data_dir, model_dir=model_dir)

    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    encoder = Encoder(state_dim=cfg["state_dim"], hidden_dim=cfg["hidden_dim"],
                       embedding_dim=cfg["embedding_dim"]).to(device)
    predictor = Predictor(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    decoder = DecoderHead(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location=device))
    predictor.load_state_dict(torch.load(os.path.join(model_dir, "predictor.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(model_dir, "decoder.pt"), map_location=device))
    encoder.eval(); predictor.eval(); decoder.eval()

    trajectories = _load_test_trajectories(os.path.join(data_dir, "dataset.npz"))
    if verbose:
        print(f"Test trajectory sayısı: {len(trajectories)}")

    # records[field] = liste of (step, patient_id, trajectory_id, true, pred)
    records = {f: [] for f in SCALAR_TARGET_FIELDS}
    step0_true = {f: [] for f in SCALAR_TARGET_FIELDS}  # persistence baseline icin (frame 0 gercek deger)

    with torch.no_grad():
        for traj_idx, frames in enumerate(trajectories):
            state0 = build_state_vector(frames[0]["traj_p"], frames[0]["traj_v"],
                                         frames[0]["patient_covariates"],
                                         current_hr=frames[0]["current_hr"])
            state0_norm = norm_stats.normalize_state(state0).astype(np.float32)
            embedding = encoder(torch.from_numpy(state0_norm).unsqueeze(0).to(device))
            patient_id = frames[0]["patient_id"]

            n_steps = min(max_steps, len(frames) - 1)
            for step in range(1, n_steps + 1):
                action_raw = np.array([[frames[step]["conc_mg_L"]]], dtype=np.float32)
                action_norm = norm_stats.normalize_action(action_raw).astype(np.float32)
                action_t = torch.from_numpy(action_norm).to(device)

                embedding = predict_next_embedding(predictor, embedding, action_t)  # OTOREGRESİF -- kendi tahminini besliyor
                decoded_norm = decoder(embedding).cpu().numpy()[0]

                true_scalars = frames[step]["scalars"]
                for i, field in enumerate(SCALAR_TARGET_FIELDS):
                    pred_value = norm_stats.denormalize_scalar(field, decoded_norm[i])
                    records[field].append((step, patient_id, traj_idx, true_scalars[field], pred_value))
                    step0_true[field].append((step, patient_id, traj_idx, frames[0]["scalars"][field], true_scalars[field]))

    # numpy'a cevir
    for field in SCALAR_TARGET_FIELDS:
        records[field] = np.array(records[field], dtype=np.float64)
        step0_true[field] = np.array(step0_true[field], dtype=np.float64)

    if verbose:
        steps_sorted = sorted(set(records[SCALAR_TARGET_FIELDS[0]][:, 0].astype(int)))
        print(f"\n{'Adım':<6}{'Dakika':<8}" + "".join(f"{f + ' MAE':<12}{f + ' R2':<10}" for f in SCALAR_TARGET_FIELDS))
        print("-" * (6 + 8 + 22 * len(SCALAR_TARGET_FIELDS)))
        for step in steps_sorted:
            elapsed_min = step * 2.5
            row = f"{step:<6}{elapsed_min:<8.1f}"
            for field in SCALAR_TARGET_FIELDS:
                mask = records[field][:, 0].astype(int) == step
                true_v, pred_v = records[field][mask, 3], records[field][mask, 4]
                mae = float(np.mean(np.abs(true_v - pred_v))) if mask.any() else float("nan")
                r2 = r_squared(true_v, pred_v) if mask.any() else float("nan")
                row += f"{mae:<12.3f}{r2:<10.3f}"
            print(row)

        print(f"\n{'Havuzlanmış (tüm adımlar) R²':<32}", end="")
        for field in SCALAR_TARGET_FIELDS:
            true_v, pred_v = records[field][:, 3], records[field][:, 4]
            print(f"{field}={r_squared(true_v, pred_v):.3f}  ", end="")
        print()

        print(f"{'Son adım R²':<32}", end="")
        last_step = max(steps_sorted)
        for field in SCALAR_TARGET_FIELDS:
            mask = records[field][:, 0].astype(int) == last_step
            true_v, pred_v = records[field][mask, 3], records[field][mask, 4]
            print(f"{field}={r_squared(true_v, pred_v):.3f}  ", end="")
        print()

        print("\nHatanın adım sayısına göre büyüme eğilimi (compounding error) -- "
              "ilk birkaç adımdaki hataya kıyasla son adımlardaki hata belirgin şekilde "
              "büyüyorsa, model uzun-vadeli rollout'ta kararsızlaşıyor demektir.")

    return {"records": records, "step0_true": step0_true, "n_trajectories": len(trajectories)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data", "transient_dataset"))
    parser.add_argument("--model-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "models", "dynamics_jepa_transient"))
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    rollout_evaluate(args.data, args.model_dir, args.max_steps)


if __name__ == "__main__":
    main()
