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
    mask = data["split"] == "test"
    row_indices = np.where(mask)[0]

    trajectories: dict[int, list[dict]] = {}
    for row_idx in row_indices:
        traj_id = int(data["trajectory_id"][row_idx])
        patient_row = {f: data[f][row_idx] for f in _PATIENT_FIELD_NAMES}
        patient_row["comorbidity"] = str(data["comorbidity"][row_idx])
        frame = {
            "frame_idx": int(data["frame_idx"][row_idx]),
            "conc_mg_L": float(data["conc_mg_L"][row_idx]),
            "current_hr": float(data["current_hr"][row_idx]),
            "traj_p": data["traj_p"][row_idx], "traj_v": data["traj_v"][row_idx],
            "patient_covariates": build_patient_covariate_vector(patient_row),
            "scalars": {f: float(data[f][row_idx]) if f != "hr" else float(data["current_hr"][row_idx])
                        for f in SCALAR_TARGET_FIELDS},
        }
        trajectories.setdefault(traj_id, []).append(frame)

    for traj_id in trajectories:
        trajectories[traj_id].sort(key=lambda fr: fr["frame_idx"])

    return list(trajectories.values())


def rollout_evaluate(data_dir: str, model_dir: str, max_steps: int = 30):
    device = get_device()
    norm_stats = load_norm_stats_or_raise(data_dir)

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
    print(f"Test trajectory sayısı: {len(trajectories)}")

    # step_errors[field][step_idx] = o adımdaki mutlak hatalar listesi (tüm trajectory'ler)
    step_errors = {f: {} for f in SCALAR_TARGET_FIELDS}

    with torch.no_grad():
        for frames in trajectories:
            state0 = build_state_vector(frames[0]["traj_p"], frames[0]["traj_v"],
                                         frames[0]["patient_covariates"],
                                         current_hr=frames[0]["current_hr"])
            state0_norm = norm_stats.normalize_state(state0).astype(np.float32)
            embedding = encoder(torch.from_numpy(state0_norm).unsqueeze(0).to(device))

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
                    abs_error = abs(pred_value - true_scalars[field])
                    step_errors[field].setdefault(step, []).append(abs_error)

    print(f"\n{'Adım':<6}{'Dakika':<8}" + "".join(f"{f + ' MAE':<14}" for f in SCALAR_TARGET_FIELDS))
    print("-" * (6 + 8 + 14 * len(SCALAR_TARGET_FIELDS)))
    for step in sorted(step_errors[SCALAR_TARGET_FIELDS[0]].keys()):
        elapsed_min = step * 2.5
        row = f"{step:<6}{elapsed_min:<8.1f}"
        for field in SCALAR_TARGET_FIELDS:
            errors = step_errors[field].get(step, [])
            mae = float(np.mean(errors)) if errors else float("nan")
            row += f"{mae:<14.3f}"
        print(row)

    print("\nHatanın adım sayısına göre büyüme eğilimi (compounding error) -- "
          "ilk birkaç adımdaki hataya kıyasla son adımlardaki hata belirgin şekilde "
          "büyüyorsa, model uzun-vadeli rollout'ta kararsızlaşıyor demektir.")


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
