"""
Çok-Adımlı Rollout İnce-Ayarı (Compounding Error Düzeltmesi)
================================================================

TEŞHİS: `train_jepa.py`'nin standart eğitimi (tek-adım, "teacher forcing" --
her adımda GERÇEK önceki durumdan başlanır) embedding-uzayında çok iyi bir
val_loss veriyor, ve `predict_next_embedding()`'in delta-tabanlı tasarımı
"ortalamaya kaçma" (collapse) sorununu çözdü -- AMA `rollout_evaluate.py`'de
GERÇEKTEN otoregresif (kendi tahminini besleyen) çalıştırıldığında, hata
adım sayısı arttıkça BÜYÜYOR (compounding error). Bunun standart nedeni:
model SADECE tek-adımlık geçişler görerek eğitildi, kendi ürettiği (küçük
hatalar içeren) bir embedding'i GİRDİ olarak hiç görmedi -- çıkarımda
(inference'ta) karşılaştığı dağılım, eğitimde gördüğünden farklı ("exposure
bias", literatürde "scheduled sampling"/"multi-step rollout training" ile
çözülen klasik bir otoregresif model sorunu).

ÇÖZÜM: Zaten TEK-ADIMLI eğitimle kalibre edilmiş Encoder/Predictor'ı
DONDURULMUŞ bir target_encoder'a karşı, GERÇEKTEN K adım otoregresif olarak
(kendi tahminini besleyerek) çalıştırıp, HER adımdaki tahminin gerçek
embedding'e olan uzaklığını cezalandıran küçük bir ince-ayar (fine-tuning)
turu -- ana eğitimin ÜZERİNE, küçük bir öğrenme oranıyla, kısa süreli.

Basitleştirme: sadece TAM UZUNLUKLU (kesilmemiş) trajectory'ler kullanılır
(rectangular tensor, ragged-batch karmaşıklığı yok) -- veri kümesinde
kesilme oranı %0 olduğu için (bkz. Faz 0/2 ölçümleri) bu neredeyse hiçbir
veriyi dışarıda bırakmıyor.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from worldmodel.learned_dynamics.dataset import load_norm_stats_or_raise
from worldmodel.learned_dynamics.model import Encoder, Predictor, get_device, predict_next_embedding
from worldmodel.learned_dynamics.state_repr import build_patient_covariate_vector, build_state_vector

_PATIENT_FIELD_NAMES = [
    "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp", "baseline_dbp",
    "baseline_spo2", "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
]


def load_full_length_trajectories(npz_path: str, split: str, norm_stats):
    """
    (N, T, state_dim) ve (N, T, action_dim) tensörleri döndürür -- SADECE
    aynı (maksimum, kesilmemiş) uzunluktaki trajectory'ler dahil edilir
    (dikdörtgen tensör için gerekli -- ragged trajectory'ler bu ince-ayar
    turunda ATLANIR, ana eğitimde zaten kullanıldılar).
    """
    data = np.load(npz_path, allow_pickle=True)
    # bkz. transient_dataset.py -- NpzFile lazy erisimi dongude YAPMA, once materyalize et.
    needed_fields = _PATIENT_FIELD_NAMES + [
        "comorbidity", "trajectory_id", "frame_idx", "traj_p", "traj_v",
        "current_hr", "conc_mg_L", "split",
    ]
    arrays = {f: data[f] for f in set(needed_fields)}

    mask = arrays["split"] == split
    row_indices = np.where(mask)[0]

    trajectories: dict[int, list[int]] = {}
    for row_idx in row_indices:
        trajectories.setdefault(int(arrays["trajectory_id"][row_idx]), []).append(int(row_idx))

    max_len = max(len(v) for v in trajectories.values())
    full_traj_ids = [tid for tid, rows in trajectories.items() if len(rows) == max_len]

    states, actions = [], []
    for tid in full_traj_ids:
        rows = sorted(trajectories[tid], key=lambda r: int(arrays["frame_idx"][r]))
        traj_states = []
        for row_idx in rows:
            patient_row = {f: arrays[f][row_idx] for f in _PATIENT_FIELD_NAMES}
            patient_row["comorbidity"] = str(arrays["comorbidity"][row_idx])
            covariates = build_patient_covariate_vector(patient_row)
            state = build_state_vector(arrays["traj_p"][row_idx], arrays["traj_v"][row_idx],
                                        covariates, current_hr=float(arrays["current_hr"][row_idx]))
            traj_states.append(norm_stats.normalize_state(state))
        traj_actions = [norm_stats.normalize_action(np.array([arrays["conc_mg_L"][row_idx]]))
                         for row_idx in rows]
        states.append(np.stack(traj_states))
        actions.append(np.stack(traj_actions))

    return (torch.from_numpy(np.stack(states).astype(np.float32)),
            torch.from_numpy(np.stack(actions).astype(np.float32)))


def finetune(data_dir: str, model_dir: str, epochs: int, lr: float, rollout_horizon: int):
    device = get_device()
    norm_stats = load_norm_stats_or_raise(data_dir, model_dir=model_dir)

    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    encoder = Encoder(state_dim=cfg["state_dim"], hidden_dim=cfg["hidden_dim"],
                       embedding_dim=cfg["embedding_dim"]).to(device)
    predictor = Predictor(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    target_encoder = Encoder(state_dim=cfg["state_dim"], hidden_dim=cfg["hidden_dim"],
                              embedding_dim=cfg["embedding_dim"]).to(device)
    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location=device))
    predictor.load_state_dict(torch.load(os.path.join(model_dir, "predictor.pt"), map_location=device))
    # target_encoder, ana eğitimin EMA ile kalibre ettiği checkpoint'ten
    # DONDURULMUŞ olarak yüklenir -- bu ince-ayar turunda GÜNCELLENMEZ
    # (kısa/az veri ile EMA'yı yeniden dengelemeye çalışmak riskli olurdu).
    target_encoder.load_state_dict(torch.load(os.path.join(model_dir, "target_encoder.pt"), map_location=device))
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    target_encoder.eval()

    npz_path = os.path.join(data_dir, "dataset.npz")
    train_states, train_actions = load_full_length_trajectories(npz_path, "train", norm_stats)
    val_states, val_actions = load_full_length_trajectories(npz_path, "val", norm_stats)
    train_states, train_actions = train_states.to(device), train_actions.to(device)
    val_states, val_actions = val_states.to(device), val_actions.to(device)

    max_horizon = train_states.shape[1] - 1
    horizon = min(rollout_horizon, max_horizon)
    print(f"Cihaz: {device} | train trajectory: {train_states.shape[0]} | "
          f"val trajectory: {val_states.shape[0]} | rollout ufku: {horizon} adım")

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    def rollout_loss(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        embedding = encoder(states[:, 0])
        loss = 0.0
        for k in range(1, horizon + 1):
            embedding = predict_next_embedding(predictor, embedding, actions[:, k])
            with torch.no_grad():
                target = target_encoder(states[:, k])
            loss = loss + torch.nn.functional.mse_loss(embedding, target)
        return loss / horizon

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        train_loss = rollout_loss(train_states, train_actions)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            val_loss = rollout_loss(val_states, val_actions).item()

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} | rollout_train_loss={train_loss.item():.4f} "
                  f"| rollout_val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
            torch.save(predictor.state_dict(), os.path.join(model_dir, "predictor.pt"))

    print(f"\nEn iyi rollout_val_loss={best_val_loss:.4f} -- encoder.pt/predictor.pt "
          f"{model_dir} içinde İNCE-AYARLI hâlleriyle GÜNCELLENDİ "
          "(decoder.pt AYRICA yeniden eğitilmeli -- embedding uzayı değişti).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rollout-horizon", type=int, default=10)
    args = parser.parse_args()

    finetune(args.data, args.model_dir, args.epochs, args.lr, args.rollout_horizon)


if __name__ == "__main__":
    main()
