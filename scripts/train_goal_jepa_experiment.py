"""
DENEY (2026-08-01): "Hedef-durum" (goal-conditioned) JEPA -- kullanıcının
önerisi. Eski `Predictor` (sınırsız delta) yerine `model.py >
GoalConditionedPredictor` (ağırlıklı ortalama, embedding_baseline'a doğru
gevşeme) ile Encoder'ı sıfırdan eğitip, aynı otoregresif rollout protokolüyle
(rollout_evaluate.py ile BİREBİR aynı ölçüm) mevcut (canlı) modelle karşılaştırır.

Bu script GEÇİCİ/deneysel -- proje kaynağının kalıcı bir parçası değil.
Birden fazla VARYANTI (farklı alpha/grad-clip ayarları) sırayla eğitip
karşılaştırmalı bir özet tablosu basar -- "sürekli dene" iş akışı için.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from worldmodel.learned_dynamics.transient_dataset import TransientDynamicsDataset
from worldmodel.learned_dynamics.model import (
    Encoder, GoalConditionedPredictor, DecoderHead, update_target_encoder, get_device,
    DEFAULT_EMBEDDING_DIM, DEFAULT_HIDDEN_DIM,
)
from worldmodel.learned_dynamics.state_repr import TRANSIENT_STATE_DIM, SCALAR_TARGET_FIELDS, build_state_vector
from worldmodel.learned_dynamics.train_jepa import variance_regularization, VARIANCE_LOSS_WEIGHT
from worldmodel.learned_dynamics.train_decoder import train as train_decoder
from worldmodel.learned_dynamics.rollout_evaluate import _load_test_trajectories, r_squared
from worldmodel.learned_dynamics.dataset import load_norm_stats_or_raise

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "transient_dataset_large")
EPOCHS = 200
BATCH_SIZE = 64
LR = 3e-4
MOMENTUM = 0.99
SEED = 0


def train_goal_jepa(out_dir: str, fixed_alpha: float | None, grad_clip: float | None,
                     weight_decay: float = 0.0, embedding_l2_weight: float = 0.0, lr: float = LR,
                     stable_baseline: bool = False, seed: int = SEED):
    torch.manual_seed(seed)
    device = get_device()
    print(f"[{os.path.basename(out_dir)}] cihaz={device} seed={seed} fixed_alpha={fixed_alpha} "
          f"grad_clip={grad_clip} lr={lr} l2={embedding_l2_weight}")

    npz_path = os.path.join(DATA_DIR, "dataset.npz")
    train_ds = TransientDynamicsDataset(npz_path, split="train")
    val_ds = TransientDynamicsDataset(npz_path, split="val", norm_stats=train_ds.norm_stats)

    os.makedirs(out_dir, exist_ok=True)
    train_ds.norm_stats.save(os.path.join(out_dir, "norm_stats.json"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    encoder = Encoder(state_dim=TRANSIENT_STATE_DIM, hidden_dim=DEFAULT_HIDDEN_DIM,
                       embedding_dim=DEFAULT_EMBEDDING_DIM).to(device)
    goal_predictor = GoalConditionedPredictor(embedding_dim=DEFAULT_EMBEDDING_DIM,
                                               hidden_dim=DEFAULT_HIDDEN_DIM,
                                               fixed_alpha=fixed_alpha).to(device)
    target_encoder = Encoder(state_dim=TRANSIENT_STATE_DIM, hidden_dim=DEFAULT_HIDDEN_DIM,
                              embedding_dim=DEFAULT_EMBEDDING_DIM).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    params = list(encoder.parameters()) + list(goal_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    diverged = False
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        goal_predictor.train()
        train_loss_sum, train_std_sum, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            next_state = batch["next_state"].to(device)
            baseline_state = batch["baseline_state"].to(device)

            online_embedding = encoder(state)
            with torch.no_grad():
                # stable_baseline=True: "hedef" (goal) referans noktasi olan
                # baseline embedding'i, her batch'te hizla degisen online
                # encoder yerine YAVAS hareket eden EMA target_encoder ile
                # hesapla. Boylece goal_embedding = baseline + goal_net(action)
                # kaymayan sabit bir ankraja dayanir -- online encoder'in
                # erken egitimdeki hizli/kaotik degisimi goal referansini
                # sarsmaz.
                baseline_encoder = target_encoder if stable_baseline else encoder
                embedding_baseline = baseline_encoder(baseline_state)
            predicted_next_embedding, _ = goal_predictor(online_embedding, embedding_baseline, action)
            with torch.no_grad():
                target_next_embedding = target_encoder(next_state)

            prediction_loss = torch.nn.functional.mse_loss(predicted_next_embedding, target_next_embedding)
            var_loss = variance_regularization(online_embedding)
            loss = prediction_loss + VARIANCE_LOSS_WEIGHT * var_loss
            if embedding_l2_weight > 0:
                # Varyans cezası SADECE alt sınır koyuyor (std>=1.0) -- üst
                # sınır YOK, embedding büyüklüğü sınırsızca büyüyebilir.
                # Bu, embedding'in ORTALAMA KARESİNİ (L2) doğrudan
                # cezalandırarak üst sınır ekler.
                loss = loss + embedding_l2_weight * online_embedding.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            update_target_encoder(target_encoder, encoder, MOMENTUM)

            train_loss_sum += prediction_loss.item()
            train_std_sum += online_embedding.detach().std(dim=0).mean().item()
            n_batches += 1

        encoder.eval()
        goal_predictor.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"].to(device)
                action = batch["action"].to(device)
                next_state = batch["next_state"].to(device)
                baseline_state = batch["baseline_state"].to(device)
                baseline_encoder = target_encoder if stable_baseline else encoder
                embedding_baseline = baseline_encoder(baseline_state)
                predicted, _ = goal_predictor(encoder(state), embedding_baseline, action)
                target = target_encoder(next_state)
                val_loss_sum += torch.nn.functional.mse_loss(predicted, target).item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)
        train_loss_avg = train_loss_sum / n_batches

        if not np.isfinite(train_loss_avg) or train_loss_avg > 1e6:
            diverged = True
            print(f"  epoch {epoch}: IRAKSADI (train_loss={train_loss_avg:.3e}) -- durduruluyor.")
            break

        if epoch % 20 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss_avg:.4f} | val_loss={val_loss:.4f} "
                  f"| embedding_std={train_std_sum / n_batches:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(out_dir, "encoder.pt"))
            torch.save(goal_predictor.state_dict(), os.path.join(out_dir, "goal_predictor.pt"))
            torch.save(target_encoder.state_dict(), os.path.join(out_dir, "target_encoder.pt"))

    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump({"embedding_dim": DEFAULT_EMBEDDING_DIM, "hidden_dim": DEFAULT_HIDDEN_DIM,
                    "state_dim": TRANSIENT_STATE_DIM, "fixed_alpha": fixed_alpha,
                    "stable_baseline": stable_baseline}, f)

    print(f"  -> en iyi val_loss={best_val_loss:.4f}{' (IRAKSADI, erken durduruldu)' if diverged else ''}")
    return best_val_loss, diverged


def goal_rollout_evaluate(out_dir: str, max_steps: int = 16, verbose: bool = True):
    device = get_device()
    norm_stats = load_norm_stats_or_raise(DATA_DIR, model_dir=out_dir)

    with open(os.path.join(out_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    stable_baseline = cfg.get("stable_baseline", False)
    encoder = Encoder(state_dim=cfg["state_dim"], hidden_dim=cfg["hidden_dim"],
                       embedding_dim=cfg["embedding_dim"]).to(device)
    goal_predictor = GoalConditionedPredictor(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"],
                                               fixed_alpha=cfg.get("fixed_alpha")).to(device)
    decoder = DecoderHead(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    encoder.load_state_dict(torch.load(os.path.join(out_dir, "encoder.pt"), map_location=device))
    goal_predictor.load_state_dict(torch.load(os.path.join(out_dir, "goal_predictor.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(out_dir, "decoder.pt"), map_location=device))
    encoder.eval(); goal_predictor.eval(); decoder.eval()

    baseline_encoder = encoder
    if stable_baseline:
        baseline_encoder = Encoder(state_dim=cfg["state_dim"], hidden_dim=cfg["hidden_dim"],
                                    embedding_dim=cfg["embedding_dim"]).to(device)
        baseline_encoder.load_state_dict(torch.load(os.path.join(out_dir, "target_encoder.pt"), map_location=device))
        baseline_encoder.eval()

    trajectories = _load_test_trajectories(os.path.join(DATA_DIR, "dataset.npz"))
    alpha_val = goal_predictor._alpha().item()
    if verbose:
        print(f"Test trajectory sayısı: {len(trajectories)} | alpha={alpha_val:.3f}")

    records = {f: [] for f in SCALAR_TARGET_FIELDS}

    with torch.no_grad():
        for traj_idx, frames in enumerate(trajectories):
            state0 = build_state_vector(frames[0]["traj_p"], frames[0]["traj_v"],
                                         frames[0]["patient_covariates"],
                                         current_hr=frames[0]["current_hr"])
            state0_norm = norm_stats.normalize_state(state0).astype(np.float32)
            embedding = encoder(torch.from_numpy(state0_norm).unsqueeze(0).to(device))
            embedding_baseline = baseline_encoder(torch.from_numpy(state0_norm).unsqueeze(0).to(device))
            patient_id = frames[0]["patient_id"]

            n_steps = min(max_steps, len(frames) - 1)
            for step in range(1, n_steps + 1):
                action_raw = np.array([[frames[step]["conc_mg_L"]]], dtype=np.float32)
                action_norm = norm_stats.normalize_action(action_raw).astype(np.float32)
                action_t = torch.from_numpy(action_norm).to(device)

                embedding, _ = goal_predictor(embedding, embedding_baseline, action_t)
                decoded_norm = decoder(embedding).cpu().numpy()[0]

                true_scalars = frames[step]["scalars"]
                for i, field in enumerate(SCALAR_TARGET_FIELDS):
                    pred_value = norm_stats.denormalize_scalar(field, decoded_norm[i])
                    records[field].append((step, patient_id, traj_idx, true_scalars[field], pred_value))

    for field in SCALAR_TARGET_FIELDS:
        records[field] = np.array(records[field], dtype=np.float64)

    steps_sorted = sorted(set(records[SCALAR_TARGET_FIELDS[0]][:, 0].astype(int)))
    if verbose:
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

    pooled = {}
    last_step = max(steps_sorted)
    last_step_r2 = {}
    for field in SCALAR_TARGET_FIELDS:
        true_v, pred_v = records[field][:, 3], records[field][:, 4]
        pooled[field] = r_squared(true_v, pred_v)
        mask = records[field][:, 0].astype(int) == last_step
        last_step_r2[field] = r_squared(records[field][mask, 3], records[field][mask, 4])

    if verbose:
        print(f"\n{'Havuzlanmış R²':<20}" + "".join(f"{f}={pooled[f]:.3f}  " for f in SCALAR_TARGET_FIELDS))
        print(f"{'Son adım R²':<20}" + "".join(f"{f}={last_step_r2[f]:.3f}  " for f in SCALAR_TARGET_FIELDS))

    return pooled, last_step_r2


CHAMPION = ("a0.7_l2_1e-3", 0.7, 1.0, 0.0, 1e-3, LR, False)

VARIANTS = [
    # (isim, fixed_alpha, grad_clip, weight_decay, embedding_l2_weight, lr, stable_baseline, seed)
    # Tur 4: stable_baseline hipotezi CURUTULDU (daha erken ve daha kotu
    # ıraksadı). Tur 5: sampiyon ayarı (alpha=0.7, L2=1e-3, online baseline)
    # SABIT tutup sadece SEED'i degistirerek sonucun tesadufi olup
    # olmadigini dogruluyoruz -- canliya gecis karari icin sart.
    (CHAMPION[0] + "_seed0", *CHAMPION[1:7], 0),
    (CHAMPION[0] + "_seed1", *CHAMPION[1:7], 1),
    (CHAMPION[0] + "_seed2", *CHAMPION[1:7], 2),
]

BASELINE_POOLED_R2 = {"ef": 0.991, "co": 0.126, "hr": 0.973, "edv": 0.937, "esv": 0.988}
BASELINE_LAST_STEP_R2 = {"ef": 0.990, "co": 0.209, "hr": 0.977, "edv": 0.943, "esv": 0.985}
# Tur 1'in en iyisi (duz alpha=0.7, wd/l2 yok) -- referans olarak
ROUND1_BEST_POOLED_R2 = {"ef": 0.985, "co": 0.067, "hr": 0.974, "edv": 0.942, "esv": 0.978}
ROUND1_BEST_LAST_STEP_R2 = {"ef": 0.996, "co": 0.146, "hr": 0.985, "edv": 0.989, "esv": 0.998}


def main():
    summary = {}
    for name, fixed_alpha, grad_clip, weight_decay, embedding_l2_weight, lr, stable_baseline, seed in VARIANTS:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "models", f"dynamics_jepa_goal_{name}")
        print(f"\n{'=' * 70}\nVARYANT: {name}\n{'=' * 70}")
        best_val_loss, diverged = train_goal_jepa(out_dir, fixed_alpha, grad_clip, weight_decay, embedding_l2_weight,
                                                    lr, stable_baseline, seed)
        if diverged and best_val_loss == float("inf"):
            summary[name] = None
            continue
        print(f"\n--- Decoder eğitimi ({name}) ---")
        train_decoder(DATA_DIR, out_dir, epochs=60, batch_size=64, lr=1e-3,
                      dataset_cls=TransientDynamicsDataset, seed=seed)
        print(f"\n--- Rollout değerlendirmesi ({name}) ---")
        pooled, last_step = goal_rollout_evaluate(out_dir, max_steps=16, verbose=False)
        print(f"Havuzlanmış R²: " + ", ".join(f"{k}={v:.3f}" for k, v in pooled.items()))
        print(f"Son adım R²:    " + ", ".join(f"{k}={v:.3f}" for k, v in last_step.items()))
        summary[name] = {"pooled": pooled, "last_step": last_step}

    print(f"\n\n{'=' * 90}\nÖZET -- TÜM VARYANTLAR vs MEVCUT (CANLI) MODEL\n{'=' * 90}")
    header = f"{'Varyant':<24}" + "".join(f"{f:<10}" for f in SCALAR_TARGET_FIELDS)
    print(header + "  (havuzlanmış R²)")
    print(f"{'mevcut (canli)':<24}" + "".join(f"{BASELINE_POOLED_R2[f]:<10.3f}" for f in SCALAR_TARGET_FIELDS))
    for name, result in summary.items():
        if result is None:
            print(f"{name:<24} IRAKSADI")
        else:
            print(f"{name:<24}" + "".join(f"{result['pooled'][f]:<10.3f}" for f in SCALAR_TARGET_FIELDS))

    print()
    print(header + "  (son adım R², 40dk)")
    print(f"{'mevcut (canli)':<24}" + "".join(f"{BASELINE_LAST_STEP_R2[f]:<10.3f}" for f in SCALAR_TARGET_FIELDS))
    for name, result in summary.items():
        if result is None:
            print(f"{name:<24} IRAKSADI")
        else:
            print(f"{name:<24}" + "".join(f"{result['last_step'][f]:<10.3f}" for f in SCALAR_TARGET_FIELDS))


if __name__ == "__main__":
    main()
