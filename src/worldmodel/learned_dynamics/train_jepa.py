"""
JEPA Eğitim Döngüsü (Faz 3)
==============================

Kayıp: predictor'ın ürettiği tahmini sonraki-durum embedding'i ile,
target-encoder'ın GERÇEK sonraki durumdan (CircAdapt'in ilaçlı koşumu)
ürettiği embedding arasındaki ortalama-kare mesafe (MSE). target-encoder
dalına gradyan AKMAZ (bkz. model.py > update_target_encoder, stop-gradient).

Ek güvence (VICReg'den esinlenen, hafif bir varyans terimi): sadece
stop-gradient/EMA'ya güvenmek yerine, batch içindeki embedding'lerin her
boyutunun standart sapmasını bir alt eşiğin (örn. 1.0) ÜZERİNDE tutmaya
zorlayan küçük bir ceza terimi de ekleniyor -- state/action uzayı burada
düşük boyutlu ve TAMAMEN deterministik (CircAdapt her zaman aynı girdiye
aynı çıktıyı verir, görüntüdeki gibi doğal bir belirsizlik yok), bu yüzden
saf JEPA'nın standart collapse-önleme mekanizması (stop-gradient+EMA) tek
başına yeterli olsa da, ucuz bir ikinci güvence katmanı eklemek makul.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from torch.utils.data import DataLoader

from worldmodel.learned_dynamics.dataset import DynamicsDataset
from worldmodel.learned_dynamics.model import (
    Encoder, Predictor, update_target_encoder, get_device, predict_next_embedding,
    DEFAULT_EMBEDDING_DIM, DEFAULT_HIDDEN_DIM,
)
from worldmodel.learned_dynamics.state_repr import STATE_DIM

VARIANCE_TARGET_STD = 1.0
VARIANCE_LOSS_WEIGHT = 0.05


def variance_regularization(embeddings: torch.Tensor) -> torch.Tensor:
    """Batch içindeki her embedding boyutunun std'sini VARIANCE_TARGET_STD'nin
    altına düşürmeyi cezalandırır -- collapse'e karşı ikinci savunma hattı."""
    std = embeddings.std(dim=0)
    return torch.relu(VARIANCE_TARGET_STD - std).mean()


def train(data_dir: str, out_dir: str, epochs: int, batch_size: int, lr: float,
          embedding_dim: int, hidden_dim: int, momentum: float,
          dataset_cls: type = DynamicsDataset, state_dim: int = STATE_DIM,
          seed: int = 0):
    """
    dataset_cls/state_dim: OPSİYONEL, varsayılanlarıyla tek-adımlı MVP'nin
    (`DynamicsDataset`/`STATE_DIM`) eski davranışını AYNEN korur -- bu ikisi
    SADECE `train_jepa_transient.py`'nin `TransientDynamicsDataset`/
    `TRANSIENT_STATE_DIM` ile çağırması için eklendi. Eğitim döngüsünün
    kendisi (optimizer, EMA güncelleme, varyans cezası) burada TEK YERDE
    kalıyor, kopyalanmıyor.

    seed: OPSİYONEL (varsayılan 0), tekrarlanabilirlik için. Tek-tohumlu tek
    bir koşum, R² farkının veri artışından mı tohum gürültüsünden mi
    geldiğini AYIRAMAZ -- karşılaştırma yapılıyorsa birden fazla seed ile
    koşup ortalama±std raporlanmalı.
    """
    torch.manual_seed(seed)
    device = get_device()
    print(f"Cihaz: {device} | seed={seed}")

    npz_path = os.path.join(data_dir, "dataset.npz")
    train_ds = dataset_cls(npz_path, split="train")
    val_ds = dataset_cls(npz_path, split="val", norm_stats=train_ds.norm_stats)

    os.makedirs(out_dir, exist_ok=True)
    train_ds.norm_stats.save(os.path.join(data_dir, "norm_stats.json"))
    # AYRICA checkpoint dizinine kaydet -- veri dizini daha sonra baska bir
    # veri setiyle DEGISTIRILIRSE (bkz. data/transient_dataset_large'in 1560
    # veriyle degistirilmesi sirasinda eski norm_stats.json'un sessizce
    # silinmesi), checkpoint kendi norm_stats'ini kaybetmez.
    train_ds.norm_stats.save(os.path.join(out_dir, "norm_stats.json"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    encoder = Encoder(state_dim=state_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    predictor = Predictor(embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    target_encoder = Encoder(state_dim=state_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr,
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        train_loss_sum, train_std_sum, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            next_state = batch["next_state"].to(device)

            online_embedding = encoder(state)
            predicted_next_embedding = predict_next_embedding(predictor, online_embedding, action)
            with torch.no_grad():
                target_next_embedding = target_encoder(next_state)

            prediction_loss = torch.nn.functional.mse_loss(predicted_next_embedding, target_next_embedding)
            var_loss = variance_regularization(online_embedding)
            loss = prediction_loss + VARIANCE_LOSS_WEIGHT * var_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_target_encoder(target_encoder, encoder, momentum)

            train_loss_sum += prediction_loss.item()
            train_std_sum += online_embedding.detach().std(dim=0).mean().item()
            n_batches += 1

        encoder.eval()
        predictor.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"].to(device)
                action = batch["action"].to(device)
                next_state = batch["next_state"].to(device)
                predicted = predict_next_embedding(predictor, encoder(state), action)
                target = target_encoder(next_state)
                val_loss_sum += torch.nn.functional.mse_loss(predicted, target).item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss_sum / n_batches:.4f} "
              f"| val_loss={val_loss:.4f} | embedding_std={train_std_sum / n_batches:.3f} "
              f"(collapse uyarı eşiği: <0.1)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(out_dir, "encoder.pt"))
            torch.save(predictor.state_dict(), os.path.join(out_dir, "predictor.pt"))
            torch.save(target_encoder.state_dict(), os.path.join(out_dir, "target_encoder.pt"))

    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"embedding_dim": embedding_dim, "hidden_dim": hidden_dim, "state_dim": state_dim}, f)

    print(f"\nEn iyi val_loss={best_val_loss:.4f} -- checkpoint'ler {out_dir} içine kaydedildi.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data", "dynamics_dataset"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "models", "dynamics_jepa"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args.data, args.out_dir, args.epochs, args.batch_size, args.lr,
          args.embedding_dim, args.hidden_dim, args.momentum, seed=args.seed)


if __name__ == "__main__":
    main()
