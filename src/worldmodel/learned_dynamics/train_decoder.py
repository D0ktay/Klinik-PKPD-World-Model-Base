"""
Decoder Eğitimi (Faz 4)
==========================

JEPA encoder'ı (train_jepa.py tarafından eğitildi) DONDURULMUŞ halde
kullanılır -- burada SADECE küçük bir DecoderHead, embedding'i geri
yorumlanabilir fizyolojik sayılara (EF, CO, HR, EDV, ESV -- bkz.
state_repr.SCALAR_TARGET_FIELDS) çevirmeyi supervised (gözetimli) olarak
öğrenir. Bu, sohbette netleştirdiğimiz "kullanıcıya giden nihai çıktı
CircAdapt'in bugün ürettiğiyle AYNI formatta olmalı" gereksinimini
karşılıyor -- embedding'in kendisi insan tarafından yorumlanamaz, decoder
bunu tekrar anlamlı sayılara çeviriyor.

Hem baseline hem ilaçlı durum kayıtları kullanılıyor (encoder ikisine de
aynı şekilde uygulanıyor) -- bu, decoder eğitim verisini etkin şekilde
2 katına çıkarıyor.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from worldmodel.learned_dynamics.dataset import DynamicsDataset, load_norm_stats_or_raise
from worldmodel.learned_dynamics.model import Encoder, DecoderHead, get_device
from worldmodel.learned_dynamics.state_repr import SCALAR_TARGET_FIELDS, STATE_DIM


def scalars_to_tensor(scalars_dict_batch: dict, norm_stats) -> torch.Tensor:
    """batch içindeki {"ef": tensor([...]), "co": ..., ...} sözlüğünü,
    SCALAR_TARGET_FIELDS sırasıyla normalize edilmiş tek bir (B, 5) tensöre çevirir."""
    cols = []
    for field in SCALAR_TARGET_FIELDS:
        raw = scalars_dict_batch[field].numpy()
        cols.append(norm_stats.normalize_scalar(field, raw))
    import numpy as np
    return torch.from_numpy(np.stack(cols, axis=1).astype("float32"))


def train(data_dir: str, model_dir: str, epochs: int, batch_size: int, lr: float,
          dataset_cls: type = DynamicsDataset, seed: int = 0):
    """dataset_cls: OPSİYONEL -- varsayılanıyla (`DynamicsDataset`) eski
    davranış korunur; `train_decoder_transient.py` bunu
    `TransientDynamicsDataset` ile çağırır. seed: decoder rastgele
    başlatıldığı + DataLoader shuffle kullandığı için tekrarlanabilirlik
    açısından önemli."""
    torch.manual_seed(seed)
    device = get_device()
    norm_stats = load_norm_stats_or_raise(data_dir, model_dir=model_dir)

    npz_path = os.path.join(data_dir, "dataset.npz")
    train_ds = dataset_cls(npz_path, split="train", norm_stats=norm_stats)
    val_ds = dataset_cls(npz_path, split="val", norm_stats=norm_stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    encoder = Encoder(state_dim=cfg.get("state_dim", STATE_DIM),
                       hidden_dim=cfg["hidden_dim"], embedding_dim=cfg["embedding_dim"]).to(device)
    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    decoder = DecoderHead(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        decoder.train()
        train_loss_sum, n_batches = 0.0, 0
        for batch in train_loader:
            state = batch["state"].to(device)
            next_state = batch["next_state"].to(device)
            base_targets = scalars_to_tensor(batch["base_scalars"], norm_stats).to(device)
            drug_targets = scalars_to_tensor(batch["drug_scalars"], norm_stats).to(device)

            with torch.no_grad():
                base_embedding = encoder(state)
                drug_embedding = encoder(next_state)

            pred_base = decoder(base_embedding)
            pred_drug = decoder(drug_embedding)
            loss = (torch.nn.functional.mse_loss(pred_base, base_targets) +
                    torch.nn.functional.mse_loss(pred_drug, drug_targets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        decoder.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"].to(device)
                next_state = batch["next_state"].to(device)
                base_targets = scalars_to_tensor(batch["base_scalars"], norm_stats).to(device)
                drug_targets = scalars_to_tensor(batch["drug_scalars"], norm_stats).to(device)
                pred_base = decoder(encoder(state))
                pred_drug = decoder(encoder(next_state))
                val_loss_sum += (torch.nn.functional.mse_loss(pred_base, base_targets) +
                                  torch.nn.functional.mse_loss(pred_drug, drug_targets)).item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss_sum / n_batches:.4f} "
                  f"| val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder.pt"))

    print(f"\nEn iyi val_loss={best_val_loss:.4f} -- decoder.pt {model_dir} içine kaydedildi.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data", "dynamics_dataset"))
    parser.add_argument("--model-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "models", "dynamics_jepa"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args.data, args.model_dir, args.epochs, args.batch_size, args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
