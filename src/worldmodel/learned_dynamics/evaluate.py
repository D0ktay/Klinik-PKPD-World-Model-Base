"""
Değerlendirme (Faz 5)
========================

Held-out (eğitimde HİÇ görülmemiş) test split'i üzerinde: baseline durumdan
başlayıp predictor+decoder zinciriyle üretilen "ilaç sonrası tahmin"i,
CircAdapt'in GERÇEKTEN ürettiği ilaçlı durumla (EF, CO, HR, EDV, ESV)
karşılaştırır. MAE (ortalama mutlak hata) ve R² (açıklanan varyans oranı,
1.0 = mükemmel) raporlar.

Ayrıca bir referans karşılaştırması da yapılır: "hiçbir şey tahmin etme,
sadece baseline değerini kopyala" -- modelin bu saf/triviyal temelden daha
iyi olduğunu göstermek için (aksi halde düşük MAE yanıltıcı olabilir, ilaç
etkisi zaten küçükse baseline'ı kopyalamak da düşük hata verir).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from worldmodel.learned_dynamics.dataset import DynamicsDataset, load_norm_stats_or_raise
from worldmodel.learned_dynamics.model import Encoder, Predictor, DecoderHead, get_device, predict_next_embedding
from worldmodel.learned_dynamics.state_repr import SCALAR_TARGET_FIELDS


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")


def evaluate(data_dir: str, model_dir: str):
    device = get_device()
    norm_stats = load_norm_stats_or_raise(data_dir, model_dir=model_dir)

    npz_path = os.path.join(data_dir, "dataset.npz")
    test_ds = DynamicsDataset(npz_path, split="test", norm_stats=norm_stats)

    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    encoder = Encoder(hidden_dim=cfg["hidden_dim"], embedding_dim=cfg["embedding_dim"]).to(device)
    predictor = Predictor(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    decoder = DecoderHead(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location=device))
    predictor.load_state_dict(torch.load(os.path.join(model_dir, "predictor.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(model_dir, "decoder.pt"), map_location=device))
    encoder.eval(); predictor.eval(); decoder.eval()

    state = torch.from_numpy(test_ds.state_norm).to(device)
    action = torch.from_numpy(test_ds.action_norm).to(device)

    with torch.no_grad():
        base_embedding = encoder(state)
        predicted_drug_embedding = predict_next_embedding(predictor, base_embedding, action)
        predicted_scalars_norm = decoder(predicted_drug_embedding).cpu().numpy()

    print(f"Test kümesi: {len(test_ds)} (hasta, doz) çifti\n")
    print(f"{'Metrik':<8}{'MAE (model)':>15}{'MAE (baseline kopyası)':>25}{'R² (model)':>14}")
    print("-" * 62)

    for i, field in enumerate(SCALAR_TARGET_FIELDS):
        true_drug = test_ds.drug_scalars[field]
        true_base = test_ds.base_scalars[field]
        pred_drug = norm_stats.denormalize_scalar(field, predicted_scalars_norm[:, i])

        mae_model = np.abs(true_drug - pred_drug).mean()
        mae_baseline_copy = np.abs(true_drug - true_base).mean()
        r2 = r_squared(true_drug, pred_drug)

        print(f"{field:<8}{mae_model:>15.3f}{mae_baseline_copy:>25.3f}{r2:>14.3f}")

    print("\n'MAE (model)' < 'MAE (baseline kopyası)' ise model gerçekten ilaç "
          "etkisini öğrenmiş demektir -- sadece baseline'ı ezberlemiyor.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data", "dynamics_dataset"))
    parser.add_argument("--model-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "models", "dynamics_jepa"))
    args = parser.parse_args()

    evaluate(args.data, args.model_dir)


if __name__ == "__main__":
    main()
