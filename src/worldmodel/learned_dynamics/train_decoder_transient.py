"""Transient Decoder Eğitimi -- İnce Sarmalayıcı (bkz. train_jepa_transient.py
ile aynı desen). Eğitim döngüsü train_decoder.py::train()'de kalıyor."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worldmodel.learned_dynamics.train_decoder import train
from worldmodel.learned_dynamics.transient_dataset import TransientDynamicsDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data", "transient_dataset"))
    parser.add_argument("--model-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "models", "dynamics_jepa_transient"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args.data, args.model_dir, args.epochs, args.batch_size, args.lr,
          dataset_cls=TransientDynamicsDataset, seed=args.seed)


if __name__ == "__main__":
    main()
