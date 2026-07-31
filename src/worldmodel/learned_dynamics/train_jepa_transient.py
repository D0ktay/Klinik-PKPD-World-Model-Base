"""
Transient JEPA Eğitimi -- İnce Sarmalayıcı (Faz 5)
======================================================

Eğitim döngüsünün kendisi `train_jepa.py::train()`'de TEK YERDE yaşıyor --
bu script sadece `dataset_cls=TransientDynamicsDataset` ve
`state_dim=TRANSIENT_STATE_DIM` parametrelerini geçirerek onu çağırıyor.
Checkpoint'ler `models/dynamics_jepa_transient/`'a kaydedilir -- tek-adımlı
MVP'nin `models/dynamics_jepa/` checkpoint'lerinin ÜZERİNE YAZILMAZ.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worldmodel.learned_dynamics.train_jepa import train
from worldmodel.learned_dynamics.transient_dataset import TransientDynamicsDataset
from worldmodel.learned_dynamics.state_repr import TRANSIENT_STATE_DIM
from worldmodel.learned_dynamics.model import DEFAULT_EMBEDDING_DIM, DEFAULT_HIDDEN_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data", "transient_dataset"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "models", "dynamics_jepa_transient"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args.data, args.out_dir, args.epochs, args.batch_size, args.lr,
          args.embedding_dim, args.hidden_dim, args.momentum,
          dataset_cls=TransientDynamicsDataset, state_dim=TRANSIENT_STATE_DIM,
          seed=args.seed)


if __name__ == "__main__":
    main()
