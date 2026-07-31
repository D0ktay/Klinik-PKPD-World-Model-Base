"""
PyTorch Dataset -- scripts/generate_dynamics_dataset.py'nin ürettiği
data/dynamics_dataset/dataset.npz dosyasını, state_repr.py'deki vektör
temsiline çevirerek okur.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodel.learned_dynamics.state_repr import (
    build_patient_covariate_vector, build_state_vector, NormStats, PATIENT_COVARIATE_DIM,
)


class DynamicsDataset(Dataset):
    """
    Her örnek, TEK bir (hasta, doz) çiftine karşılık gelir:
      state       -- ilaçsız (baseline) durumun normalize edilmiş vektörü
      action      -- normalize edilmiş doz
      next_state  -- ilaçlı durumun normalize edilmiş vektörü (JEPA hedefi)
      base_scalars / drug_scalars -- decoder eğitimi için ham (normalize
        EDİLMEMİŞ, train_decoder.py kendi normalize eder) EF/CO/HR/EDV/ESV
        sözlükleri -- hem baseline hem ilaçlı durumdan decoder eğitim
        çifti üretmek için (aynı encoder her iki duruma da uygulanıyor).

    split: "train" | "val" | "test" -- manifest'teki hasta-bazlı bölünmeye göre
    filtrelenir (bkz. generate_dynamics_dataset.py, sızıntı önleme notu).
    norm_stats: None ise (yalnızca split="train" ile) bu split'ten hesaplanır;
    val/test için train'den hesaplanan NormStats MUTLAKA dışarıdan verilmeli
    (aksi halde val/test istatistikleri eğitime "sızar").
    """

    def __init__(self, npz_path: str, split: str, norm_stats: NormStats | None = None):
        data = np.load(npz_path, allow_pickle=True)
        mask = data["split"] == split
        if not mask.any():
            raise ValueError(f"'{split}' split'inde hiç örnek yok -- {npz_path}")

        n = int(mask.sum())
        patient_covariates = np.zeros((n, PATIENT_COVARIATE_DIM), dtype=np.float32)
        for i, row_idx in enumerate(np.where(mask)[0]):
            row = {f: data[f][row_idx] for f in [
                "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp", "baseline_dbp",
                "baseline_spo2", "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
            ]}
            row["comorbidity"] = str(data["comorbidity"][row_idx])
            patient_covariates[i] = build_patient_covariate_vector(row)

        traj_base_p, traj_base_v = data["traj_base_p"][mask], data["traj_base_v"][mask]
        traj_drug_p, traj_drug_v = data["traj_drug_p"][mask], data["traj_drug_v"][mask]

        self.state = np.stack([
            build_state_vector(traj_base_p[i], traj_base_v[i], patient_covariates[i])
            for i in range(n)
        ])
        self.next_state = np.stack([
            build_state_vector(traj_drug_p[i], traj_drug_v[i], patient_covariates[i])
            for i in range(n)
        ])
        self.action = data["dose_mg_per_kg"][mask].reshape(-1, 1).astype(np.float32)

        self.base_scalars = {
            "ef": data["ef_base"][mask], "co": data["co_base"][mask],
            "hr": data["hr_base"][mask], "edv": data["edv_base"][mask], "esv": data["esv_base"][mask],
        }
        self.drug_scalars = {
            "ef": data["ef_drug"][mask], "co": data["co_drug"][mask],
            "hr": data["hr_drug"][mask], "edv": data["edv_drug"][mask], "esv": data["esv_drug"][mask],
        }

        if norm_stats is None:
            if split != "train":
                raise ValueError("norm_stats sadece split='train' iken otomatik hesaplanabilir")
            # Decoder normalizasyon istatistikleri hem baseline hem ilaçlı
            # kayıtlardan (ikisi de aynı ölçek/dağılımda fizyolojik sayılar).
            combined_scalars = {
                k: np.concatenate([self.base_scalars[k], self.drug_scalars[k]])
                for k in self.base_scalars
            }
            norm_stats = NormStats.compute(
                state_matrix=np.concatenate([self.state, self.next_state], axis=0),
                action_matrix=self.action,
                scalar_targets=combined_scalars,
            )
        self.norm_stats = norm_stats

        self.state_norm = norm_stats.normalize_state(self.state).astype(np.float32)
        self.next_state_norm = norm_stats.normalize_state(self.next_state).astype(np.float32)
        self.action_norm = norm_stats.normalize_action(self.action).astype(np.float32)

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return {
            "state": torch.from_numpy(self.state_norm[idx]),
            "action": torch.from_numpy(self.action_norm[idx]),
            "next_state": torch.from_numpy(self.next_state_norm[idx]),
            "base_scalars": {k: float(v[idx]) for k, v in self.base_scalars.items()},
            "drug_scalars": {k: float(v[idx]) for k, v in self.drug_scalars.items()},
        }


def load_norm_stats_or_raise(data_dir: str, model_dir: str | None = None) -> NormStats:
    """model_dir verilirse ÖNCE model_dir/norm_stats.json aranır (checkpoint'in
    EĞİTİLDİĞİ norm_stats -- veri dizini sonradan başka bir veri setiyle
    değiştirilmiş olsa bile doğru istatistikleri garanti eder), yoksa
    data_dir/norm_stats.json'a düşer (eski/geriye-uyumlu davranış)."""
    if model_dir is not None:
        model_path = os.path.join(model_dir, "norm_stats.json")
        if os.path.exists(model_path):
            return NormStats.load(model_path)
    path = os.path.join(data_dir, "norm_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} yok -- önce train_jepa.py'yi çalıştırın, o train split'ten "
            "NormStats hesaplayıp bu dosyaya kaydediyor."
        )
    return NormStats.load(path)
