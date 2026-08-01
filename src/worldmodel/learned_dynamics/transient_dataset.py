"""
PyTorch Dataset -- Zaman-İçi (Transient) Trajectory'ler (Faz 4)
===================================================================

`scripts/generate_transient_dataset.py`'nin ürettiği `data/transient_dataset/
dataset.npz`'i (SATIR=kare) okur, her trajectory'nin ARDIŞIK kare çiftlerini
`(state_t, action_t, next_state_t)` üçlülerine düzleştirir.

`DynamicsDataset` (tek-adımlı MVP, `dataset.py`) ile AYNI dış arayüzü
(`__getitem__` sözlük anahtarları, `.norm_stats` özelliği) sağlar -- bu
sayede `train_jepa.py`/`train_decoder.py` HİÇ değiştirilmeden, sadece
`dataset_cls=TransientDynamicsDataset` parametresiyle bu sınıfı da
kullanabiliyor.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodel.learned_dynamics.state_repr import (
    build_patient_covariate_vector, build_state_vector, NormStats, PATIENT_COVARIATE_DIM,
)

_PATIENT_FIELD_NAMES = [
    "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp", "baseline_dbp",
    "baseline_spo2", "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
]


class TransientDynamicsDataset(Dataset):
    def __init__(self, npz_path: str, split: str, norm_stats: NormStats | None = None):
        data = np.load(npz_path, allow_pickle=True)
        # NpzFile lazy/zip-backed -- data[key] HER cagrida tum diziyi yeniden
        # okuyup unpickle ediyor. Ihtiyac duyulan alanlari BIR KEZ materyalize
        # ediyoruz (dongu icinde data[...] ERISIMI YOK) -- 26K satirlik veride
        # bu duzeltme olmadan kurulum ~55 dk, duzeltmeyle ~1 dk suruyor.
        needed_fields = _PATIENT_FIELD_NAMES + [
            "comorbidity", "traj_p", "traj_v", "current_hr", "ef", "co", "edv", "esv",
            "conc_mg_L", "trajectory_id", "frame_idx", "split",
        ]
        arrays = {f: data[f] for f in needed_fields}

        mask = arrays["split"] == split
        if not mask.any():
            raise ValueError(f"'{split}' split'inde hiç örnek yok -- {npz_path}")

        row_indices = np.where(mask)[0]
        n = len(row_indices)

        patient_covariates = np.zeros((n, PATIENT_COVARIATE_DIM), dtype=np.float32)
        for local_i, row_idx in enumerate(row_indices):
            row = {f: arrays[f][row_idx] for f in _PATIENT_FIELD_NAMES}
            row["comorbidity"] = str(arrays["comorbidity"][row_idx])
            patient_covariates[local_i] = build_patient_covariate_vector(row)

        states = np.stack([
            build_state_vector(arrays["traj_p"][row_idx], arrays["traj_v"][row_idx],
                                patient_covariates[local_i], current_hr=float(arrays["current_hr"][row_idx]))
            for local_i, row_idx in enumerate(row_indices)
        ])
        scalars = {
            "ef": arrays["ef"][row_indices], "co": arrays["co"][row_indices],
            "hr": arrays["current_hr"][row_indices],
            "edv": arrays["edv"][row_indices], "esv": arrays["esv"][row_indices],
        }
        actions_raw = arrays["conc_mg_L"][row_indices]
        trajectory_ids = arrays["trajectory_id"][row_indices]
        frame_idxs = arrays["frame_idx"][row_indices]

        # --- Trajectory'ye göre grupla, HER grup içinde frame_idx'e göre sırala,
        # ardından ardışık kare ÇİFTLERİNİ (i, i+1) çıkar. Kesilmiş (truncated)
        # trajectory'ler otomatik olarak daha az çift üretir -- SABİT bir uzunluk
        # VARSAYILMAZ.
        pair_state_idx, pair_next_idx, pair_baseline_idx = [], [], []
        order = np.lexsort((frame_idxs, trajectory_ids))  # once trajectory_id, sonra frame_idx
        sorted_traj_ids = trajectory_ids[order]
        boundaries = np.where(np.diff(sorted_traj_ids) != 0)[0] + 1
        groups = np.split(order, boundaries)
        for group in groups:
            # group zaten frame_idx'e göre sıralı (lexsort garantisi) --
            # group[0] HER ZAMAN o trajectory'nin frame_idx=0 (ilaç-öncesi
            # baseline) karesi (run_transient_trajectory frame_idx=0'ı HER
            # ZAMAN ilk ekler, bkz. transient_integration.py).
            for j in range(len(group) - 1):
                pair_state_idx.append(group[j])
                pair_next_idx.append(group[j + 1])
                pair_baseline_idx.append(group[0])

        pair_state_idx = np.array(pair_state_idx, dtype=np.int64)
        pair_next_idx = np.array(pair_next_idx, dtype=np.int64)
        pair_baseline_idx = np.array(pair_baseline_idx, dtype=np.int64)
        if len(pair_state_idx) == 0:
            raise ValueError(f"'{split}' split'inde ardışık kare çifti üretilemedi (her trajectory tek kareli mi?)")

        self.state = states[pair_state_idx]
        self.next_state = states[pair_next_idx]
        # "Hedef-durum" (goal-state) deneyi için -- HER çiftin AİT OLDUĞU
        # trajectory'nin GERÇEK ilaç-öncesi baseline karesi (bkz.
        # model.py > GoalConditionedPredictor). Eski (delta-tabanlı)
        # eğitim yolu bu alanı hiç KULLANMAZ -- sadece EK bir alan,
        # geriye dönük uyumluluğu bozmaz.
        self.baseline_state = states[pair_baseline_idx]
        # Aksiyon HEDEF karenin (next_state'in) konsantrasyonu -- bu, o geçişi
        # ÜRETEN aksiyon (bkz. transient_integration.py: frame_idx=i+1'in
        # conc_mg_L'i, modeli frame_idx=i'den i+1'e taşımak için kullanıldı).
        self.action = actions_raw[pair_next_idx].reshape(-1, 1).astype(np.float32)

        self.base_scalars = {k: v[pair_state_idx] for k, v in scalars.items()}
        self.drug_scalars = {k: v[pair_next_idx] for k, v in scalars.items()}

        if norm_stats is None:
            if split != "train":
                raise ValueError("norm_stats sadece split='train' iken otomatik hesaplanabilir")
            combined_scalars = {k: np.concatenate([self.base_scalars[k], self.drug_scalars[k]])
                                 for k in self.base_scalars}
            norm_stats = NormStats.compute(
                state_matrix=np.concatenate([self.state, self.next_state], axis=0),
                action_matrix=self.action,
                scalar_targets=combined_scalars,
            )
        self.norm_stats = norm_stats

        self.state_norm = norm_stats.normalize_state(self.state).astype(np.float32)
        self.next_state_norm = norm_stats.normalize_state(self.next_state).astype(np.float32)
        self.baseline_state_norm = norm_stats.normalize_state(self.baseline_state).astype(np.float32)
        self.action_norm = norm_stats.normalize_action(self.action).astype(np.float32)

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return {
            "state": torch.from_numpy(self.state_norm[idx]),
            "action": torch.from_numpy(self.action_norm[idx]),
            "next_state": torch.from_numpy(self.next_state_norm[idx]),
            "baseline_state": torch.from_numpy(self.baseline_state_norm[idx]),
            "base_scalars": {k: float(v[idx]) for k, v in self.base_scalars.items()},
            "drug_scalars": {k: float(v[idx]) for k, v in self.drug_scalars.items()},
        }
