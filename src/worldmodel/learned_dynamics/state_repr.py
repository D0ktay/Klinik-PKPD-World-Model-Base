"""
Durum/Aksiyon Temsili ve Normalizasyon (Faz 2)
=================================================

scripts/generate_dynamics_dataset.py'nin ürettiği ham CircAdapt verisini
(basınç/hacim eğrileri + hasta kovaryatları), JEPA modelinin (bkz. model.py)
beklediği sabit-boyutlu, normalize edilmiş vektörlere çevirir.

Durum vektörü = [yeniden-örneklenmiş LV (sol karıncık -- kalbin vücuda kan
pompalayan ana odacığı) basınç eğrisi (100 nokta)] +
[yeniden-örneklenmiş LV hacim eğrisi (100 nokta)] +
[hasta kovaryatları (11 sürekli + 3 kategorik one-hot = 14)] = 214 boyut.

Aksiyon vektörü (MVP'de tek ilaç -- esmolol) = [normalize edilmiş doz] = 1 boyut.
"""

import json
from dataclasses import dataclass, field

import numpy as np

TRAJECTORY_N_POINTS = 100

CONTINUOUS_PATIENT_FIELDS = [
    "age", "weight_kg", "height_cm",
    "baseline_hr", "baseline_sbp", "baseline_dbp", "baseline_spo2",
    "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
]
COMORBIDITY_CATEGORIES = ["none", "heart_failure", "hypertension"]

PATIENT_COVARIATE_DIM = len(CONTINUOUS_PATIENT_FIELDS) + len(COMORBIDITY_CATEGORIES)  # 14
STATE_DIM = 2 * TRAJECTORY_N_POINTS + PATIENT_COVARIATE_DIM  # 214
# Transient (zaman-içi, atım-atım) pipeline'ın state_dim'i -- bkz.
# build_state_vector()'ın current_hr parametresi. STATE_DIM'in KENDİSİ
# değişmiyor (tek-adımlı MVP geriye dönük uyumlu kalıyor), bu SADECE
# current_hr verildiğinde üretilen vektörün boyutu.
TRANSIENT_STATE_DIM = STATE_DIM + 1  # 215
ACTION_DIM = 1

# Decoder'ın (bkz. train_decoder.py) embedding'den geri çevireceği yorumlanabilir
# fizyolojik sayılar -- EF (ejeksiyon fraksiyonu), CO (kardiyak debi), HR (nabız),
# EDV/ESV (diyastol-sonu/sistol-sonu hacim). Bkz. worldmodel/clinical_metrics.py.
SCALAR_TARGET_FIELDS = ["ef", "co", "hr", "edv", "esv"]


def comorbidity_one_hot(comorbidity: str) -> np.ndarray:
    vec = np.zeros(len(COMORBIDITY_CATEGORIES), dtype=np.float32)
    idx = COMORBIDITY_CATEGORIES.index(comorbidity if comorbidity in COMORBIDITY_CATEGORIES else "none")
    vec[idx] = 1.0
    return vec


def build_patient_covariate_vector(row: dict) -> np.ndarray:
    """row: dataset.npz'den okunan bir kaydın alanları (dict-benzeri, örn. bir
    pandas Series ya da npz sütunlarından indekslenmiş bir sözlük)."""
    continuous = np.array([float(row[f]) for f in CONTINUOUS_PATIENT_FIELDS], dtype=np.float32)
    return np.concatenate([continuous, comorbidity_one_hot(row["comorbidity"])])


def build_state_vector(traj_p: np.ndarray, traj_v: np.ndarray, patient_covariates: np.ndarray,
                        current_hr: float | None = None) -> np.ndarray:
    """
    current_hr: SADECE transient (zaman-içi) pipeline'ın kullandığı, o anki
    (KAREYE ÖZGÜ) nabız -- statik `baseline_hr` kovaryatından FARKLI.
    `resample_trajectory()` zaman eksenini kendi döngüsünün fraksiyonuna
    normalize ettiği için, döngü SÜRESİNİN (dolayısıyla nabzın) mutlak
    büyüklüğü eğrinin şeklinden kayboluyor -- tek-adımlı MVP'de sorun
    değildi (nabız sabitti, decoder'a ayrı hedef olarak veriliyordu), ama
    transient'ta nabız HER KAREDE değişiyor.

    `None` (varsayılan, tek-adımlı `DynamicsDataset`'in davranışı):
    STATE_DIM (214) boyutlu vektör üretir, DAVRANIŞ HİÇ DEĞİŞMEZ. Sayısal
    bir değer verilirse (`TransientDynamicsDataset`): TRANSIENT_STATE_DIM
    (215) boyutlu vektör üretir.
    """
    parts = [
        np.asarray(traj_p, dtype=np.float32),
        np.asarray(traj_v, dtype=np.float32),
        np.asarray(patient_covariates, dtype=np.float32),
    ]
    if current_hr is not None:
        parts.append(np.array([current_hr], dtype=np.float32))
    return np.concatenate(parts)


@dataclass
class NormStats:
    """
    Eğitim kümesi (SADECE train split -- val/test'e sızıntı olmaması için)
    istatistiklerinden hesaplanan z-score (ortalama/std) normalizasyon
    parametreleri. State/action vektörünün HER boyutu için ayrı ayrı.

    Neden gerekli: trajectory değerleri (mmHg, mL) ile hasta kovaryatları
    (yaş, kg, mEq/L) tamamen farklı ölçeklerde -- normalize edilmeden sinir
    ağına verilirse büyük ölçekli boyutlar (örn. hacim mL) eğitimi domine eder.
    """
    state_mean: list = field(default_factory=list)
    state_std: list = field(default_factory=list)
    action_mean: list = field(default_factory=list)
    action_std: list = field(default_factory=list)
    scalar_target_mean: dict = field(default_factory=dict)
    scalar_target_std: dict = field(default_factory=dict)

    @staticmethod
    def compute(state_matrix: np.ndarray, action_matrix: np.ndarray,
                scalar_targets: dict) -> "NormStats":
        # One-hot komorbidite boyutları (son 3 boyut) 0/1 zaten -- std=0
        # olabileceği (tek bir kategori hiç görülmemişse) için 1e-6 ile
        # alt sınırlanıyor, yoksa 0'a bölme oluşur.
        state_std = state_matrix.std(axis=0)
        state_std = np.where(state_std < 1e-6, 1.0, state_std)
        action_std = action_matrix.std(axis=0)
        action_std = np.where(action_std < 1e-6, 1.0, action_std)

        scalar_mean, scalar_std = {}, {}
        for name, values in scalar_targets.items():
            std = float(values.std())
            scalar_mean[name] = float(values.mean())
            scalar_std[name] = std if std > 1e-6 else 1.0

        return NormStats(
            state_mean=state_matrix.mean(axis=0).tolist(),
            state_std=state_std.tolist(),
            action_mean=action_matrix.mean(axis=0).tolist(),
            action_std=action_std.tolist(),
            scalar_target_mean=scalar_mean,
            scalar_target_std=scalar_std,
        )

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - np.asarray(self.state_mean)) / np.asarray(self.state_std)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - np.asarray(self.action_mean)) / np.asarray(self.action_std)

    def normalize_scalar(self, name: str, value):
        return (value - self.scalar_target_mean[name]) / self.scalar_target_std[name]

    def denormalize_scalar(self, name: str, value):
        return value * self.scalar_target_std[name] + self.scalar_target_mean[name]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @staticmethod
    def load(path: str) -> "NormStats":
        with open(path, "r", encoding="utf-8") as f:
            return NormStats(**json.load(f))
