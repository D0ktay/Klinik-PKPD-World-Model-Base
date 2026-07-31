"""
`scripts/generate_transient_dataset.py::determine_split()` için testler --
Karar 2'nin (veri büyütmesi sırasında split ataması) DOĞRU uygulandığını
kanıtlıyor. Saf fonksiyon, CircAdapt/rastgelelik İÇERMEZ -- hızlı testler.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_transient_dataset import determine_split, SPLIT_BOUNDARY_PATIENT_INDEX


# data/transient_dataset_large/dataset.npz üzerinde ADIM 0'da DOĞRUDAN
# doğrulanmış, gerçek test hasta index'leri -- bu liste UYDURULMADI.
EXISTING_TEST_PATIENT_INDICES = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129]
EXISTING_VAL_PATIENT_INDICES = [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128]


def test_existing_test_patients_unchanged():
    """Karar 2'nin ÇEKİRDEK gereksinimi: mevcut 13 test hastası, yeni
    kodda da BİREBİR AYNI split'e ('test') düşmeli -- aksi halde önceki/
    sonraki R² karşılaştırması anlamsızlaşır."""
    for i in EXISTING_TEST_PATIENT_INDICES:
        assert determine_split(i) == "test", f"hasta {i} artık 'test' değil -- REGRESYON"


def test_existing_val_patients_unchanged():
    for i in EXISTING_VAL_PATIENT_INDICES:
        assert determine_split(i) == "val", f"hasta {i} artık 'val' değil -- REGRESYON"


def test_existing_train_patients_unchanged():
    """i<130 aralığında test/val OLMAYAN her index -- train olmalı (eski davranış)."""
    for i in range(SPLIT_BOUNDARY_PATIENT_INDEX):
        if i % 10 < 8:
            assert determine_split(i) == "train", f"hasta {i} artık 'train' değil -- REGRESYON"


def test_new_patients_always_train():
    """i>=130 (yeni eklenen hastalar) HER ZAMAN train -- i%10 kuralına
    hiç girmemeli, val/test'e asla sızmamalı."""
    for i in [130, 131, 139, 140, 149, 199, 999, 2599]:
        assert determine_split(i) == "train", f"hasta {i} (>=130) train DIŞINDA bir split'e düştü"


def test_boundary_patient_index_is_130():
    """Sınırın tam olarak 130'da olduğunu doğrular -- 129 eski kurala
    tabi (test), 130 her zaman train."""
    assert SPLIT_BOUNDARY_PATIENT_INDEX == 130
    assert determine_split(129) == "test"   # 129 % 10 == 9
    assert determine_split(130) == "train"  # sınırın tam üzerinde, artık her zaman train
