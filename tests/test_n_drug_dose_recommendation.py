"""
ADIM 5 -- N-ilaç doz önerisi testleri.

streamlit_app.py'deki `len(drugs) == 2` kilidinin kaldırılmasının
(recommend_dose artık HER N için, kullanıcının seçtiği hedef ilaç
için çalışıyor) ve recommend_polypharmacy_dose_scale()'in ADIM 3.1'den
sonra PK-DDI'yi hesaba katmasının DOĞRULANMASI.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from worldmodel.patient import load_drugs, load_patients, load_drug_pk_interactions
from worldmodel.simulation import (
    recommend_dose, recommend_polypharmacy_dose_scale, run_polypharmacy_simulation,
    build_pk_interaction_matrix,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture(scope="module")
def world():
    patients = load_patients(os.path.join(CONFIG_DIR, "patients.yaml"))
    drugs = load_drugs(os.path.join(CONFIG_DIR, "drugs.yaml"))
    pk_interactions = load_drug_pk_interactions(os.path.join(CONFIG_DIR, "drug_pk_interactions.yaml"))
    return {
        "hasta_a": patients["hasta_a"],
        "beta": drugs["beta_bloker"],
        "digoxin": drugs["digoxin"],
        "vazodilator": drugs["vazodilator"],
        "pk_interactions": pk_interactions,
    }


# --- ADIM 3.1'in recommend_polypharmacy_dose_scale'e etkisi (N=3) ----------

def test_recommend_polypharmacy_dose_scale_n3_uses_pk_ddi_when_given(world):
    """
    N_DRUG_AUDIT.md Şüphe A: recommend_polypharmacy_dose_scale() ESKİDEN
    (ADIM 3.1'den önce) PK-seviyeli ilaç etkileşimini HER ZAMAN yok
    sayıyordu -- N=3 kombinasyonda esmolol->digoksin PK etkileşimi
    (Kessler 1987) verilse de verilmese de AYNI sonucu üretiyordu. Bu test,
    drug_keys/pk_interaction_matrix verildiğinde sonucun GERÇEKTEN
    değiştiğini -- yani artık gerçekten hesaba katıldığını -- doğrudan
    ölçer (varsaymaz).
    """
    combo = [world["beta"], world["digoxin"], world["vazodilator"]]
    drug_keys = ["beta_bloker", "digoxin", "vazodilator"]
    pk_interaction_matrix = build_pk_interaction_matrix(drug_keys, world["pk_interactions"])
    assert pk_interaction_matrix, "beklenen PK etkileşim kaydı (esmolol->digoxin) bulunamadı"

    kwargs = dict(n_candidates=6, n_realizations=20, hours=6.0, n_timepoints=25, seed=5)

    without_pk = recommend_polypharmacy_dose_scale(world["hasta_a"], combo, **kwargs)
    with_pk = recommend_polypharmacy_dose_scale(
        world["hasta_a"], combo, drug_keys=drug_keys, pk_interaction_matrix=pk_interaction_matrix, **kwargs,
    )

    # PK etkileşimi digoksin klerensini düşürüp maruziyeti artırdığı için
    # (bkz. configs/drug_pk_interactions.yaml), en azından ADAY listesindeki
    # sürekli istatistiklerin (mean_min_hr -- bradikardi riskinden daha
    # DUYARLI bir metrik, eşik-aşımı gerektirmiyor) bazıları FARKLI olmalı --
    # iki taramanın TAMAMEN özdeş çıkması, PK-DDI'nin hesaba katılmadığının
    # kanıtı olurdu.
    means_without = [c[2]["mean_min_hr"] for c in without_pk["candidates"]]
    means_with = [c[2]["mean_min_hr"] for c in with_pk["candidates"]]
    assert means_without != means_with, (
        "PK-DDI verilmesi sonucu DEĞİŞTİRMEDİ -- Şüphe A'nın düzeltilmediğine işaret eder"
    )


def test_recommend_polypharmacy_dose_scale_default_none_unaffected(world):
    """drug_keys/pk_interaction_matrix verilmezse (varsayılan None), davranış
    ADIM 3.1 ÖNCESİYLE (PK-DDI'siz) AYNI kalmalı -- regresyon kilidi."""
    combo = [world["beta"], world["digoxin"]]
    kwargs = dict(n_candidates=5, n_realizations=15, hours=6.0, n_timepoints=20, seed=9)
    rec1 = recommend_polypharmacy_dose_scale(world["hasta_a"], combo, **kwargs)
    rec2 = recommend_polypharmacy_dose_scale(world["hasta_a"], combo, drug_keys=None, pk_interaction_matrix=None, **kwargs)
    assert rec1["scale"] == pytest.approx(rec2["scale"])


# --- ADIM 5: "hedef ilaç" mantığının N=3+ için de recommend_dose ile -------
# tutarlı çalıştığını doğrula (streamlit_app.py'nin artık yaptığı şey)

def test_target_drug_dose_recommendation_works_for_n3(world):
    """streamlit_app.py artık N≥2'de (eskiden SADECE N=2'de) kullanıcının
    seçtiği hedef ilacı, DİĞER TÜM ilaçları 'zaten kullanılıyor' rolünde
    sabit tutarak optimize ediyor -- bu örüntüyü N=3 için doğrudan test
    eder (recommend_dose + run_polypharmacy_simulation kombinasyonu,
    streamlit_app.py:838-856'daki mantığın birebir aynısı)."""
    combo = [world["beta"], world["digoxin"], world["vazodilator"]]
    mc_kwargs = dict(n_realizations=20, hours=6.0, n_timepoints=25, seed=13)

    mc_result = run_polypharmacy_simulation(world["hasta_a"], combo, **mc_kwargs)

    target_idx = 0  # beta_bloker'ı optimize ediyoruz, digoxin+vazodilator sabit
    other_names = ", ".join(d.display_name for i, d in enumerate(combo) if i != target_idx)

    dose_rec = recommend_dose(
        world["hasta_a"], combo[target_idx],
        polypharmacy_result=mc_result, polypharmacy_description=other_names,
        n_candidates=6, n_realizations=15, hours=6.0, n_timepoints=20,
    )

    assert "dose_mg" in dose_rec
    assert np.isfinite(dose_rec["dose_mg"])
    assert other_names.count(",") == 1  # 2 diğer ilaç ismi listelendi
