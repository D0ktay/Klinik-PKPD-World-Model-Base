"""
ADIM 2 -- N-ilac genellemesi calismasi (ADIM 3+) boyunca N=1 ve N=2
davranisinin SAYISAL OLARAK DEGISMEDIGINI kilitleyen regresyon testleri.

tests/golden/n1_n2_golden.npz + n1_n2_golden_scalars.json,
tests/golden/generate_golden_snapshots.py ile (ADIM 3'ten ONCE, mevcut
kod uzerinde) uretildi. Bu dosya SADECE karsilastirma yapar -- yeniden
uretmez. Bir test kirilirsa bu, N=1/2 davranisinda bir REGRESYON
oldugu anlamina gelir (MUTLAK KURAL #1) -- testi gevsetmek degil, kodu
duzeltmek dogru tepki.

Kapsam, generate_golden_snapshots.py ile BIREBIR ayni senaryolari kapsar:
N=1/N=2 istatistiksel motor (additive+Loewe), CircAdapt, doz onerisi,
AV-duyarli ilac + anormal potasyum (hasta_c_hiperkalemi), zit yonlu
kombinasyon (beta_bloker + nicardipine).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from worldmodel.patient import load_drugs, load_patients, load_drug_interactions, load_drug_pk_interactions
from worldmodel.simulation import (
    run_monte_carlo, recommend_dose, run_polypharmacy_simulation, run_polypharmacy_simulation_loewe,
    build_interaction_matrix, build_pk_interaction_matrix, recommend_polypharmacy_dose_scale,
)
from integrate_drug_with_circadapt import run_comparison, run_polypharmacy_comparison

from golden.generate_golden_snapshots import (
    mc_kwargs, DOSE_REC_KWARGS, array_payload, circadapt_payload, circadapt_scalars,
    dose_rec_scalars, scale_rec_scalars, CONFIG_DIR,
)

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


@pytest.fixture(scope="module")
def golden_arrays():
    return np.load(os.path.join(GOLDEN_DIR, "n1_n2_golden.npz"))


@pytest.fixture(scope="module")
def golden_scalars():
    with open(os.path.join(GOLDEN_DIR, "n1_n2_golden_scalars.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def world():
    patients = load_patients(os.path.join(CONFIG_DIR, "patients.yaml"))
    drugs = load_drugs(os.path.join(CONFIG_DIR, "drugs.yaml"))
    interactions = load_drug_interactions(os.path.join(CONFIG_DIR, "drug_interactions.yaml"))
    pk_interactions = load_drug_pk_interactions(os.path.join(CONFIG_DIR, "drug_pk_interactions.yaml"))
    return {
        "hasta_a": patients["hasta_a"],
        "hasta_k": patients["hasta_c_hiperkalemi"],
        "beta": drugs["beta_bloker"],
        "digoxin": drugs["digoxin"],
        "nicardipine": drugs["nicardipine"],
        "interactions": interactions,
        "pk_interactions": pk_interactions,
    }


def assert_arrays_match(golden_arrays, prefix, result):
    payload = array_payload(prefix, result)
    for key, arr in payload.items():
        np.testing.assert_allclose(
            arr, golden_arrays[key], rtol=0, atol=0,
            err_msg=f"REGRESYON: {key} altin anlik goruntuden sapiyor",
        )


def assert_circadapt_arrays_match(golden_arrays, prefix, result):
    payload = circadapt_payload(prefix, result)
    for key, arr in payload.items():
        np.testing.assert_allclose(
            arr, golden_arrays[key], rtol=0, atol=0, equal_nan=True,
            err_msg=f"REGRESYON: {key} altin anlik goruntuden sapiyor",
        )


def assert_scalars_match(golden, actual, path=""):
    if isinstance(golden, dict):
        for k, v in golden.items():
            assert_scalars_match(v, actual[k], f"{path}.{k}")
        return
    if isinstance(golden, float):
        assert actual == pytest.approx(golden, rel=1e-12, abs=1e-12), \
            f"REGRESYON: {path} sapiyor ({actual} != {golden})"
        return
    assert actual == golden, f"REGRESYON: {path} sapiyor ({actual!r} != {golden!r})"


# --- N=1 -----------------------------------------------------------------

@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n1_monte_carlo_no_regression(world, golden_arrays, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    result = run_monte_carlo(patient, world["beta"], **mc_kwargs())
    assert_arrays_match(golden_arrays, f"n1_{label}_mc", result)


@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n1_recommend_dose_no_regression(world, golden_scalars, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    rec = recommend_dose(patient, world["beta"], **DOSE_REC_KWARGS)
    assert_scalars_match(golden_scalars[f"n1_{label}_recommend_dose"], dose_rec_scalars(rec))


@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n1_circadapt_no_regression(world, golden_arrays, golden_scalars, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    result = run_comparison(patient, world["beta"])
    assert_circadapt_arrays_match(golden_arrays, f"n1_{label}_circadapt", result)
    assert_scalars_match(golden_scalars[f"n1_{label}_circadapt"], circadapt_scalars(result))


# --- N=2: beta_bloker + digoxin ------------------------------------------

@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n2_additive_no_regression(world, golden_arrays, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    drug_keys = ["beta_bloker", "digoxin"]
    interaction_matrix = build_interaction_matrix(drug_keys, world["interactions"])
    pk_interaction_matrix = build_pk_interaction_matrix(drug_keys, world["pk_interactions"])
    combo = [world["beta"], world["digoxin"]]
    result = run_polypharmacy_simulation(
        patient, combo, interaction_matrix=interaction_matrix,
        drug_keys=drug_keys, pk_interaction_matrix=pk_interaction_matrix, **mc_kwargs(),
    )
    assert_arrays_match(golden_arrays, f"n2_{label}_additive", result)


@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n2_loewe_no_regression(world, golden_arrays, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    combo = [world["beta"], world["digoxin"]]
    result = run_polypharmacy_simulation_loewe(patient, combo, **mc_kwargs())
    assert_arrays_match(golden_arrays, f"n2_{label}_loewe", result)


@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n2_dose_scale_no_regression(world, golden_scalars, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    combo = [world["beta"], world["digoxin"]]
    rec = recommend_polypharmacy_dose_scale(patient, combo, **DOSE_REC_KWARGS)
    assert_scalars_match(golden_scalars[f"n2_{label}_dose_scale"], scale_rec_scalars(rec))


@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n2_recommend_dose_no_regression(world, golden_scalars, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    drug_keys = ["beta_bloker", "digoxin"]
    interaction_matrix = build_interaction_matrix(drug_keys, world["interactions"])
    pk_interaction_matrix = build_pk_interaction_matrix(drug_keys, world["pk_interactions"])
    combo = [world["beta"], world["digoxin"]]
    additive = run_polypharmacy_simulation(
        patient, combo, interaction_matrix=interaction_matrix,
        drug_keys=drug_keys, pk_interaction_matrix=pk_interaction_matrix, **mc_kwargs(),
    )
    rec = recommend_dose(
        patient, world["beta"], polypharmacy_result=additive,
        polypharmacy_description=world["digoxin"].display_name, **DOSE_REC_KWARGS,
    )
    assert_scalars_match(golden_scalars[f"n2_{label}_recommend_dose"], dose_rec_scalars(rec))


@pytest.mark.parametrize("label", ["hasta_a", "hasta_k"])
def test_n2_circadapt_no_regression(world, golden_arrays, golden_scalars, label):
    patient = world["hasta_a"] if label == "hasta_a" else world["hasta_k"]
    combo = [world["beta"], world["digoxin"]]
    result = run_polypharmacy_comparison(patient, combo)
    assert_circadapt_arrays_match(golden_arrays, f"n2_{label}_circadapt", result)
    assert_scalars_match(golden_scalars[f"n2_{label}_circadapt"], circadapt_scalars(result))


# --- N=2: zit yonlu kombinasyon -------------------------------------------

def test_n2_opposite_direction_additive_no_regression(world, golden_arrays):
    combo = [world["beta"], world["nicardipine"]]
    result = run_polypharmacy_simulation(world["hasta_a"], combo, **mc_kwargs())
    assert_arrays_match(golden_arrays, "n2_opposite_additive", result)


def test_n2_opposite_direction_loewe_now_returns_grouped_result_not_error(world, golden_scalars):
    """
    ADIM 3.4 (Şüphe D / ADR-4) KASITLI davranış değişikliği: golden
    snapshot alındığında (ADIM 2) bu kombinasyon ValueError fırlatıyordu
    (mesaj golden_scalars["n2_opposite_loewe_error"]'da donduruldu, bkz.
    generate_golden_snapshots.py). pd.grouped_loewe_combined_effect()
    eklendikten SONRA artık ValueError YERİNE tanımlı bir sonuç üretiyor --
    bu, kullanıcının ADIM 3 talimatında AÇIKÇA istediği, dondurulmuş
    eski davranışla KARŞILAŞTIRILABİLİR bir değişiklik (MUTLAK KURAL #1'in
    istisnası -- kural N=1/N=2'nin SESSİZCE değişmemesini istiyor, bu
    burada açıkça belgeleniyor ve test ediliyor).
    """
    combo = [world["beta"], world["nicardipine"]]
    result = run_polypharmacy_simulation_loewe(world["hasta_a"], combo, **mc_kwargs())

    assert np.isfinite(result.hr_runs).all()
    assert np.isfinite(result.sbp_runs).all()
    assert (result.hr_runs >= 0).all()
    assert (result.sbp_runs >= 0).all()
    # Eskiden bu satıra hiç ulaşılamıyordu (ValueError) -- artık golden
    # snapshot'taki additive yol sonucuyla (n2_opposite_additive) AYNI
    # büyüklük mertebesinde, makul bir HR profili üretiyor.
    assert 30 < result.hr_runs.mean() < 130


def test_n2_opposite_direction_circadapt_no_regression(world, golden_arrays, golden_scalars):
    combo = [world["beta"], world["nicardipine"]]
    result = run_polypharmacy_comparison(world["hasta_a"], combo)
    assert_circadapt_arrays_match(golden_arrays, "n2_opposite_circadapt", result)
    assert_scalars_match(golden_scalars["n2_opposite_circadapt"], circadapt_scalars(result))
