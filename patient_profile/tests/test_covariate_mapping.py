"""
Kovaryat eşleme testleri -- Cockcroft-Gault, Child-Pugh, allometrik
ölçekleme fonksiyonları ELLE HESAPLANMIŞ bilinen doğru sonuçlarla
karşılaştırılır. apply_patient_covariates() için de eksik-veri
fallback davranışı (varsayılana düşme + adjustment_log) doğrulanır.
"""

import pytest

from patient_profile.covariate_mapping import (
    allometric_scale,
    apply_patient_covariates,
    child_pugh_class,
    child_pugh_score,
    child_pugh_to_hepatic_function,
    cockcroft_gault,
    crcl_to_renal_function,
)
from patient_profile.schema import ExtractedField, PatientCoreParameters, PatientProfile

BASE_PK_PARAMS = dict(
    weight_kg=76,
    renal_function=1.0,
    hepatic_function=1.0,
    baseline_hr=78,
    baseline_sbp=125,
    baseline_dbp=80,
    potassium_mEqL=4.25,
    calcium_mgdL=9.5,
)


# --- Cockcroft-Gault ---
# Elle hesap: 70 yaş, 70 kg, kreatinin 1.0 mg/dL, erkek
# CrCl = ((140-70)*70) / (72*1.0) = 4900/72 = 68.0555...
def test_cockcroft_gault_matches_hand_calculation_male():
    crcl = cockcroft_gault(age=70, weight_kg=70, serum_creatinine=1.0, sex="male")
    assert crcl == pytest.approx(68.0556, abs=0.01)


def test_cockcroft_gault_female_applies_085_factor():
    male_crcl = cockcroft_gault(age=70, weight_kg=70, serum_creatinine=1.0, sex="male")
    female_crcl = cockcroft_gault(age=70, weight_kg=70, serum_creatinine=1.0, sex="female")
    assert female_crcl == pytest.approx(male_crcl * 0.85, abs=0.001)


def test_cockcroft_gault_raises_on_missing_param():
    with pytest.raises(ValueError):
        cockcroft_gault(age=70, weight_kg=70, serum_creatinine=None, sex="male")


def test_cockcroft_gault_raises_on_zero_creatinine():
    with pytest.raises(ValueError):
        cockcroft_gault(age=70, weight_kg=70, serum_creatinine=0, sex="male")


def test_crcl_to_renal_function_reference_100_gives_1():
    assert crcl_to_renal_function(100.0) == pytest.approx(1.0)


def test_crcl_to_renal_function_half_reference_gives_half():
    assert crcl_to_renal_function(50.0) == pytest.approx(0.5)


def test_crcl_to_renal_function_clamped_at_zero():
    assert crcl_to_renal_function(-10.0) == 0.0


# --- Child-Pugh ---
# Elle hesap: bilirubin 1.5 (<2 -> 1), albumin 4.0 (>3.5 -> 1),
# INR 1.2 (<1.7 -> 1), asit yok (-> 1), ensefalopati 0 (-> 1)
# Toplam = 5 -> Sınıf A
def test_child_pugh_score_all_normal_gives_minimum_score_5():
    score = child_pugh_score(1.5, 4.0, 1.2, False, 0)
    assert score == 5
    assert child_pugh_class(score) == "A"


# Elle hesap: bilirubin 5 (>3 -> 3), albumin 2.0 (<2.8 -> 3),
# INR 3.0 (>2.3 -> 3), asit var (-> 2), ensefalopati 4 (>2 -> 3)
# Toplam = 3+3+3+2+3 = 14 -> Sınıf C
def test_child_pugh_score_all_severe_gives_high_score():
    score = child_pugh_score(5.0, 2.0, 3.0, True, 4)
    assert score == 14
    assert child_pugh_class(score) == "C"


def test_child_pugh_class_boundaries():
    assert child_pugh_class(6) == "A"
    assert child_pugh_class(7) == "B"
    assert child_pugh_class(9) == "B"
    assert child_pugh_class(10) == "C"


def test_child_pugh_score_raises_on_missing_criterion():
    with pytest.raises(ValueError):
        child_pugh_score(1.5, 4.0, 1.2, False, None)


def test_child_pugh_to_hepatic_function_class_a_near_1():
    assert child_pugh_to_hepatic_function(5) == pytest.approx(1.0)
    assert child_pugh_to_hepatic_function(6) == pytest.approx(0.95)


def test_child_pugh_to_hepatic_function_class_c_lowest():
    assert child_pugh_to_hepatic_function(15) == pytest.approx(0.2)


def test_child_pugh_to_hepatic_function_monotonically_decreasing():
    values = [child_pugh_to_hepatic_function(s) for s in range(5, 16)]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


# --- Allometrik ölçekleme ---
# Elle hesap: 10 L/saat referans (70kg) -> 35kg hasta, exponent=0.75
# 10 * (35/70)**0.75 = 10 * 0.5**0.75 = 10 * 0.5946 = 5.946
def test_allometric_scale_matches_hand_calculation():
    result = allometric_scale(reference_clearance=10.0, reference_weight_kg=70.0, patient_weight_kg=35.0)
    assert result == pytest.approx(5.9460, abs=0.001)


def test_allometric_scale_same_weight_returns_reference_unchanged():
    result = allometric_scale(reference_clearance=10.0, reference_weight_kg=70.0, patient_weight_kg=70.0)
    assert result == pytest.approx(10.0)


# --- apply_patient_covariates: eksik veri fallback ---
def test_apply_patient_covariates_uses_default_when_creatinine_missing():
    profile = PatientProfile(core_parameters=PatientCoreParameters(age=ExtractedField(value=70)))
    adjusted = apply_patient_covariates(profile, BASE_PK_PARAMS)
    assert adjusted.renal_function == BASE_PK_PARAMS["renal_function"]
    assert any("renal ayarlama uygulanmadı" in entry for entry in adjusted.adjustment_log)


def test_apply_patient_covariates_derives_renal_function_when_data_complete():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(
            age=ExtractedField(value=70),
            weight_kg=ExtractedField(value=70),
            serum_creatinine=ExtractedField(value=1.0),
            sex=ExtractedField(value="male"),
        )
    )
    adjusted = apply_patient_covariates(profile, BASE_PK_PARAMS)
    assert adjusted.renal_function == pytest.approx(0.6806, abs=0.001)
    assert adjusted.weight_kg == 70


def test_apply_patient_covariates_derives_hepatic_function_when_data_complete():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(
            bilirubin_total=ExtractedField(value=1.5),
            albumin=ExtractedField(value=4.0),
            inr=ExtractedField(value=1.2),
            ascites_present=ExtractedField(value=False),
            hepatic_encephalopathy_grade=ExtractedField(value=0),
        )
    )
    adjusted = apply_patient_covariates(profile, BASE_PK_PARAMS)
    assert adjusted.hepatic_function == pytest.approx(1.0)


def test_apply_patient_covariates_falls_back_to_default_weight_when_missing():
    profile = PatientProfile()
    adjusted = apply_patient_covariates(profile, BASE_PK_PARAMS)
    assert adjusted.weight_kg == BASE_PK_PARAMS["weight_kg"]
    assert any("popülasyon-ortalaması varsayılan" in entry for entry in adjusted.adjustment_log)
