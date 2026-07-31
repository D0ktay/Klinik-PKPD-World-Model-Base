"""
Onay ekranı testleri -- eksik zorunlu alan / çözülmemiş çelişki varsa
`ready_to_confirm=False` olduğunu, ve onaylanmamış (`user_confirmed=False`)
ya da hazır olmayan bir ekranın ASLA `covariate_mapping`'e veri
göndermediğini doğrular.
"""

import pytest

from patient_profile.schema import ExtractedField, PatientCoreParameters, PatientProfile
from patient_profile.review_data import build_review_screen, confirm_and_apply

BASE_PK_PARAMS = dict(
    weight_kg=76, renal_function=1.0, hepatic_function=1.0, baseline_hr=78,
    baseline_sbp=125, baseline_dbp=80, potassium_mEqL=4.25, calcium_mgdL=9.5,
)


def _complete_profile() -> PatientProfile:
    return PatientProfile(
        core_parameters=PatientCoreParameters(
            age=ExtractedField(value=70),
            sex=ExtractedField(value="male"),
            weight_kg=ExtractedField(value=70),
            baseline_heart_rate=ExtractedField(value=82),
            baseline_bp_systolic=ExtractedField(value=130),
        )
    )


def test_ready_to_confirm_true_when_required_fields_present_and_no_conflicts():
    screen = build_review_screen("hasta_test", _complete_profile(), [])
    assert screen.ready_to_confirm is True


def test_ready_to_confirm_false_when_required_field_missing():
    profile = PatientProfile(core_parameters=PatientCoreParameters(age=ExtractedField(value=70)))
    screen = build_review_screen("hasta_test", profile, [])
    assert screen.ready_to_confirm is False


def test_ready_to_confirm_false_when_unresolved_conflict_exists():
    screen = build_review_screen(
        "hasta_test", _complete_profile(), conflicts=[{"field": "serum_creatinine", "candidates": []}]
    )
    assert screen.ready_to_confirm is False


def test_confirm_and_apply_succeeds_when_ready_and_confirmed():
    screen = build_review_screen("hasta_test", _complete_profile(), [])
    result = confirm_and_apply(screen, _complete_profile(), BASE_PK_PARAMS, user_confirmed=True)
    assert result.weight_kg == 70


def test_confirm_and_apply_rejects_when_not_confirmed():
    screen = build_review_screen("hasta_test", _complete_profile(), [])
    with pytest.raises(ValueError):
        confirm_and_apply(screen, _complete_profile(), BASE_PK_PARAMS, user_confirmed=False)


def test_confirm_and_apply_rejects_when_screen_not_ready():
    incomplete_profile = PatientProfile(core_parameters=PatientCoreParameters(age=ExtractedField(value=70)))
    screen = build_review_screen("hasta_test", incomplete_profile, [])
    with pytest.raises(ValueError):
        confirm_and_apply(screen, incomplete_profile, BASE_PK_PARAMS, user_confirmed=True)
