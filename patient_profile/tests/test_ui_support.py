"""
ui_support.py testleri -- Streamlit widget render'ının KENDİSİ burada
test edilmiyor (bkz. modül docstring'i); sadece Streamlit'ten bağımsız
saf mantık: widget türü eşlemesi, dosya-hash cache anahtarı, onay
sonrası birleştirilmiş parametre sözlüğü, düzenlemeyle çözülen çelişkiler.
"""

from patient_profile.covariate_mapping import AdjustedPatientParams
from patient_profile.schema import ExtractedField, PatientCoreParameters
from patient_profile.ui_support import (
    UNKNOWN_OPTION,
    build_confirmed_patient_params,
    hash_uploaded_files,
    resolve_conflicts_overridden_by_edits,
    select_index_for_value,
    select_options_with_unknown,
    widget_kind_for_field,
)


def test_widget_kind_for_select_fields():
    assert widget_kind_for_field("sex") == "select"
    assert widget_kind_for_field("known_av_block_degree") == "select"


def test_widget_kind_for_checkbox_field():
    assert widget_kind_for_field("ascites_present") == "checkbox"


def test_widget_kind_defaults_to_number():
    assert widget_kind_for_field("age") == "number"
    assert widget_kind_for_field("serum_creatinine") == "number"
    assert widget_kind_for_field("baseline_qtc") == "number"


def test_select_options_with_unknown_prepends_unknown_option():
    options = select_options_with_unknown("sex")
    assert options[0] == UNKNOWN_OPTION
    assert options[1:] == ["male", "female"]


def test_select_index_for_known_value():
    assert select_index_for_value("sex", "female") == 2  # [UNKNOWN, male, female]


def test_select_index_for_none_value_is_unknown():
    assert select_index_for_value("sex", None) == 0


def test_select_index_for_unrecognized_value_is_unknown():
    assert select_index_for_value("known_av_block_degree", "bozuk_deger") == 0


def test_hash_uploaded_files_deterministic_for_same_content():
    a = hash_uploaded_files([b"dosya1", b"dosya2"])
    b = hash_uploaded_files([b"dosya1", b"dosya2"])
    assert a == b


def test_hash_uploaded_files_differs_for_different_content():
    a = hash_uploaded_files([b"dosya1"])
    b = hash_uploaded_files([b"dosya2"])
    assert a != b


def test_hash_uploaded_files_is_order_sensitive():
    a = hash_uploaded_files([b"dosya1", b"dosya2"])
    b = hash_uploaded_files([b"dosya2", b"dosya1"])
    assert a != b


def test_hash_uploaded_files_empty_list_is_stable():
    assert hash_uploaded_files([]) == hash_uploaded_files([])


def test_build_confirmed_patient_params_merges_adjusted_and_core():
    adjusted = AdjustedPatientParams(
        weight_kg=80.0, renal_function=0.7, hepatic_function=1.0,
        baseline_hr=90.0, baseline_sbp=130.0, baseline_dbp=85.0,
        potassium_mEqL=5.1, calcium_mgdL=9.0, adjustment_log=["test log"],
    )
    core = PatientCoreParameters(
        age=ExtractedField(value=70),
        height_cm=ExtractedField(value=178),
        known_av_block_degree=ExtractedField(value="second"),
    )
    result = build_confirmed_patient_params(adjusted, core)
    assert result["weight_kg"] == 80.0
    assert result["renal_function"] == 0.7
    assert result["baseline_hr"] == 90.0
    assert result["age"] == 70
    assert result["height_cm"] == 178
    assert result["known_av_block_degree"] == "second"
    assert result["adjustment_log"] == ["test log"]


def test_build_confirmed_patient_params_none_when_core_field_missing():
    adjusted = AdjustedPatientParams(
        weight_kg=76.0, renal_function=1.0, hepatic_function=1.0,
        baseline_hr=78.0, baseline_sbp=125.0, baseline_dbp=80.0,
        potassium_mEqL=4.25, calcium_mgdL=9.5,
    )
    core = PatientCoreParameters()  # hiçbir alan çıkarılmamış
    result = build_confirmed_patient_params(adjusted, core)
    assert result["age"] is None
    assert result["height_cm"] is None
    assert result["known_av_block_degree"] is None


def test_resolve_conflicts_removes_edited_field():
    conflicts = [{"field": "weight_kg", "candidates": []}, {"field": "potassium", "candidates": []}]
    remaining = resolve_conflicts_overridden_by_edits(conflicts, {"weight_kg": 80.0})
    assert remaining == [{"field": "potassium", "candidates": []}]


def test_resolve_conflicts_keeps_unedited_conflicts():
    conflicts = [{"field": "weight_kg", "candidates": []}]
    remaining = resolve_conflicts_overridden_by_edits(conflicts, {"potassium": 5.0})
    assert remaining == conflicts


def test_resolve_conflicts_empty_edits_changes_nothing():
    conflicts = [{"field": "weight_kg", "candidates": []}]
    assert resolve_conflicts_overridden_by_edits(conflicts, {}) == conflicts
