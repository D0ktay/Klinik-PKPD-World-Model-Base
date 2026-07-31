"""
Şema testleri -- PatientProfile'ın doğru validate/reddettiğini,
eksik/null alanları kabul ettiğini, kaynak alıntısı olmadan liste
elemanı oluşturulamadığını doğrular.
"""

import pytest
from pydantic import ValidationError

from patient_profile.schema import (
    ChronicCondition,
    ExtractedField,
    PatientCoreParameters,
    PatientFlags,
    PatientProfile,
)


def test_empty_patient_profile_builds_with_all_nulls():
    profile = PatientProfile()
    assert profile.core_parameters.age.value is None
    assert profile.flags.chronic_conditions == []
    assert profile.extraction_metadata == {}


def test_extracted_field_accepts_null_value_with_confidence_low():
    ef = ExtractedField(value=None, source_quote="kreatinin yüksek", confidence="low")
    assert ef.value is None
    assert ef.confidence == "low"


def test_extracted_field_rejects_invalid_confidence_literal():
    with pytest.raises(ValidationError):
        ExtractedField(value=1.0, confidence="very_high")


def test_chronic_condition_requires_source_quote_and_document():
    with pytest.raises(ValidationError):
        ChronicCondition(name="Hipertansiyon")  # source_quote/source_document eksik


def test_chronic_condition_valid_with_all_required_fields():
    cond = ChronicCondition(name="Hipertansiyon", source_quote="HT tanılı hasta", source_document="rapor1.pdf")
    assert cond.name == "Hipertansiyon"


def test_patient_core_parameters_child_pugh_fields_present():
    core = PatientCoreParameters()
    assert hasattr(core, "ascites_present")
    assert hasattr(core, "hepatic_encephalopathy_grade")


def test_patient_flags_default_lists_are_empty_not_shared_mutable():
    f1 = PatientFlags()
    f2 = PatientFlags()
    f1.chronic_conditions.append(
        ChronicCondition(name="X", source_quote="x", source_document="d")
    )
    assert f2.chronic_conditions == []  # default_factory doğru izole ediyor mu


def test_patient_profile_model_json_schema_generates_without_error():
    schema = PatientProfile.model_json_schema()
    assert "properties" in schema
    assert "core_parameters" in schema["properties"]


def test_patient_profile_round_trips_through_json():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(age=ExtractedField(value=45, source_document="r.pdf"))
    )
    dumped = profile.model_dump_json()
    restored = PatientProfile.model_validate_json(dumped)
    assert restored.core_parameters.age.value == 45


def test_known_av_block_degree_accepts_string_values():
    core = PatientCoreParameters(known_av_block_degree=ExtractedField(value="second"))
    assert core.known_av_block_degree.value == "second"
