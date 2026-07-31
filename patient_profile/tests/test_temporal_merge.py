"""
Zaman bazlı birleştirme testleri -- en güncel tarihli değerin kazandığını,
tarih yoksa yükleme sırasının fallback olduğunu, kalıcı öykü alanlarının
UNION+tekilleştirildiğini, ve anlamlı farkların conflicts'e düştüğünü
doğrular.
"""

from datetime import date

from patient_profile.schema import (
    ChronicCondition,
    ExtractedField,
    PatientCoreParameters,
    PatientFlags,
    PatientProfile,
)
from patient_profile.temporal_merge import merge_profiles


def _profile(creatinine=None, creatinine_date=None, weight=None, conditions=None) -> PatientProfile:
    core_kwargs = {}
    if creatinine is not None:
        core_kwargs["serum_creatinine"] = ExtractedField(
            value=creatinine, source_document="doc.pdf", source_date=creatinine_date
        )
    if weight is not None:
        core_kwargs["weight_kg"] = ExtractedField(value=weight, source_document="doc.pdf")
    return PatientProfile(
        core_parameters=PatientCoreParameters(**core_kwargs),
        flags=PatientFlags(chronic_conditions=conditions or []),
    )


def test_single_profile_returned_unchanged_without_merge_logic():
    p = _profile(creatinine=1.0)
    merged, conflicts = merge_profiles([p])
    assert merged is p
    assert conflicts == []


def test_most_recent_dated_value_wins():
    older = _profile(creatinine=1.0, creatinine_date=date(2026, 1, 1))
    newer = _profile(creatinine=1.05, creatinine_date=date(2026, 3, 1))
    merged, _ = merge_profiles([older, newer])
    assert merged.core_parameters.serum_creatinine.value == 1.05


def test_small_difference_between_dated_values_is_not_a_conflict():
    older = _profile(creatinine=1.0, creatinine_date=date(2026, 1, 1))
    newer = _profile(creatinine=1.05, creatinine_date=date(2026, 3, 1))  # %5 fark
    _, conflicts = merge_profiles([older, newer])
    assert conflicts == []


def test_large_difference_between_dated_values_is_a_conflict():
    older = _profile(creatinine=1.0, creatinine_date=date(2026, 1, 1))
    newer = _profile(creatinine=2.0, creatinine_date=date(2026, 3, 1))  # %100 fark
    merged, conflicts = merge_profiles([older, newer])
    assert merged.core_parameters.serum_creatinine.value == 2.0  # yine de en güncel kazanır
    assert any(c["field"] == "serum_creatinine" for c in conflicts)


def test_no_date_falls_back_to_upload_order_with_low_confidence():
    first_uploaded = _profile(creatinine=1.0)
    last_uploaded = _profile(creatinine=1.8)
    merged, conflicts = merge_profiles([first_uploaded, last_uploaded])
    assert merged.core_parameters.serum_creatinine.value == 1.8
    assert merged.core_parameters.serum_creatinine.confidence == "low"


def test_chronic_conditions_union_and_deduped_case_insensitively():
    p1 = _profile(conditions=[ChronicCondition(name="Hipertansiyon", source_quote="q1", source_document="d1")])
    p2 = _profile(conditions=[ChronicCondition(name="hipertansiyon", source_quote="q2", source_document="d2")])
    p3 = _profile(conditions=[ChronicCondition(name="Diyabet", source_quote="q3", source_document="d3")])
    merged, _ = merge_profiles([p1, p2, p3])
    names = {c.name.lower() for c in merged.flags.chronic_conditions}
    assert names == {"hipertansiyon", "diyabet"}
    assert len(merged.flags.chronic_conditions) == 2


def test_field_present_in_only_one_document_is_preserved():
    p1 = _profile(weight=70)
    p2 = _profile(creatinine=1.0)
    merged, _ = merge_profiles([p1, p2])
    assert merged.core_parameters.weight_kg.value == 70
    assert merged.core_parameters.serum_creatinine.value == 1.0
