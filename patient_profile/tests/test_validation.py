"""
Validasyon testleri -- her fizyolojik aralık kuralının SINIR DEĞERLER
dahil doğru çalıştığını, çapraz-alan tutarlılık kurallarının doğru
tetiklendiğini, ve validasyonun veriyi asla DEĞİŞTİRMEDİĞİNİ doğrular.
"""

from patient_profile.schema import (
    ChronicCondition,
    ExtractedField,
    PatientCoreParameters,
    PatientFlags,
    PatientProfile,
)
from patient_profile.validation import (
    PHYSIOLOGICAL_RANGES,
    cross_field_consistency_checks,
    validate_core_parameters,
    validate_profile,
)


def _profile_with(**core_field_values) -> PatientProfile:
    fields = {name: ExtractedField(value=value) for name, value in core_field_values.items()}
    return PatientProfile(core_parameters=PatientCoreParameters(**fields))


def test_missing_field_reports_missing():
    core = PatientCoreParameters()
    report = validate_core_parameters(core)
    assert report["age"] == "missing"


def test_age_lower_boundary_ok():
    core = PatientCoreParameters(age=ExtractedField(value=0))
    assert validate_core_parameters(core)["age"] == "ok"


def test_age_upper_boundary_ok():
    core = PatientCoreParameters(age=ExtractedField(value=120))
    assert validate_core_parameters(core)["age"] == "ok"


def test_age_just_below_lower_boundary_out_of_range():
    core = PatientCoreParameters(age=ExtractedField(value=-1))
    assert validate_core_parameters(core)["age"] == "out_of_range"


def test_age_just_above_upper_boundary_out_of_range():
    core = PatientCoreParameters(age=ExtractedField(value=121))
    assert validate_core_parameters(core)["age"] == "out_of_range"


def test_all_physiological_ranges_have_ok_boundaries():
    for field_name, (low, high) in PHYSIOLOGICAL_RANGES.items():
        core = PatientCoreParameters(**{field_name: ExtractedField(value=low)})
        assert validate_core_parameters(core)[field_name] == "ok", f"{field_name} lower bound failed"
        core = PatientCoreParameters(**{field_name: ExtractedField(value=high)})
        assert validate_core_parameters(core)[field_name] == "ok", f"{field_name} upper bound failed"


def test_heart_rate_extreme_value_flagged_out_of_range():
    core = PatientCoreParameters(baseline_heart_rate=ExtractedField(value=900))
    assert validate_core_parameters(core)["baseline_heart_rate"] == "out_of_range"


def test_validate_does_not_mutate_input():
    core = PatientCoreParameters(age=ExtractedField(value=-5))
    before = core.model_dump()
    validate_core_parameters(core)
    assert core.model_dump() == before


def test_cross_field_age_pregnancy_conflict():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(age=ExtractedField(value=0)),
        flags=PatientFlags(pregnancy_status=ExtractedField(value="pregnant_trimester_1")),
    )
    issues = cross_field_consistency_checks(profile)
    assert any(i["rule"] == "age_pregnancy_conflict" for i in issues)


def test_cross_field_sex_pregnancy_conflict():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(sex=ExtractedField(value="male")),
        flags=PatientFlags(pregnancy_status=ExtractedField(value="pregnant_trimester_2")),
    )
    issues = cross_field_consistency_checks(profile)
    assert any(i["rule"] == "sex_pregnancy_conflict" for i in issues)


def test_cross_field_no_conflict_for_not_pregnant_male():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(sex=ExtractedField(value="male")),
        flags=PatientFlags(pregnancy_status=ExtractedField(value="not_pregnant")),
    )
    issues = cross_field_consistency_checks(profile)
    assert issues == []


def test_cross_field_low_ef_without_heart_failure_history_flagged():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(baseline_ejection_fraction=ExtractedField(value=15))
    )
    issues = cross_field_consistency_checks(profile)
    assert any(i["rule"] == "low_ef_without_heart_failure_history" for i in issues)


def test_cross_field_low_ef_with_heart_failure_history_not_flagged():
    profile = PatientProfile(
        core_parameters=PatientCoreParameters(baseline_ejection_fraction=ExtractedField(value=15)),
        flags=PatientFlags(
            chronic_conditions=[
                ChronicCondition(name="Kalp Yetmezliği", source_quote="KY tanılı", source_document="r.pdf")
            ]
        ),
    )
    issues = cross_field_consistency_checks(profile)
    assert not any(i["rule"] == "low_ef_without_heart_failure_history" for i in issues)


def test_validate_profile_combines_field_status_and_consistency_issues():
    profile = _profile_with(age=-5, sex="male")
    result = validate_profile(profile)
    assert "field_status" in result
    assert "consistency_issues" in result
    assert result["field_status"]["age"] == "out_of_range"


# --- Kategorik/boolean alan kapsamı (bug fix regresyon testleri) ---
# Bkz. review_data.py > _validation_status_for(): field_status'ta hiç
# GİRDİSİ olmayan bir alan (önceden sex/known_av_block_degree/
# ascites_present/height_cm/hepatic_encephalopathy_grade) DEĞERİ NE
# OLURSA OLSUN "missing" olarak fallback ediyordu -- bu testler artık
# bu 5 alanın da gerçekten değerlendirildiğini doğruluyor.


def test_sex_missing_when_none():
    core = PatientCoreParameters()
    assert validate_core_parameters(core)["sex"] == "missing"


def test_sex_ok_when_canonical_value():
    core = PatientCoreParameters(sex=ExtractedField(value="male"))
    assert validate_core_parameters(core)["sex"] == "ok"

    core = PatientCoreParameters(sex=ExtractedField(value="female"))
    assert validate_core_parameters(core)["sex"] == "ok"


def test_sex_out_of_range_when_untranslated_value():
    # LLM, "male"/"female" yerine çevrilmemiş bir kelime yazdıysa (örn.
    # "Erkek") -- bu artık "missing" DEĞİL, "out_of_range" (beklenen
    # küme dışı) sayılmalı, çünkü değer VAR, sadece normalize edilmemiş.
    core = PatientCoreParameters(sex=ExtractedField(value="Erkek"))
    assert validate_core_parameters(core)["sex"] == "out_of_range"


def test_known_av_block_degree_ok_for_each_valid_option():
    for value in ("none", "first", "second", "third"):
        core = PatientCoreParameters(known_av_block_degree=ExtractedField(value=value))
        assert validate_core_parameters(core)["known_av_block_degree"] == "ok"


def test_known_av_block_degree_missing_when_none():
    core = PatientCoreParameters()
    assert validate_core_parameters(core)["known_av_block_degree"] == "missing"


def test_known_av_block_degree_out_of_range_for_unrecognized_value():
    core = PatientCoreParameters(known_av_block_degree=ExtractedField(value="3. derece"))
    assert validate_core_parameters(core)["known_av_block_degree"] == "out_of_range"


def test_ascites_present_ok_for_true_and_false():
    core = PatientCoreParameters(ascites_present=ExtractedField(value=True))
    assert validate_core_parameters(core)["ascites_present"] == "ok"
    core = PatientCoreParameters(ascites_present=ExtractedField(value=False))
    assert validate_core_parameters(core)["ascites_present"] == "ok"


def test_ascites_present_missing_when_none():
    core = PatientCoreParameters()
    assert validate_core_parameters(core)["ascites_present"] == "missing"


def test_height_cm_and_hepatic_encephalopathy_grade_now_covered():
    # Bu iki alan sayısal olduğu için PHYSIOLOGICAL_RANGES'e eklendi --
    # daha önce hiç kapsanmıyorlardı.
    core = PatientCoreParameters(
        height_cm=ExtractedField(value=175), hepatic_encephalopathy_grade=ExtractedField(value=2)
    )
    report = validate_core_parameters(core)
    assert report["height_cm"] == "ok"
    assert report["hepatic_encephalopathy_grade"] == "ok"


def test_all_core_parameter_fields_are_covered_by_validate_core_parameters():
    # Regresyon kilidi: PatientCoreParameters'a yeni bir alan eklenip
    # PHYSIOLOGICAL_RANGES/CATEGORICAL_FIELD_OPTIONS/BOOLEAN_FIELDS'in
    # HİÇBİRİNE dahil edilmezse, bu test bunu erkenden yakalasın --
    # aksi halde o alan sessizce "her zaman missing" bug'ını tekrar eder.
    core = PatientCoreParameters()
    report = validate_core_parameters(core)
    assert set(report.keys()) == set(PatientCoreParameters.model_fields.keys())
