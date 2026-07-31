"""
Validasyon katmanı -- fizyolojik olarak MÜMKÜN aralık kontrolü ve
çapraz-alan tutarlılık kontrolü.

ÖNEMLİ: Bu aralıklar "normal" (klinik referans aralığı) DEĞİL,
"fizyolojik olarak mümkün" sınırlardır -- amaç, veri girişi/OCR/LLM
hatası sonucu ortaya çıkabilecek imkânsız değerleri (örn. yaş=-5,
nabız=900) elemek. Alarm eşiği (örn. "K>5.5 hiperkalemi say") ayrı,
klinik bir katman -- bu dosyanın kapsamında DEĞİL.

Validasyon veriyi ASLA otomatik düzeltmez/silmez (immutable) -- sadece
her alana bir durum etiketi ("ok" | "out_of_range" | "missing") ekler.
Karar (veriyi kabul et/reddet/düzelt) kullanıcıya kalır (bkz.
review_data.py).
"""

from typing import Literal

from patient_profile.schema import PatientCoreParameters, PatientProfile

ValidationStatus = Literal["ok", "out_of_range", "missing"]

# Fizyolojik olarak MÜMKÜN sınırlar (normal aralık değil -- bkz. modül
# docstring'i). (age: yıl, weight_kg/height_cm: kg/cm, serum_creatinine/
# bilirubin_total/albumin: mg/dL veya g/dL, inr: birimsiz oran,
# baseline_heart_rate: bpm, baseline_bp_*: mmHg, potassium/calcium: mEq/L
# veya mg/dL, baseline_ejection_fraction: %, baseline_qtc: ms,
# hepatic_encephalopathy_grade: 0-4 arası ordinal derece.)
PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "weight_kg": (1, 300),
    "height_cm": (30, 250),
    "serum_creatinine": (0.1, 15),
    "bilirubin_total": (0.1, 30),
    "albumin": (1, 6),
    "inr": (0.5, 10),
    "baseline_heart_rate": (20, 250),
    "baseline_bp_systolic": (50, 250),
    "baseline_bp_diastolic": (30, 150),
    "potassium": (1.5, 9),
    "magnesium": (0.5, 5),
    "calcium": (4, 15),
    "baseline_ejection_fraction": (5, 80),
    "baseline_qtc": (250, 700),
    "hepatic_encephalopathy_grade": (0, 4),
}

# Sabit bir seçenek kümesi olan (sayısal ARALIK kavramı anlamsız olan)
# alanlar -- PHYSIOLOGICAL_RANGES'teki low<=value<=high mantığı bunlara
# uygulanamaz, ayrı bir KÜME ÜYELİĞİ kontrolü gerekir.
#
# BUG NOTU (bu sözlük eklenmeden önceki durum): bu alanlar hiçbir yerde
# değerlendirilmiyordu -- review_data.py > _validation_status_for()'daki
# `field_status.get(field_name, "missing")` fallback'i yüzünden, DEĞER
# GERÇEKTEN DOLU olsa bile (örn. sex="male") onay ekranında HER ZAMAN
# "missing" (Bulunamadı) gösteriliyordu. Ayrıca LLM'in şemanın "male"/
# "female" ipucunu Türkçe rapordan çevirmeden ("Erkek" gibi) aynen
# kopyalaması da mümkün -- bu durumda değer VAR ama BEKLENEN kümenin
# DIŞINDA, bu da "missing" değil "out_of_range" (beklenen küme dışı)
# olarak işaretlenmeli ki kullanıcı onay ekranında bunu görüp düzeltsin.
CATEGORICAL_FIELD_OPTIONS: dict[str, frozenset[str]] = {
    "sex": frozenset({"male", "female"}),
    "known_av_block_degree": frozenset({"none", "first", "second", "third"}),
}

# Boolean alanlar -- "aralık" ya da "küme üyeliği" değil, sadece tip
# kontrolü (gerçekten bool mu) gerekir.
BOOLEAN_FIELDS: frozenset[str] = frozenset({"ascites_present"})


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_core_parameters(core: PatientCoreParameters) -> dict[str, ValidationStatus]:
    """
    PatientCoreParameters'ın HER alanı için bir durum döndürür --
    PHYSIOLOGICAL_RANGES'teki sayısal alanlar (aralık kontrolü),
    CATEGORICAL_FIELD_OPTIONS'taki kategorik alanlar (küme üyeliği) ve
    BOOLEAN_FIELDS'teki boolean alan (tip kontrolü). Bu üç kümenin
    BİRLİKTE PatientCoreParameters'ın TÜM alanlarını kapsadığından emin
    olun -- kapsanmayan bir alan, review_data.py'nin "missing" fallback'i
    yüzünden değeri ne olursa olsun hep "Bulunamadı" görünür (bkz. modül
    içindeki CATEGORICAL_FIELD_OPTIONS yorumundaki bug notu).
    """
    report: dict[str, ValidationStatus] = {}
    for field_name in PatientCoreParameters.model_fields:
        ef = getattr(core, field_name)
        if field_name in PHYSIOLOGICAL_RANGES:
            low, high = PHYSIOLOGICAL_RANGES[field_name]
            if ef.value is None:
                report[field_name] = "missing"
            elif _is_numeric(ef.value) and (low <= ef.value <= high):
                report[field_name] = "ok"
            elif _is_numeric(ef.value):
                report[field_name] = "out_of_range"
            else:
                # sayısal olması beklenirken sayısal değilse (şema zaten
                # bunu engeller, ama ekstra güvenlik için) missing say.
                report[field_name] = "missing"
        elif field_name in CATEGORICAL_FIELD_OPTIONS:
            if ef.value is None:
                report[field_name] = "missing"
            elif ef.value in CATEGORICAL_FIELD_OPTIONS[field_name]:
                report[field_name] = "ok"
            else:
                report[field_name] = "out_of_range"
        elif field_name in BOOLEAN_FIELDS:
            if ef.value is None:
                report[field_name] = "missing"
            elif isinstance(ef.value, bool):
                report[field_name] = "ok"
            else:
                report[field_name] = "out_of_range"
    return report


def cross_field_consistency_checks(profile: PatientProfile) -> list[dict]:
    """
    Tek bir alanın aralığıyla değil, alanlar ARASI tutarlılıkla ilgili
    kontroller. Bunlar da veriyi REDDETMEZ -- sadece işaretler
    (`conflicts`/`inconsistencies` listesine eklenir, kullanıcı onay
    ekranında görür).
    """
    issues: list[dict] = []
    core = profile.core_parameters
    age = core.age.value
    sex = core.sex.value
    pregnancy = profile.flags.pregnancy_status.value
    ef_percent = core.baseline_ejection_fraction.value
    chronic_names = [c.name.lower() for c in profile.flags.chronic_conditions]

    is_pregnant_status = pregnancy is not None and pregnancy not in ("not_pregnant", "unknown")

    if age is not None and age < 1 and is_pregnant_status:
        issues.append({
            "rule": "age_pregnancy_conflict",
            "detail": f"age={age} ile pregnancy_status={pregnancy} birlikte tutarsız (1 yaşından küçük hasta gebe olamaz).",
        })

    if sex == "male" and is_pregnant_status:
        issues.append({
            "rule": "sex_pregnancy_conflict",
            "detail": f"sex=male ile pregnancy_status={pregnancy} birlikte tutarsız.",
        })

    if ef_percent is not None and _is_numeric(ef_percent) and ef_percent < 20:
        heart_failure_mentioned = any(
            "kalp yetmezliği" in name or "heart failure" in name for name in chronic_names
        )
        if not heart_failure_mentioned:
            issues.append({
                "rule": "low_ef_without_heart_failure_history",
                "detail": (
                    f"baseline_ejection_fraction={ef_percent}% (%20'nin altı, ciddi düşük) "
                    "ama kronik hastalık listesinde kalp yetmezliği geçmiyor -- uyumsuzluk, "
                    "reddedilmedi, sadece işaretlendi."
                ),
            })

    return issues


def validate_profile(profile: PatientProfile) -> dict:
    """
    Tam validasyon raporu: her Kategori A alanı için durum +
    çapraz-alan tutarlılık sorunları. Ham `profile`'ı DEĞİŞTİRMEZ.
    """
    return {
        "field_status": validate_core_parameters(profile.core_parameters),
        "consistency_issues": cross_field_consistency_checks(profile),
    }
