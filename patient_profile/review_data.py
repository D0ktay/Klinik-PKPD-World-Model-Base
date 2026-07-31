"""
Kullanıcı onay ekranı için veri yapısı -- ARAYÜZ KODU DEĞİL. Streamlit
tarafı (`streamlit_app.py`) bu yapıyı okuyup gösterecek; burada sadece
onay ekranını besleyecek, frontend'den bağımsız veri modeli var.

ZORUNLU AKIŞ: kullanıcı onayı olmadan hiçbir veri
`covariate_mapping.py` fonksiyonlarına gönderilmemeli --
`build_review_screen()` bunu ÜRETİR ama UYGULAMAZ; `confirm_and_apply()`
sadece `ready_to_confirm=True` VE kullanıcının gerçekten onayladığı
(`user_confirmed=True`) durumda `apply_patient_covariates()`'i çağırır.
"""

from typing import Literal, Optional

from pydantic import BaseModel

from patient_profile.covariate_mapping import AdjustedPatientParams, apply_patient_covariates
from patient_profile.schema import PatientCoreParameters, PatientProfile
from patient_profile.validation import validate_profile

# Kategori A'nın hangi alt kümesi "zorunlu" (simülasyona girmeden önce
# dolu ya da kullanıcı tarafından elle girilmiş olması gereken) --
# geri kalan Kategori A alanları (örn. baseline_qtc, magnesium) varsa
# kullanılır ama eksikse simülasyon yine de çalışabilir (mevcut
# Patient dataclass'ında zaten opsiyonel/varsayılanlı).
REQUIRED_FIELD_NAMES = ("age", "sex", "weight_kg", "baseline_heart_rate", "baseline_bp_systolic")


class ReviewItem(BaseModel):
    field_name: str
    value: Optional[float | int | str | bool] = None
    source_quote: Optional[str] = None
    source_document: Optional[str] = None
    validation_status: Literal["ok", "out_of_range", "conflict", "missing"]
    editable: bool = True


class ReviewScreen(BaseModel):
    patient_id: str
    items: list[ReviewItem]
    conflicts: list[dict]
    ready_to_confirm: bool


def _validation_status_for(field_name: str, field_status: dict, conflicted_fields: set[str]) -> Literal[
    "ok", "out_of_range", "conflict", "missing"
]:
    if field_name in conflicted_fields:
        return "conflict"
    return field_status.get(field_name, "missing")


def build_review_screen(patient_id: str, profile: PatientProfile, conflicts: list[dict]) -> ReviewScreen:
    """
    `profile` -- temporal_merge.merge_profiles()'ın çıktısı (tek dosya
    yüklendiyse birleştirmesiz de kullanılabilir). `conflicts` -- aynı
    fonksiyonun döndürdüğü çelişki listesi.
    """
    validation_report = validate_profile(profile)
    field_status = validation_report["field_status"]
    conflicted_fields = {c["field"] for c in conflicts if "field" in c}

    items: list[ReviewItem] = []
    for field_name in PatientCoreParameters.model_fields:
        ef = getattr(profile.core_parameters, field_name)
        status = _validation_status_for(field_name, field_status, conflicted_fields)
        items.append(
            ReviewItem(
                field_name=field_name,
                value=ef.value,
                source_quote=ef.source_quote,
                source_document=ef.source_document,
                validation_status=status,
            )
        )

    all_conflicts = conflicts + [
        {"rule": issue["rule"], "detail": issue["detail"]} for issue in validation_report["consistency_issues"]
    ]

    required_ok = all(
        getattr(profile.core_parameters, name).value is not None and name not in conflicted_fields
        for name in REQUIRED_FIELD_NAMES
    )
    no_unresolved_conflicts = len(conflicts) == 0
    ready_to_confirm = required_ok and no_unresolved_conflicts

    return ReviewScreen(
        patient_id=patient_id,
        items=items,
        conflicts=all_conflicts,
        ready_to_confirm=ready_to_confirm,
    )


def confirm_and_apply(
    review_screen: ReviewScreen,
    profile: PatientProfile,
    base_pk_params: dict,
    user_confirmed: bool,
) -> AdjustedPatientParams:
    """
    Onaylanmamış veri asla `covariate_mapping.py`'ye gönderilmez --
    bu fonksiyon o zorunluluğu uygular. `user_confirmed`, kullanıcının
    onay ekranında gerçekten "onayla" butonuna bastığını temsil eder
    (Streamlit tarafında bir session_state flag'i olacak).
    """
    if not review_screen.ready_to_confirm:
        raise ValueError(
            "Onay ekranı hazır değil (zorunlu alan eksik ya da çözülmemiş çelişki var) -- "
            "covariate_mapping'e gönderilemez."
        )
    if not user_confirmed:
        raise ValueError("Kullanıcı onayı alınmadı -- covariate_mapping'e gönderilemez.")
    return apply_patient_covariates(profile, base_pk_params)
