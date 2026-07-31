"""
Zaman bazlı birleştirme -- kullanıcı aynı hasta için birden fazla PDF
yüklediyse, her dosyadan ayrı çıkarılmış PatientProfile'ları TEK bir
birleşik PatientProfile'a indirger.

İki farklı birleştirme stratejisi:
  - Kategori A (zamana bağlı değişkenler -- kreatinin, EF, elektrolitler,
    vital bulgular vb.): en güncel `source_date`'e sahip değer kazanır.
    Tarih hiçbir dosyada tespit edilemediyse, yükleme sırası (son
    yüklenen) fallback olarak kullanılır ve confidence="low" işaretlenir.
  - Kalıcı öykü alanları (kronik hastalıklar, ilaç alerjileri, güncel
    ilaçlar, OTC takviyeler): TÜM dosyalardan birikir (union), aynı isim
    tekrar ederse tekilleştirilir, her kaydın kendi kaynağı korunur.

ÖZEL DURUM -- known_av_block_degree: bu alan yapısal olarak
PatientCoreParameters'ta (Kategori A gibi) duruyor, ama KLİNİK OLARAK
kalıcı bir öykü bilgisi (bir kez AV blok tanısı konduysa, sonraki bir
rapor bundan bahsetmiyor diye "düzeldi" anlamına gelmez -- muhtemelen
o rapor bu konuyu hiç ele almamıştır). Bu yüzden GÜVENLİ TARAF seçildi:
en güncel değer değil, EN CİDDİ (en yüksek dereceli) bildirilen değer
kazanır.

Çelişki tespiti: aynı alan için farklı dosyalarda birbirinden
FİZYOLOJİK OLARAK ANLAMLI ÖLÇÜDE farklı (sayısal alanlarda >%30 göreli
fark, kategorik alanlarda herhangi bir uyuşmazlık) iki değer varsa, bu
OTOMATİK ÇÖZÜLMEZ -- `conflicts` listesine eklenir, kullanıcı onay
ekranında ayrıca gösterilir (bkz. review_data.py).
"""

from patient_profile.schema import (
    ChronicCondition,
    CurrentMedication,
    DrugAllergy,
    ExtractedField,
    InteractingSupplement,
    PatientCoreParameters,
    PatientFlags,
    PatientProfile,
)

NUMERIC_CONFLICT_RELATIVE_THRESHOLD = 0.30

_AV_BLOCK_SEVERITY = {"none": 0, "first": 1, "second": 2, "third": 3}

# known_av_block_degree, PatientCoreParameters'ın parçası ama merge
# stratejisi Kategori A'dan farklı (bkz. modül docstring'i) -- bu yüzden
# genel Kategori A döngüsünden ayrı tutulur.
_GENERIC_CORE_FIELDS = [
    name for name in PatientCoreParameters.model_fields if name != "known_av_block_degree"
]


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _significantly_different(a, b) -> bool:
    if _is_numeric(a) and _is_numeric(b):
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom > NUMERIC_CONFLICT_RELATIVE_THRESHOLD
    return a != b


def _merge_time_dependent_field(field_label: str, candidates: list[ExtractedField]) -> tuple[ExtractedField, dict | None]:
    """
    candidates: value'su dolu olan ExtractedField'lar, DOSYA YÜKLEME
    SIRASINA göre (candidates[-1] = en son yüklenen).

    Dönüş: (kazanan ExtractedField, çelişki kaydı ya da None).
    """
    if not candidates:
        return ExtractedField(), None

    distinct_values = {}
    for ef in candidates:
        distinct_values.setdefault(ef.value, []).append(ef)

    conflict = None
    if len(distinct_values) > 1:
        values_list = list(distinct_values.keys())
        has_meaningful_conflict = any(
            _significantly_different(values_list[i], values_list[j])
            for i in range(len(values_list))
            for j in range(i + 1, len(values_list))
        )
        if has_meaningful_conflict:
            conflict = {
                "field": field_label,
                "candidates": [
                    {
                        "value": ef.value,
                        "source_document": ef.source_document,
                        "source_quote": ef.source_quote,
                        "source_date": ef.source_date.isoformat() if ef.source_date else None,
                    }
                    for ef in candidates
                ],
            }

    dated = [ef for ef in candidates if ef.source_date is not None]
    if dated:
        winner = max(dated, key=lambda ef: ef.source_date)
    else:
        winner = candidates[-1].model_copy(update={"confidence": "low"})
    return winner, conflict


def _merge_av_block_degree(candidates: list[ExtractedField]) -> tuple[ExtractedField, dict | None]:
    """
    En güncel değil, EN CİDDİ (en yüksek dereceli) bildirilen değeri
    seçer -- gerekçe için modül docstring'ine bakın.
    """
    if not candidates:
        return ExtractedField(), None

    known = [ef for ef in candidates if ef.value in _AV_BLOCK_SEVERITY]
    if not known:
        return candidates[-1], None

    winner = max(known, key=lambda ef: _AV_BLOCK_SEVERITY[ef.value])

    conflict = None
    distinct_values = {ef.value for ef in known}
    if len(distinct_values) > 1:
        conflict = {
            "field": "known_av_block_degree",
            "note": "En ciddi (en yüksek dereceli) değer seçildi -- kalıcı öykü kabul edilir, bkz. temporal_merge.py.",
            "candidates": [
                {
                    "value": ef.value,
                    "source_document": ef.source_document,
                    "source_quote": ef.source_quote,
                    "source_date": ef.source_date.isoformat() if ef.source_date else None,
                }
                for ef in known
            ],
        }
    return winner, conflict


def _merge_core_parameters(profiles: list[PatientProfile]) -> tuple[PatientCoreParameters, list[dict]]:
    conflicts: list[dict] = []
    merged: dict = {}

    for field_name in _GENERIC_CORE_FIELDS:
        candidates = [
            getattr(p.core_parameters, field_name)
            for p in profiles
            if getattr(p.core_parameters, field_name).value is not None
        ]
        winner, conflict = _merge_time_dependent_field(field_name, candidates)
        merged[field_name] = winner
        if conflict:
            conflicts.append(conflict)

    av_candidates = [
        p.core_parameters.known_av_block_degree
        for p in profiles
        if p.core_parameters.known_av_block_degree.value is not None
    ]
    winner, conflict = _merge_av_block_degree(av_candidates)
    merged["known_av_block_degree"] = winner
    if conflict:
        conflicts.append(conflict)

    return PatientCoreParameters(**merged), conflicts


def _dedupe_by_key(items: list, key_fn) -> list:
    seen = set()
    result = []
    for item in items:
        key = key_fn(item).strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _merge_flags(profiles: list[PatientProfile]) -> tuple[PatientFlags, list[dict]]:
    conflicts: list[dict] = []

    pregnancy_candidates = [p.flags.pregnancy_status for p in profiles if p.flags.pregnancy_status.value is not None]
    pregnancy_winner, pregnancy_conflict = _merge_time_dependent_field("pregnancy_status", pregnancy_candidates)
    if pregnancy_conflict:
        conflicts.append(pregnancy_conflict)

    lactation_candidates = [p.flags.lactation_status for p in profiles if p.flags.lactation_status.value is not None]
    lactation_winner, lactation_conflict = _merge_time_dependent_field("lactation_status", lactation_candidates)
    if lactation_conflict:
        conflicts.append(lactation_conflict)

    all_allergies: list[DrugAllergy] = [a for p in profiles for a in p.flags.drug_allergies]
    all_conditions: list[ChronicCondition] = [c for p in profiles for c in p.flags.chronic_conditions]
    all_medications: list[CurrentMedication] = [m for p in profiles for m in p.flags.current_medications]
    all_supplements: list[InteractingSupplement] = [s for p in profiles for s in p.flags.otc_supplements]

    merged_flags = PatientFlags(
        drug_allergies=_dedupe_by_key(all_allergies, lambda a: a.drug_or_class),
        chronic_conditions=_dedupe_by_key(all_conditions, lambda c: c.name),
        pregnancy_status=pregnancy_winner,
        lactation_status=lactation_winner,
        current_medications=_dedupe_by_key(all_medications, lambda m: m.drug_name),
        otc_supplements=_dedupe_by_key(all_supplements, lambda s: s.name),
    )
    return merged_flags, conflicts


def merge_profiles(profiles: list[PatientProfile]) -> tuple[PatientProfile, list[dict]]:
    """
    Birden fazla dosyadan çıkarılmış PatientProfile'ları TEK bir
    birleşik PatientProfile'a indirger.

    Dönüş: (birleşik_profil, conflicts). conflicts boşsa hiçbir alan
    çelişkili değil demektir -- ama yine de `ready_to_confirm` (bkz.
    review_data.py) tüm zorunlu alanların DOLU olup olmadığını ayrıca
    kontrol eder, conflicts boş olması "hepsi tamam" anlamına gelmez.
    """
    if not profiles:
        raise ValueError("En az bir PatientProfile verilmeli.")
    if len(profiles) == 1:
        return profiles[0], []

    core, core_conflicts = _merge_core_parameters(profiles)
    flags, flag_conflicts = _merge_flags(profiles)

    merged = PatientProfile(
        core_parameters=core,
        flags=flags,
        extraction_metadata={
            "merged_from_documents": [
                p.extraction_metadata.get("source_document", f"document_{i}") for i, p in enumerate(profiles)
            ],
            "document_count": len(profiles),
        },
    )
    return merged, core_conflicts + flag_conflicts
