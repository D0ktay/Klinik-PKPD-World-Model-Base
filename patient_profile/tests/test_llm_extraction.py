"""
LLM extraction testleri -- GERÇEK API ÇAĞRISI YAPILMAZ (maliyetli/
kararsız). Gemini'nin döndüreceği yanıtı taklit eden sahte bir client
enjekte edilir, sadece şema parse/hata-yakalama mantığı test edilir.
"""

import pytest

from patient_profile.llm_extraction import (
    LLMExtractionError,
    extract_patient_profile_from_pdf_bytes,
    extract_patient_profile_from_text,
)
from patient_profile.schema import ExtractedField, PatientCoreParameters
from patient_profile.llm_extraction import _normalize_categorical_fields

VALID_RESPONSE_JSON = """
{
  "core_parameters": {
    "age": {"value": 70, "source_quote": "70 yaşında", "source_document": "sample_report_1.txt", "confidence": "high"},
    "sex": {"value": "male", "source_quote": "erkek", "source_document": "sample_report_1.txt", "confidence": "high"},
    "serum_creatinine": {"value": 1.0, "source_quote": "Serum kreatinin: 1.0 mg/dL", "source_document": "sample_report_1.txt", "confidence": "high"}
  },
  "flags": {
    "chronic_conditions": [
      {"name": "Hipertansiyon", "source_quote": "Hipertansiyon (10 yıldır izlemde)", "source_document": "sample_report_1.txt"}
    ]
  },
  "extraction_metadata": {}
}
"""

MALFORMED_RESPONSE_JSON = '{"core_parameters": {"age": "bu bir sayı değil ama şema sayı bekliyor -- olmayan_alan": true'


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, model, contents, config):
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text):
        self.models = _FakeModels(text)


def test_extract_from_text_with_valid_mock_response_parses_correctly():
    profile = extract_patient_profile_from_text(
        document_text="(sentetik metin)", filename="sample_report_1.txt", client=_FakeClient(VALID_RESPONSE_JSON)
    )
    assert profile.core_parameters.age.value == 70
    assert profile.core_parameters.sex.value == "male"
    assert profile.flags.chronic_conditions[0].name == "Hipertansiyon"


def test_extract_from_text_attaches_extraction_metadata():
    profile = extract_patient_profile_from_text(
        document_text="(sentetik metin)", filename="sample_report_1.txt", client=_FakeClient(VALID_RESPONSE_JSON)
    )
    assert profile.extraction_metadata["source_document"] == "sample_report_1.txt"
    assert "llm_model" in profile.extraction_metadata
    assert "extracted_at_utc" in profile.extraction_metadata


def test_extract_from_text_with_malformed_json_raises_llm_extraction_error():
    with pytest.raises(LLMExtractionError):
        extract_patient_profile_from_text(
            document_text="(sentetik metin)", filename="bozuk.txt", client=_FakeClient(MALFORMED_RESPONSE_JSON)
        )


def test_extract_from_text_with_empty_response_raises_error():
    with pytest.raises(LLMExtractionError):
        extract_patient_profile_from_text(document_text="x", filename="bos.txt", client=_FakeClient(""))


def test_extract_from_pdf_bytes_uses_multimodal_path_with_mock_client():
    profile = extract_patient_profile_from_pdf_bytes(
        pdf_bytes=b"%PDF-1.4 (sahte, gercek PDF degil, test icerigi)",
        filename="taranmis_rapor.pdf",
        client=_FakeClient(VALID_RESPONSE_JSON),
    )
    assert profile.core_parameters.age.value == 70


def test_never_returns_silent_default_profile_on_api_exception():
    class _RaisingModels:
        def generate_content(self, model, contents, config):
            raise TimeoutError("simulated network timeout")

    class _RaisingClient:
        def __init__(self):
            self.models = _RaisingModels()

    with pytest.raises(LLMExtractionError):
        extract_patient_profile_from_text(document_text="x", filename="y.txt", client=_RaisingClient())


# --- Kategorik alan normalizasyonu (bug fix regresyon testleri) ---
# Gerçek bir denemede LLM, "Cinsiyet: Erkek" içeren bir raporda
# sex.value için çevrilmemiş "Erkek" (ya da bazı çalıştırmalarda null)
# döndürdü -- şemanın description'ındaki "male"/"female" ipucuna rağmen
# prompt'ta ÖNCEDEN çeviri talimatı yoktu. Kural 8 eklendi (prompt) VE
# bu deterministik normalizasyon katmanı eklendi (LLM talimata her
# zaman uymayabilir diye savunma amaçlı).

UNTRANSLATED_SEX_RESPONSE_JSON = """
{
  "core_parameters": {
    "sex": {"value": "Erkek", "source_quote": "Cinsiyet: Erkek", "source_document": "r.pdf", "confidence": "high"}
  },
  "flags": {}
}
"""


def test_extract_from_text_normalizes_untranslated_sex_value():
    profile = extract_patient_profile_from_text(
        document_text="(sentetik metin)", filename="r.pdf",
        client=_FakeClient(UNTRANSLATED_SEX_RESPONSE_JSON),
    )
    assert profile.core_parameters.sex.value == "male"
    # source_quote DEĞİŞMEMELİ -- sadece value normalize edilir.
    assert profile.core_parameters.sex.source_quote == "Cinsiyet: Erkek"


def test_normalize_categorical_fields_translates_turkish_sex_value():
    core = PatientCoreParameters(sex=ExtractedField(value="Kadın", source_quote="Kadın hasta"))
    normalized = _normalize_categorical_fields(core)
    assert normalized.sex.value == "female"
    assert normalized.sex.source_quote == "Kadın hasta"


def test_normalize_categorical_fields_leaves_canonical_value_unchanged():
    core = PatientCoreParameters(sex=ExtractedField(value="male"))
    normalized = _normalize_categorical_fields(core)
    assert normalized.sex.value == "male"


def test_normalize_categorical_fields_leaves_unrecognized_value_untouched():
    # Eşlemede olmayan bir değer SESSİZCE bir tahmine zorlanmaz --
    # validation.py bunu "out_of_range" olarak işaretleyip kullanıcıya
    # gösterir.
    core = PatientCoreParameters(sex=ExtractedField(value="belirsiz"))
    normalized = _normalize_categorical_fields(core)
    assert normalized.sex.value == "belirsiz"


def test_normalize_categorical_fields_leaves_none_untouched():
    core = PatientCoreParameters()
    normalized = _normalize_categorical_fields(core)
    assert normalized.sex.value is None


def test_normalize_categorical_fields_translates_av_block_degree():
    core = PatientCoreParameters(known_av_block_degree=ExtractedField(value="3. derece"))
    normalized = _normalize_categorical_fields(core)
    assert normalized.known_av_block_degree.value == "third"
