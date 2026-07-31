# GÜVENLİK NOTU: Bu modül hassas hasta verisi işler.
# - Yüklenen PDF'ler ve çıkarılan veriler, harici LLM API'sine (Gemini)
#   gönderilir -- bu, verinin üçüncü taraf bir servise iletildiği anlamına gelir.
# - Prod ortamında: veri saklama süresi, şifreleme, erişim logları,
#   ve KVKK/GDPR uyumluluğu ayrıca değerlendirilmelidir (bu görevin kapsamı dışı).
# - Bu implementasyon bir demo/POC niteliğindedir, prodüksiyon güvenlik
#   gereksinimlerini karşılamak için ek çalışma gerekir.
"""
LLM extraction katmanı -- Gemini API çağrısı + prompt yönetimi.

MİMARİ PRENSİP (asla ihlal etme): LLM burada karar verici değil, veri
çıkarıcıdır. LLM hiçbir klerens değeri, doz önerisi veya klinik yargı
üretmez -- sadece metinden (veya taranmış PDF görüntüsünden) şemaya
uygun yapılandırılmış veri çıkarır. Tüm hesaplamalar (Cockcroft-Gault,
Child-Pugh, allometrik ölçekleme) covariate_mapping.py'deki deterministik
Python fonksiyonlarıyla yapılır -- bu dosyada YOK.

`google-genai` (güncel birleşik Gemini SDK'sı) kullanılıyor -- eski
`google-generativeai` paketi yerine, çünkü bu proje sıfırdan kuruluyor
ve google-genai aktif olarak sürdürülen pakettir.
"""

import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError

from patient_profile.schema import ExtractedField, PatientCoreParameters, PatientFlags, PatientProfile

# Model adı hardcode EDİLMEDİ -- ortam değişkeninden okunabilir,
# okunamazsa bu sabit varsayılan kullanılır. Değiştirmek için kod
# değişikliği gerekmez.
# "gemini-2.5-pro" -- Google'ın yeni ücretsiz/API-only hesaplar için
# ERİŞİMİ KAPATTIĞI bir model (canlı testte doğrulandı: "no longer
# available to new users" / free-tier kotası limit=0) -- "gemini-flash-latest"
# Google'ın kendi önerdiği, sürekli güncel tutulan bir ALIAS (belirli bir
# model sürümüne SABİTLENMİYOR), bu yüzden varsayılan olarak seçildi.
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL_NAME", "gemini-flash-latest")


class _PatientProfileExtraction(BaseModel):
    """
    Gemini'ye `response_schema` olarak verilen şema -- `PatientProfile`'ın
    KENDİSİ DEĞİL. `PatientProfile.extraction_metadata` serbest bir `dict`
    (LLM tarafından hiç doldurulmuyor, `_validate_and_attach_metadata`
    tarafından sonradan ekleniyor) -- ama pydantic bu tip için üretilen JSON
    şemasında `additionalProperties: true` döndürüyor, ve Gemini Developer
    API (Enterprise/Vertex modu DEĞİL) bunu desteklemiyor ("additionalProperties
    is only supported in Gemini Enterprise Agent Platform mode" hatası).
    Bu yüzden LLM'e sadece gerçekten doldurması istenen iki alanı
    (core_parameters/flags) içeren bu alt-şema veriliyor.
    """

    core_parameters: PatientCoreParameters = Field(default_factory=PatientCoreParameters)
    flags: PatientFlags = Field(default_factory=PatientFlags)

PROMPT_TEMPLATE = """\
Sen bir tıbbi doküman analiz sistemisin. Görevin, verilen hasta raporu
metninden, belirtilen JSON şemasına uygun yapılandırılmış veri çıkarmaktır.

MUTLAK KURALLAR:
1. Bir bilgi metinde AÇIKÇA belirtilmemişse, o alanı null bırak.
   ASLA tahmin etme, ASLA "muhtemelen normaldir" gibi bir varsayımda bulunma,
   ASLA tipik/ortalama bir değer uydurma.
2. Her bulduğun değer için, değerin geçtiği ORİJİNAL CÜMLEYİ
   source_quote alanına birebir kopyala. Parafraz etme, metinden aynen al.
3. Sayısal bir değer belirsizse (örn. "kreatinin yüksek" yazıyor ama
   sayı verilmemişse), value'yu null bırak, ama source_quote'a o cümleyi
   yine de ekle ve confidence: "low" işaretle.
4. Birimler farklıysa (örn. kreatinin µmol/L cinsinden verilmişse),
   mg/dL'ye çevir ve source_quote'ta orijinal değeri/birimi koru.
   Emin değilsen dönüştürme yapma, value'yu null bırak,
   source_quote'a orijinal ifadeyi yaz.
5. Aynı bilgi metinde birden fazla farklı değerle geçiyorsa
   (örn. iki farklı tarihte iki farklı kreatinin), YALNIZCA bu dosyadaki
   EN SON tarihli değeri value olarak ver, ama diğerini de
   extraction_metadata içinde not düş.
6. Hiçbir alan için klinik yorum, risk değerlendirmesi veya öneri üretme.
   Sen sadece bir çıkarım (extraction) motorusun, bir teşhis/karar aracı değilsin.
7. Şemada olmayan bir bilgi metinde bulunsa bile, şemaya eklemeye çalışma.
   Sadece verilen şemayı doldur.
8. Bazı alanların açıklamasında (description) izin verilen SABİT değerler
   tırnak içinde İngilizce verilir (örn. sex için "male" | "female",
   known_av_block_degree için "none" | "first" | "second" | "third").
   Rapor bu bilgiyi Türkçe veya farklı bir ifadeyle içeriyorsa (örn.
   "Cinsiyet: Erkek", "Kadın hasta", "3. derece AV blok"), value alanına
   HER ZAMAN açıklamada verilen İngilizce sabit değerlerden birini yaz
   (örn. "Erkek" -> "male", "Kadın" -> "female") -- value'yu ASLA
   rapordaki orijinal Türkçe kelimeyle doldurma ve ASLA bu yüzden null
   bırakma (rapor bilgiyi açıkça içeriyorsa bu, kural 1'deki "bilgi
   metinde YOK" durumu DEĞİL). source_quote'a yine rapordaki ORİJİNAL
   (çevrilmemiş) cümleyi koy -- kural 2 burada da geçerli, sadece value
   normalize edilir. Açıklamada sabit bir değer kümesi VERİLMEYEN diğer
   tüm alanlarda (örn. serbest metin alanları) bu kural geçerli değildir.

RAPOR DOSYA ADI: {filename}
RAPOR METNİ:
{document_text}
"""


class LLMExtractionError(Exception):
    """API hatası, timeout, rate limit, ya da şemaya uymayan JSON için."""


def _build_client():
    try:
        from google import genai
    except ImportError as exc:
        raise LLMExtractionError(
            "'google-genai' paketi kurulu değil -- requirements.txt'e eklendi mi kontrol et."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMExtractionError(
            "GEMINI_API_KEY ortam değişkeni ayarlanmamış -- Gemini API'ye erişim için gerekli."
        )
    return genai.Client(api_key=api_key)


def _generate_raw_json(contents, model_name: str, client=None) -> str:
    """
    Gemini API'ye tek bir çağrı yapar, ham JSON metnini döndürür.

    `client` parametresi test edilebilirlik için var -- testlerde gerçek
    API çağrısı yapılmaz (ADIM 9), sahte bir client (contents/response'u
    taklit eden) buraya enjekte edilir.
    """
    if client is None:
        client = _build_client()

    from google.genai import types

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_PatientProfileExtraction,
            ),
        )
    except Exception as exc:
        raise LLMExtractionError(f"Gemini API çağrısı başarısız (ağ/rate-limit/timeout): {exc}") from exc

    if not response.text:
        raise LLMExtractionError("Gemini API boş yanıt döndürdü.")
    return response.text


# Kategorik alanlar için "kabul edilebilir ham değer" -> "şemanın beklediği
# kanonik değer" eşlemesi -- SAVUNMA KATMANI (kural 8, PROMPT_TEMPLATE'te
# LLM'e zaten bu normalizasyonu KENDİSİNİN yapması talimatı veriliyor, ama
# LLM bir talimata her zaman birebir uyacağı GARANTİ DEĞİL -- bu yüzden
# ayrıca burada, deterministik Python koduyla tekrarlanıyor). validation.py
# BİLEREK bunu yapmıyor (o katman "immutable", veriyi asla değiştirmez,
# sadece etiketler) -- normalizasyon, ham LLM çıktısının bizim veri
# modelimize (PatientProfile) dönüştüğü TEK sınır noktası olan bu dosyada
# yaşamalı. Eşlemede olmayan bir değer (örn. gerçekten tanınmayan bir
# ifade) DOKUNULMADAN bırakılır -- validation.py bunu "out_of_range"
# (beklenen küme dışı) olarak işaretleyip kullanıcıya gösterir, biz
# burada SESSİZCE bir tahminde bulunmayız (bkz. modül docstring'i, "LLM
# karar verici değil" prensibiyle aynı ruh: kod da veri UYDURMAZ).
_CATEGORICAL_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "sex": {
        "erkek": "male",
        "kadın": "female",
        "kadin": "female",
    },
    "known_av_block_degree": {
        "yok": "none",
        "birinci derece": "first", "1. derece": "first", "1.derece": "first",
        "ikinci derece": "second", "2. derece": "second", "2.derece": "second",
        "üçüncü derece": "third", "3. derece": "third", "3.derece": "third", "tam blok": "third",
    },
}


def _normalize_categorical_fields(core: PatientCoreParameters) -> PatientCoreParameters:
    """
    `_CATEGORICAL_VALUE_ALIASES`'taki her alan için, ham değer (küçük
    harfe çevrilip boşlukları kırpılarak) eşleme tablosunda bulunursa
    kanonik değere çevrilir. Zaten kanonik olan (örn. "male") ya da
    eşlemede hiç bulunmayan değerler DOKUNULMADAN kalır.
    """
    updates: dict[str, ExtractedField] = {}
    for field_name, aliases in _CATEGORICAL_VALUE_ALIASES.items():
        ef = getattr(core, field_name)
        if isinstance(ef.value, str):
            canonical = aliases.get(ef.value.strip().lower())
            if canonical is not None and canonical != ef.value:
                updates[field_name] = ef.model_copy(update={"value": canonical})
    if not updates:
        return core
    return core.model_copy(update=updates)


def _validate_and_attach_metadata(raw_json: str, filename: str, model_name: str) -> PatientProfile:
    try:
        extracted = _PatientProfileExtraction.model_validate_json(raw_json)
    except ValidationError as exc:
        raise LLMExtractionError(
            f"'{filename}': LLM yanıtı beklenen şemaya uymuyor -- bu dosya işlenemedi, tekrar deneyin. Detay: {exc}"
        ) from exc

    return PatientProfile(
        core_parameters=_normalize_categorical_fields(extracted.core_parameters),
        flags=extracted.flags,
        extraction_metadata={
            "source_document": filename,
            "llm_model": model_name,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def extract_patient_profile_from_text(
    document_text: str, filename: str, model_name: str | None = None, client=None
) -> PatientProfile:
    """
    Metin katmanı olan (taranmamış) bir PDF'ten çıkarılmış ham metni
    Gemini'ye gönderir, PatientProfile olarak parse eder.
    """
    model_name = model_name or DEFAULT_GEMINI_MODEL
    prompt = PROMPT_TEMPLATE.format(filename=filename, document_text=document_text)
    raw_json = _generate_raw_json(prompt, model_name, client=client)
    return _validate_and_attach_metadata(raw_json, filename, model_name)


def extract_patient_profile_from_pdf_bytes(
    pdf_bytes: bytes, filename: str, model_name: str | None = None, client=None
) -> PatientProfile:
    """
    Taranmış (metin katmanı yetersiz) bir PDF için -- ham PDF byte'larını
    doğrudan Gemini'ye multimodal girdi olarak verir (bkz.
    file_ingestion.py modül docstring'i: neden yerel OCR yerine bu yol
    seçildi).
    """
    model_name = model_name or DEFAULT_GEMINI_MODEL
    prompt_text = PROMPT_TEMPLATE.format(
        filename=filename,
        document_text="(Bu bir taranmış belge -- metin ekli PDF görüntüsünden okunmalı.)",
    )

    if client is None:
        client = _build_client()
    from google.genai import types

    contents = [
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        prompt_text,
    ]
    raw_json = _generate_raw_json(contents, model_name, client=client)
    return _validate_and_attach_metadata(raw_json, filename, model_name)
