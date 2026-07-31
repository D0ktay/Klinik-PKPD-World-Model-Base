"""
Hasta profili çıkarım şeması (Pydantic v2).

Her alan üç durumdan birinde olabilir: bulundu (value + source_quote/
source_document/source_date/confidence dolu), null (rapor bu bilgiyi
içermiyor -- value=None), ya da çelişkili (temporal_merge.py'nin
conflicts listesine düşer, value burada yine None kalabilir). HİÇBİR
ALAN LLM TARAFINDAN VARSAYIM/TAHMİNLE DOLDURULMAZ -- bu, llm_extraction.py
prompt'unda da zorunlu kılınır.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExtractedField(BaseModel):
    """
    Tek bir çıkarılmış değeri, kaynağıyla birlikte taşıyan kutu.

    value None ise: rapor(lar) bu bilgiyi hiç içermiyor demektir --
    varsayılan/tahmini bir değerle DOLDURULMAZ. source_quote, değerin
    çıkarıldığı orijinal cümledir (LLM tarafından parafraz edilmeden
    birebir kopyalanır) -- değer null olsa bile (örn. "kreatinin
    yüksek" gibi belirsiz bir ifade varsa) source_quote dolu, value
    null, confidence "low" olabilir.
    """

    value: Optional[float | int | str | bool] = None
    source_quote: Optional[str] = None
    source_document: Optional[str] = None
    source_date: Optional[date] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None


class PatientCoreParameters(BaseModel):
    """
    Kategori A -- PK/PD + covariate_mapping.py formüllerine (Cockcroft-Gault,
    Child-Pugh, allometrik ölçekleme) doğrudan giren zorunlu sayısal/
    kategorik parametreler.
    """

    age: ExtractedField = Field(default_factory=ExtractedField, description="Yaş (yıl)")
    sex: ExtractedField = Field(default_factory=ExtractedField, description='"male" | "female"')
    weight_kg: ExtractedField = Field(default_factory=ExtractedField, description="Kilo (kg)")
    height_cm: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Boy (cm) -- opsiyonel, VKİ (vücut kitle indeksi) hesaplamak için",
    )
    serum_creatinine: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Serum kreatinin (mg/dL) -- böbreğin kanı ne kadar iyi süzdüğünü gösteren bir kan tahlili değeri, Cockcroft-Gault (böbrek klerensi) formülüne girer",
    )
    bilirubin_total: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Total bilirubin (mg/dL) -- karaciğer fonksiyonunun göstergelerinden biri, Child-Pugh (karaciğer fonksiyon skoru) hesabına girer",
    )
    albumin: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Albümin (g/dL) -- karaciğerin ürettiği bir kan proteini, karaciğer fonksiyonunun göstergelerinden biri",
    )
    inr: ExtractedField = Field(
        default_factory=ExtractedField,
        description="INR (kanın pıhtılaşma hızının normale göre oranı) -- karaciğer fonksiyonunun göstergelerinden biri, Child-Pugh'a girer",
    )
    baseline_heart_rate: ExtractedField = Field(
        default_factory=ExtractedField, description="Bazal (dinlenim) nabız (bpm)"
    )
    baseline_bp_systolic: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Bazal sistolik tansiyon (mmHg) -- kalp kasılırken ölçülen yüksek tansiyon değeri",
    )
    baseline_bp_diastolic: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Bazal diyastolik tansiyon (mmHg) -- kalp gevşerken ölçülen düşük tansiyon değeri",
    )
    potassium: ExtractedField = Field(default_factory=ExtractedField, description="Potasyum (mEq/L)")
    magnesium: ExtractedField = Field(default_factory=ExtractedField, description="Magnezyum (mg/dL)")
    calcium: ExtractedField = Field(default_factory=ExtractedField, description="Kalsiyum (mg/dL)")
    baseline_ejection_fraction: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Bazal ejeksiyon fraksiyonu / EF (%) -- kalbin her atışta kanın yüzde kaçını pompaladığı, varsa",
    )
    baseline_qtc: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Bazal QTc (ms) -- EKG'de kalbin elektriksel olarak toparlanma süresi, uzaması aritmi riskiyle ilişkili, varsa",
    )
    known_av_block_degree: ExtractedField = Field(
        default_factory=ExtractedField,
        description='Bilinen AV blok (kalbin elektrik iletim gecikmesi) derecesi -- "none" | "first" | "second" | "third"',
    )
    # Child-Pugh'un asit/ensefalopati kriterleri -- ADIM 0/1 incelemesinde
    # PatientFlags altında değil burada (Kategori A, covariate_mapping.py'nin
    # doğrudan girdisi) tutuluyor çünkü Child-Pugh puanlaması sayısal/
    # kategorik bir hesaplamanın parçası, serbest metin bir "flag" değil.
    ascites_present: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Asit (karın boşluğunda sıvı birikmesi, ileri karaciğer hastalığı bulgusu) var mı -- true/false, Child-Pugh kriteri",
    )
    hepatic_encephalopathy_grade: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Hepatik ensefalopati (karaciğer yetmezliğine bağlı bilinç bulanıklığı) derecesi -- 0-4 arası, Child-Pugh kriteri (0=yok)",
    )


class DrugAllergy(BaseModel):
    drug_or_class: str
    reaction_type: Optional[str] = None
    severity: Optional[str] = None
    source_quote: str
    source_document: str


class ChronicCondition(BaseModel):
    """
    Kronik hastalık -- basit string yerine kaynak alıntısı taşıyan bir
    obje: hiçbir liste elemanı kaynak alıntısı olmadan var olmamalı.
    """

    name: str
    source_quote: str
    source_document: str


class CurrentMedication(BaseModel):
    drug_name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    source_quote: str
    source_document: str


class InteractingSupplement(BaseModel):
    """
    Reçetesiz takviye/bitkisel ürün -- known_cyp_pgp_effect, LLM'in
    ÜRETTİĞİ bir yargı DEĞİL; sadece rapor metninde bu etkiye dair bir not
    varsa (örn. "hastaya greyfurt suyu ile ilaç etkileşimi anlatıldı")
    o notun aynen aktarılmasıdır -- LLM burada CYP/P-gp etkisini icat etmez.
    """

    name: str
    known_cyp_pgp_effect: Optional[str] = None
    source_quote: str
    source_document: str


class PatientFlags(BaseModel):
    """Kategori B -- flag/kural motoruna giren alanlar."""

    drug_allergies: list[DrugAllergy] = Field(default_factory=list)
    chronic_conditions: list[ChronicCondition] = Field(default_factory=list)
    pregnancy_status: ExtractedField = Field(
        default_factory=ExtractedField,
        description='"not_pregnant" | "pregnant_trimester_1" | "pregnant_trimester_2" | "pregnant_trimester_3" | "unknown"',
    )
    lactation_status: ExtractedField = Field(default_factory=ExtractedField)
    current_medications: list[CurrentMedication] = Field(default_factory=list)
    otc_supplements: list[InteractingSupplement] = Field(default_factory=list)


class PatientProfile(BaseModel):
    core_parameters: PatientCoreParameters = Field(default_factory=PatientCoreParameters)
    flags: PatientFlags = Field(default_factory=PatientFlags)
    extraction_metadata: dict = Field(default_factory=dict)
