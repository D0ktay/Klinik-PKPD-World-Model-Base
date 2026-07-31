"""
review_data.ReviewScreen/ReviewItem'ı Streamlit arayüzüne bağlayan, ama
Streamlit'e BAĞIMLI OLMAYAN saf yardımcı fonksiyonlar.

Bu dosya kasıtlı olarak Streamlit import ETMEZ -- widget render'ının
kendisi (streamlit_app.py > tab_registration) test edilmiyor, ama
"hangi alan hangi widget türüne gider", "aynı dosya seti tekrar
yüklenirse LLM'i tekrar çağırma", "onay sonrası hangi değerler
tab_patient'a akar" gibi SAF mantık burada, Streamlit çalıştırmadan
birim testiyle doğrulanabilir.
"""

import hashlib

from patient_profile.covariate_mapping import AdjustedPatientParams
from patient_profile.schema import PatientCoreParameters

# sex/known_av_block_degree -- şemada sabit bir seçenek kümesi olan tek
# iki alan (bkz. schema.py açıklamaları); geri kalan tüm PatientCoreParameters
# alanları sayısal (number_input), ascites_present ise boolean (checkbox).
SELECT_FIELD_OPTIONS: dict[str, list[str]] = {
    "sex": ["male", "female"],
    "known_av_block_degree": ["none", "first", "second", "third"],
}

CHECKBOX_FIELDS: frozenset[str] = frozenset({"ascites_present"})

# "(bilinmiyor)" -- select alanlarında value=None (rapor bu bilgiyi hiç
# içermiyor) ile kullanıcının GERÇEKTEN bir seçenek seçmiş olması
# arasındaki farkı kaybetmemek için eklenen açık bir üçüncü seçenek --
# aksi halde None, sessizce SELECT_FIELD_OPTIONS[0]'a (örn. "male")
# yuvarlanır, bu yanlış/uydurma bir veri girişi olurdu.
UNKNOWN_OPTION = "(bilinmiyor)"


def widget_kind_for_field(field_name: str) -> str:
    """
    ReviewItem.field_name'i, Streamlit tarafında hangi widget'ın
    kullanılacağına eşler: "number" | "select" | "checkbox".
    """
    if field_name in SELECT_FIELD_OPTIONS:
        return "select"
    if field_name in CHECKBOX_FIELDS:
        return "checkbox"
    return "number"


def select_options_with_unknown(field_name: str) -> list[str]:
    """SELECT_FIELD_OPTIONS[field_name]'in başına UNKNOWN_OPTION eklenmiş hâli."""
    return [UNKNOWN_OPTION] + SELECT_FIELD_OPTIONS[field_name]


def select_index_for_value(field_name: str, value) -> int:
    """value seçenekler arasındaysa onun index'i (UNKNOWN_OPTION dahil liste
    üzerinde), değilse (None ya da tanınmayan bir değer) 0 (UNKNOWN_OPTION)."""
    options = SELECT_FIELD_OPTIONS[field_name]
    if value in options:
        return 1 + options.index(value)
    return 0


def hash_uploaded_files(file_contents: list[bytes]) -> str:
    """
    Yüklenen dosyaların İÇERİĞİNE (dosya adına DEĞİL) dayalı, sıraya
    duyarlı bir hash -- aynı dosya seti (aynı sırada) tekrar
    yüklendiğinde LLM extraction'ın (ücretli/yavaş bir API çağrısı)
    TEKRAR ÇAĞRILMAMASI için cache anahtarı olarak kullanılır. Sıraya
    duyarlı olması bilinçli: aynı iki dosya farklı sırada yüklenirse
    (örn. temporal_merge.py'nin "son yüklenen kazanır" fallback'i
    nedeniyle) sonuç teorik olarak farklı olabilir, bu yüzden farklı
    bir cache anahtarı hak ediyor.
    """
    hasher = hashlib.sha256()
    for content in file_contents:
        hasher.update(content)
        hasher.update(b"\x00")
    return hasher.hexdigest()


def build_confirmed_patient_params(adjusted: AdjustedPatientParams, core: PatientCoreParameters) -> dict:
    """
    AdjustedPatientParams'ın 8 alanını (weight_kg/renal_function/
    hepatic_function/baseline_hr/baseline_sbp/baseline_dbp/
    potassium_mEqL/calcium_mgdL -- src/worldmodel/patient.py > Patient ile
    birebir eşleşiyor) + core_parameters'tan (covariate hesabından
    GEÇMEDEN, çünkü bunlar için bir formül YOK) doğrudan alınan
    age/height_cm/known_av_block_degree'yi TEK bir sözlükte birleştirir.

    Bu sözlük, streamlit_app.py > tab_patient'taki slider'ların
    value= parametrelerini besleyen TEK kaynaktır (bkz. ADIM 0 tasarım
    kararı #3/#5). age/height_cm None olabilir (rapor bu bilgiyi hiç
    içermiyorsa) -- çağıran taraf (tab_patient), None alanlar için
    kendi sabit varsayılanına düşmelidir, bu fonksiyon varsayılan
    UYDURMAZ.
    """
    return {
        "weight_kg": adjusted.weight_kg,
        "renal_function": adjusted.renal_function,
        "hepatic_function": adjusted.hepatic_function,
        "baseline_hr": adjusted.baseline_hr,
        "baseline_sbp": adjusted.baseline_sbp,
        "baseline_dbp": adjusted.baseline_dbp,
        "potassium_mEqL": adjusted.potassium_mEqL,
        "calcium_mgdL": adjusted.calcium_mgdL,
        "age": core.age.value,
        "height_cm": core.height_cm.value,
        "known_av_block_degree": core.known_av_block_degree.value,
        "adjustment_log": adjusted.adjustment_log,
    }


def resolve_conflicts_overridden_by_edits(conflicts: list[dict], edits: dict) -> list[dict]:
    """
    Kullanıcı, çelişkili bir alanı onay ekranında ELLE düzenlediyse (yeni
    bir değer girdiyse), bu artık kullanıcının BİLİNÇLİ kararıdır --
    çelişki listesinden çıkarılır (aksi halde ready_to_confirm hiçbir
    zaman True olamaz, kullanıcı çelişkiyi çözse bile). Düzenlenmemiş
    çelişkiler AYNEN kalır -- sadece kullanıcının dokunduğu alanlar
    "çözüldü" sayılır.
    """
    edited_fields = set(edits.keys())
    return [c for c in conflicts if c.get("field") not in edited_fields]
