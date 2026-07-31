# GÜVENLİK NOTU: Bu modül hassas hasta verisi (yaş, elektrolitler,
# komorbidite vb.) diske YAZAR -- kaydedilen dosya asla git'e commit
# edilmemeli (bkz. .gitignore > patient_records/). Bu bir demo/POC
# persistansı -- prodüksiyonda şifreleme/erişim kontrolü ayrıca
# gerekir (aynı uyarı patient_profile'ın diğer dosyalarında da var).
"""
Streamlit "Hasta Kaydı" sekmesinde oluşturulan/onaylanan hastaların diske
(JSON) kalıcı kaydı -- Streamlit `session_state` sadece TARAYICI OTURUMU
boyunca yaşar (uygulama yeniden başlatılınca ya da sekme kapatılınca
kaybolur); bu modül, "bir hastayı bir isimle kaydet, sonra tekrar seç"
akışının uygulama yeniden başlatıldıktan SONRA da çalışmasını sağlar.

Kaydedilen alanlar, src/worldmodel/patient.py > Patient'ın ilgili
alanlarıyla BİREBİR AYNI isimlerde -- dönüştürme katmanı gerekmiyor,
streamlit_app.py bu sözlüğü doğrudan Patient(**...) için kullanabilir
(name/blood_type/baseline_dbp/baseline_spo2 hariç -- onlar sabit/ayrı
girilen alanlar, bkz. streamlit_app.py > Patient(...) çağrısı).
"""

import json
import os
from datetime import datetime, timezone

DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patient_records", "saved_patients.json"
)

# src/worldmodel/patient.py > Patient'ın hasta-özel (comorbidity/known_av_
# block_degree dahil) alanlarıyla birebir aynı isimler.
PATIENT_RECORD_FIELDS = (
    "age", "weight_kg", "height_cm", "baseline_hr", "baseline_sbp",
    "renal_function", "hepatic_function", "potassium_mEqL", "calcium_mgdL",
    "comorbidity", "known_av_block_degree",
)


def load_saved_patients(path: str = DEFAULT_REGISTRY_PATH) -> dict[str, dict]:
    """
    Dosya hiç yoksa (ilk çalıştırma, henüz hiç hasta kaydedilmedi) boş
    sözlük döner -- hata fırlatmaz, bu geçerli/beklenen bir başlangıç
    durumu.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_patient_record(name: str, fields: dict, path: str = DEFAULT_REGISTRY_PATH,
                         source: str = "manual", extra_metadata: dict | None = None) -> dict[str, dict]:
    """
    `fields` -- PATIENT_RECORD_FIELDS'in bir alt/tam kümesini içeren bir
    sözlük (eksik/None alanlar öylece kaydedilir, çağıran taraf zaten
    Patient dataclass varsayılanlarına düşer). `source`: "manual" (sadece
    slider'larla girildi) | "pdf_extraction" (bir PDF onayından geldi).

    Aynı isimle tekrar kaydedilirse ÜZERİNE YAZILIR (upsert) -- ayrı bir
    "zaten var, emin misiniz" kontrolü YOK, kullanıcı arayüzü bunu
    bilinçli bir güncelleme olarak sunmalı.

    Dönüş: dosyaya yazılmış hâliyle TÜM kayıt sözlüğü.
    """
    if not name or not name.strip():
        raise ValueError("Hasta adı boş olamaz.")

    records = load_saved_patients(path)
    record = {field: fields.get(field) for field in PATIENT_RECORD_FIELDS}
    record["source"] = source
    record["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    if extra_metadata:
        record["extra_metadata"] = extra_metadata

    records[name.strip()] = record
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def delete_patient_record(name: str, path: str = DEFAULT_REGISTRY_PATH) -> dict[str, dict]:
    """İsim kayıtlarda yoksa sessizce hiçbir şey yapmaz (hata fırlatmaz)."""
    records = load_saved_patients(path)
    records.pop(name, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records
