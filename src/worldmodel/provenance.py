"""
Veri Kaynağı İzlenebilirliği (Audit Trail)

"Verinin kaynağı belli olmalı, kara kutu olmamalı." Bu modül, bir
simülasyonda kullanılan HER parametrenin nereden geldiğini üç kategoriye
ayırıyor:

  - "literatür"        -> yayınlanmış bir kaynaktan (FDA etiketi, hakemli
                           çalışma, ders kitabı) -- bkz. CALIBRATION_REPORT.md
  - "varsayım"          -> yönü/mekanizması gerçek fizyolojiye dayanan ama
                           kesin sayısı kalibrasyon gerektiren temsili değer
  - "kullanıcı girdisi" -> hastanın kendi verisi (kilo, bazal nabız, vb.) --
                           model parametresi değil, bu spesifik hasta için
                           girilen bir değer

Sınıflandırma, CALIBRATION_REPORT.md'deki tabloyla TUTARLI tutulmalı --
biri güncellenirse diğeri de güncellenmeli (ikisi de aynı gerçeği farklı
biçimlerde -- biri insan-okunur rapor, biri programatik sorgu -- sunuyor).
"""

from .patient import Patient, Drug

# --- İlaca özgü PK/PD parametrelerinin kaynağı (display_name ile eşleşir) ---
# NOT: display_name'e göre eşleşiyor çünkü Drug nesnesi hangi yaml
# girdisinden geldiğini ayrıca taşımıyor -- Streamlit'te dataclasses.replace
# ile sadece dose_mg/ec50 değiştirilse bile display_name aynı kaldığı için
# bu eşleşme kararlı kalıyor.

_LITERATURE = "literatür"
_ASSUMPTION = "varsayım"
_USER_INPUT = "kullanıcı girdisi"

DRUG_PARAMETER_PROVENANCE: dict[str, dict[str, tuple[str, str]]] = {
    "Esmolol (literatür kalibrasyonlu)": {
        "dose_mg_per_kg": (_LITERATURE, "FDA prescribing information -- 0.5 mg/kg bolus"),
        "ka": (_LITERATURE, "Wiest 1991 -- dağılım yarı ömrü ~2 dk"),
        "ke_mean": (_LITERATURE, "Wiest 1991, FDA label -- eliminasyon yarı ömrü ~9 dk"),
        "vd_per_kg": (_LITERATURE, "yayınlanmış yetişkin/pediatrik PK çalışmaları"),
        "emax_hr": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "emax_sbp": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "ec50": (_LITERATURE, "Reilly ve ark. (Eur J Clin Pharmacol) -- infüzyon-hızı EC50'si "
                              "(113 mcg/kg/dk) TÜRETİLEREK (Css=infüzyon/klerens) plazma "
                              "konsantrasyonuna çevrildi, doğrudan ölçülmüş değil -- bkz. "
                              "CALIBRATION_REPORT.md §1a"),
        "keo_hr": (_ASSUMPTION, "temsili -- literatürden değil"),
        "keo_sbp": (_ASSUMPTION, "temsili -- literatürden değil"),
        "renal_clearance_fraction": (_LITERATURE, "esmolol eritrosit esterazlarıyla metabolize olur, böbrekten bağımsız"),
        "hepatic_clearance_fraction": (_LITERATURE, "aynı -- karaciğerden bağımsız"),
    },
    "Nikardipin (literatür kalibrasyonlu)": {
        "dose_mg_per_kg": (_LITERATURE, "Drugs.com / DailyMed etiketi -- 30 mcg/kg IV bolus"),
        "ka": (_LITERATURE, "Clinical Pharmacokinetics 2006 -- alfa t1/2 2.7 dk"),
        "ke_mean": (_LITERATURE, "Clinical Pharmacokinetics 2006 -- beta t1/2 44.8 dk"),
        "vd_per_kg": (_LITERATURE, "Clinical Pharmacokinetics 2006 -- non-kompartman Vd"),
        "emax_hr": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "emax_sbp": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "ec50": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
    },
    "Dobutamin (literatür kalibrasyonlu, bolus-eşdeğeri yaklaşıklık)": {
        "dose_mg_per_kg": (_ASSUMPTION, "bolus-eşdeğeri yaklaşıklık -- dobutamin klinikte sadece infüzyon olarak verilir (artık kullanılmıyor, bkz. infusion_rate_mcg_per_kg_min)"),
        "ka": (_ASSUMPTION, "~1 dk denge süresi varsayımı -- literatür değeri değil (artık kullanılmıyor)"),
        "ke_mean": (_LITERATURE, "Kates & Leier 1978 -- t1/2 ~2.5 dk"),
        "vd_per_kg": (_LITERATURE, "Kates & Leier 1978 -- ~0.2 L/kg"),
        "emax_hr": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "emax_sbp": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "ec50": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "infusion_rate_mcg_per_kg_min": (_ASSUMPTION, "'klasik 5 mcg/kg/dk infüzyon' -- pediatrik YBÜ çalışması (isim atfı zayıf), gerçek bir doz-yanıt çalışmasından türetilmedi"),
    },
    "Sodyum Nitroprussid (literatür kalibrasyonlu, bolus-eşdeğeri yaklaşıklık)": {
        "dose_mg_per_kg": (_ASSUMPTION, "bolus-eşdeğeri yaklaşıklık -- nitroprussid klinikte sadece infüzyon olarak verilir (artık kullanılmıyor, bkz. infusion_rate_mcg_per_kg_min)"),
        "ka": (_ASSUMPTION, "~1.4 dk denge süresi varsayımı -- literatür değeri değil (artık kullanılmıyor)"),
        "ke_mean": (_LITERATURE, "FDA etiketi -- circulatory half-life ~2 dk"),
        "vd_per_kg": (_LITERATURE, "FDA etiketi -- ekstrasellüler sıvı hacmiyle örtüşüyor"),
        "emax_hr": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "emax_sbp": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir -- Gregoire ve ark. (aşağıya bkz.) ER50 verdi, Emax büyüklüğünü VERMEDİ"),
        "ec50": (_LITERATURE, "📚 LİTERATÜR AMA ZAYIF KATEGORİ (esmolol/Kessler'deki doğrudan-yetişkin kategorisiyle KARIŞTIRILMASIN): "
                               "Gregoire ve ark., PMC4516882 (\"A hemodynamic model to guide blood pressure control "
                               "during deliberate hypotension with sodium nitroprusside in children\") -- PEDİATRİK "
                               "popülasyon, YETİŞKİN hastaya ekstrapole edildi (bu projenin tüm hasta profilleri "
                               "yetişkin). ER50 iki alt-grup için ayrı verildi (yüksek 0.34, düşük 0.103 mcg/kg/dk) "
                               "-- ARİTMETİK ORTALAMASI (0.2215 mcg/kg/dk) tek temsili nokta olarak kullanıldı "
                               "(VARSAYIM, bilgi kaybı). Esmolol Faz 3 (§1a) ile AYNI Css=R/Cl dönüşümüyle "
                               "0.0032 mg/L'ye çevrildi. Ek kısıt: kaynak MAP ölçtü, biz sistolik basıncı "
                               "(emax_sbp) temsil ediyoruz -- birebir aynı ölçüt değil. Bkz. CALIBRATION_REPORT.md Gap #1."),
        "infusion_rate_mcg_per_kg_min": (_ASSUMPTION, "FDA onaylı doz aralığının (0.3-10 mcg/kg/dk) ortancası -- gerçek bir doz-yanıt çalışmasından (Kessler 1987 veya esmolol Faz 3 gibi) türetilmedi, düzenleyici aralıktan alınan tek-nokta temsili değer. nicardipine/dobutamine EC50'leriyle AYNI güven kategorisinde: ⚠️ temsili"),
    },
    "Digoksin (literatür kalibrasyonlu -- böbrek fonksiyonuna duyarlı)": {
        "dose_mg_per_kg": (_LITERATURE, "standart IV yükleme dozu aralığı (~10-15 mcg/kg)"),
        "ka": (_ASSUMPTION, "~30 dk denge süresi varsayımı -- literatür değeri değil"),
        "ke_mean": (_LITERATURE, "standart ders kitabı -- t1/2 ~36 saat"),
        "vd_per_kg": (_LITERATURE, "standart ders kitabı -- ~7.3 L/kg"),
        "emax_hr": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "emax_sbp": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "ec50": (_ASSUMPTION, "temsili -- kalibrasyon gerektirir"),
        "renal_clearance_fraction": (_LITERATURE, "standart ders kitabı -- ~%65 değişmeden böbrekten atılım"),
    },
    "Örnek Vazodilatör": {
        "dose_mg": (_ASSUMPTION, "tamamen temsili/uydurma -- henüz literatür kalibrasyonu yapılmadı"),
        "ka": (_ASSUMPTION, "tamamen temsili"),
        "ke_mean": (_ASSUMPTION, "tamamen temsili"),
        "vd_per_kg": (_ASSUMPTION, "tamamen temsili"),
        "emax_hr": (_ASSUMPTION, "tamamen temsili"),
        "emax_sbp": (_ASSUMPTION, "tamamen temsili"),
        "ec50": (_ASSUMPTION, "tamamen temsili"),
    },
}

# --- Hasta alanlarının kaynağı -- bunlar model parametresi değil, hasta
# verisi: hepsi "kullanıcı girdisi" (ya da o alan girilmemişse fizyolojik
# normal varsayılan) ---
PATIENT_FIELD_PROVENANCE: dict[str, str] = {
    "weight_kg": "hasta verisi",
    "height_cm": "hasta verisi",
    "baseline_hr": "hasta verisi",
    "baseline_sbp": "hasta verisi",
    "renal_function": "hasta verisi (girilmemişse normal=1.0 varsayılır)",
    "hepatic_function": "hasta verisi (girilmemişse normal=1.0 varsayılır)",
    "potassium_mEqL": "hasta verisi (girilmemişse normal orta nokta=4.25 varsayılır)",
    "calcium_mgdL": "hasta verisi (girilmemişse normal orta nokta=9.5 varsayılır)",
    "comorbidity": "hasta verisi (girilmemişse sağlıklı/None varsayılır)",
}


def provenance_report(patient: Patient, drug: Drug) -> list[dict]:
    """
    Bu hasta+ilaç kombinasyonuyla bir simülasyon çalıştırıldığında
    kullanılan HER parametrenin kaynağını listeler.

    Dönüş: [{"parameter", "value", "source_type", "detail"}, ...]
    source_type: "literatür" / "varsayım" / "hasta verisi"
    """
    report = []

    drug_params = DRUG_PARAMETER_PROVENANCE.get(drug.display_name)
    if drug_params is None:
        report.append({
            "parameter": "(tüm ilaç parametreleri)",
            "value": drug.display_name,
            "source_type": "sınıflandırılmamış",
            "detail": "Bu ilaç DRUG_PARAMETER_PROVENANCE tablosunda tanımlı değil -- "
                      "muhtemelen özel/deneysel bir Drug nesnesi.",
        })
    else:
        for param_name, (source_type, detail) in drug_params.items():
            value = getattr(drug, param_name, None)
            if value is not None:
                report.append({
                    "parameter": param_name, "value": value,
                    "source_type": source_type, "detail": detail,
                })

    for field_name, detail in PATIENT_FIELD_PROVENANCE.items():
        value = getattr(patient, field_name, None)
        report.append({
            "parameter": field_name, "value": value,
            "source_type": _USER_INPUT, "detail": detail,
        })

    return report


SOURCE_TYPE_LABEL = {
    _LITERATURE: "Literatür",
    _ASSUMPTION: "Varsayım",
    _USER_INPUT: "Hasta Verisi",
    "sınıflandırılmamış": "Sınıflandırılmamış",
}
