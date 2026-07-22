"""
Hasta ve İlaç veri modelleri.

Bunlar birer "dataclass" — yani sadece veri taşıyan, mantık içermeyen
temiz kutular. Mantık (denklemler) pk.py ve pd.py içinde yaşıyor.
Bu ayrım önemli: veri ile davranış birbirine karışmasın diye.
"""

from dataclasses import dataclass
import numpy as np
import yaml


@dataclass
class Patient:
    name: str
    weight_kg: float
    height_cm: float
    age: int
    blood_type: str
    baseline_hr: float       # bazal nabız (bpm) -- kalbin dinlenme halinde dakikada kaç kez attığı
    baseline_sbp: float      # bazal sistolik tansiyon (mmHg) -- kalp kasılırken (sistol) ölçülen YÜKSEK tansiyon değeri
    baseline_dbp: float      # bazal diyastolik tansiyon (mmHg) -- kalp gevşerken (diyastol) ölçülen DÜŞÜK tansiyon değeri
    baseline_spo2: float     # bazal oksijen satürasyonu (%) -- kandaki hemoglobinin ne kadarının oksijen taşıdığı
    # Böbrek/karaciğer fonksiyonu (0-1, normal=1.0, 0=tam yetmezlik).
    # Geriye dönük uyumluluk için opsiyonel -- eski hasta profilleri bu
    # alanları içermez, varsayılan olarak normal (1.0) fonksiyon kabul edilir.
    # Bir ilacın bu parametrelerden ne kadar etkileneceği, İLACIN KENDİSİNDE
    # tanımlı (Drug.renal_clearance_fraction/hepatic_clearance_fraction) --
    # her ilaç böbrek/karaciğerden eşit ölçüde atılmaz (bkz. drugs.yaml >
    # digoxin vs esmolol).
    renal_function: float = 1.0
    hepatic_function: float = 1.0
    # Elektrolit/lab değerleri -- normal aralık: potasyum 3.5-5.0 mEq/L,
    # kalsiyum 8.5-10.5 mg/dL. Varsayılanlar normal aralığın ORTA noktası
    # (geriye dönük uyumluluk: eski hasta profillerinde bu alanlar yok,
    # varsayılan normal kabul edilir -- pd.py'deki çarpan fonksiyonları
    # tam orta noktada çarpan=1.0 verecek şekilde tasarlandı, yani mevcut
    # davranış BOZULMAZ).
    potassium_mEqL: float = 4.25
    calcium_mgdL: float = 9.5
    # Kronik komorbidite -- ilaçtan/elektrolitten bağımsız, hastanın TEMEL
    # kalp/damar durumu. None = sağlıklı. "heart_failure" (sistolik kalp
    # yetmezliği) / "hypertension" (kronik hipertansiyon) -- bkz.
    # integrate_drug_with_circadapt.py > apply_comorbidity_to_circadapt.
    comorbidity: str | None = None

    @property
    def bsa(self) -> float:
        """Vücut yüzey alanı (Mosteller formülü) — klinik dozlamada gerçekten kullanılır."""
        return np.sqrt((self.height_cm * self.weight_kg) / 3600)

    @property
    def has_abnormal_electrolytes(self) -> bool:
        """recommend_dose()'un otomatik uyarı üretmesi için kullanılır."""
        return not (3.5 <= self.potassium_mEqL <= 5.0) or not (8.5 <= self.calcium_mgdL <= 10.5)


@dataclass
class Drug:
    display_name: str
    dose_mg: float
    ka: float
    ke_mean: float
    vd_per_kg: float
    emax_hr: float
    emax_sbp: float
    ec50: float
    # Kilo bazlı dozlama -- verilirse dose_mg yerine (dose_mg_per_kg * kilo)
    # kullanılır. Geriye dönük uyumluluk için opsiyonel: eski konfigürasyonlar
    # dose_mg_per_kg içermez, sabit dose_mg ile çalışmaya devam eder.
    dose_mg_per_kg: float | None = None
    # "beta_blocker" / "vasodilator" / "positive_inotrope" -- CircAdapt
    # entegrasyonunun ilacı hangi fizyolojik mekanizmaya bağlayacağını belirler.
    drug_class: str | None = None
    # İki-kompartmanlı IV bolus modeli için opsiyonel mikro hız sabitleri
    # (1/saat) ve santral dağılım hacmi (L/kg). Üçü de verilmişse
    # simulation.py'da pk_model="two_compartment" seçilebilir; verilmezse
    # ka/ke_mean/vd_per_kg ile tek-kompartmanlı model kullanılmaya devam eder.
    k10: float | None = None
    k12: float | None = None
    k21: float | None = None
    vd_central_per_kg: float | None = None
    # Etki bölgesi (effect-compartment) denge hız sabitleri (1/saat) --
    # nabız ve tansiyon etkisinin plazma konsantrasyonuna FARKLI hızlarda
    # "yetiştiğini" modellemek için. None ise gecikmesiz (eski) davranışa
    # düşülür: etki doğrudan plazma konsantrasyonundan hesaplanır.
    keo_hr: float | None = None
    keo_sbp: float | None = None
    # Toplam eliminasyonun ne kadarının böbrek/karaciğer yoluyla olduğu
    # (0-1). Patient.renal_function/hepatic_function'ın ke üzerindeki
    # etkisini belirler -- bkz. pk.py > organ_function_adjusted_ke.
    # None/0.0: bu ilaç organ fonksiyonundan ETKİLENMEZ (örn. esmolol,
    # eritrosit esterazlarıyla metabolize olur -- böbrek/karaciğerden
    # bağımsız). Bunu None bırakmak yerine 0.0 yazmak, "etkilenmez"in
    # bilinçli bir modelleme kararı olduğunu (unutulmuş bir alan değil)
    # açıkça gösterir.
    renal_clearance_fraction: float = 0.0
    hepatic_clearance_fraction: float = 0.0


def load_patients(path: str) -> dict[str, Patient]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {key: Patient(**vals) for key, vals in raw.items()}


def load_drugs(path: str) -> dict[str, Drug]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {key: Drug(**vals) for key, vals in raw.items()}


# configs/drugs_verified.yaml, Drug dataclass alanlarının yanında
# izlenebilirlik (provenance) alanları da içerir -- bunlar Drug'ın parçası
# DEĞİL, bu yüzden Drug(**vals) çağrılmadan önce ayrıştırılmaları gerekir.
_PROVENANCE_FIELDS = {"rxcui", "source_url", "retrieved_date", "calibration_notes"}


def load_verified_drugs(path: str) -> dict[str, dict]:
    """
    configs/drugs_verified.yaml'ı okur. Her girdi için hem çalışan bir
    Drug nesnesi hem de o verinin kaynağını (RxNorm RxCUI, openFDA
    source_url, çekilme tarihi, kalibrasyon notu) ayrı ayrı döndürür:

        {"esmolol": {"drug": Drug(...), "provenance": {"rxcui": "49737", ...}}, ...}

    load_drugs()'tan farkı budur -- drugs.yaml'daki kaynak bilgisi sadece
    YAML yorumu olarak duruyor (insan okuyabilir, program okuyamaz);
    burada source_url/retrieved_date birer VERİ alanı, programatik olarak
    erişilebilir (audit trail için gerekli).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    result = {}
    for key, vals in raw.items():
        provenance = {field: vals[field] for field in _PROVENANCE_FIELDS if field in vals}
        drug_fields = {k: v for k, v in vals.items() if k not in _PROVENANCE_FIELDS}
        result[key] = {"drug": Drug(**drug_fields), "provenance": provenance}
    return result
