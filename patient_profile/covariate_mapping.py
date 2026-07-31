"""
Kovaryat eşleme -- LLM'DEN TAMAMEN BAĞIMSIZ, saf deterministik Python
fonksiyonları. Bu dosyada hiçbir LLM çağrısı YOK; sadece sayısal
formüller (Cockcroft-Gault, Child-Pugh, allometrik ölçekleme) ve
bunların çıktısını mevcut `src/worldmodel/patient.py > Patient`
parametrelerine (renal_function/hepatic_function, 0-1 skala) eşleyen
bir entegrasyon fonksiyonu.

MİMARİ NOT (ADIM 0'da tespit edilen uyumsuzluk): mevcut sistem
`renal_function`/`hepatic_function`'ı SOYUT 0-1 skorlar olarak tutuyor
(ham kreatinin/bilirubin DEĞİL). Bu modül, CrCl/Child-Pugh gibi klinik
skorları bu soyut skalaya köprüleyen dönüşüm fonksiyonları sağlıyor --
bunlar KESİN klinik formüller değil, mevcut mimariye uyum sağlamak için
yapılan belgelenmiş yaklaşıklıklardır (bkz. her fonksiyonun docstring'i).
"""

from dataclasses import dataclass, field
from typing import Literal

from patient_profile.schema import PatientProfile

# ---------------------------------------------------------------------------
# Cockcroft-Gault (kreatinin klerensi -- CrCl, böbreğin kanı dakikada ne
# kadar süzdüğünün bir tahmini, mL/dk)
# ---------------------------------------------------------------------------


def cockcroft_gault(age: int, weight_kg: float, serum_creatinine: float, sex: str) -> float:
    """
    Kreatinin klerensini (mL/dk) hesaplar.
    Formül: CrCl = ((140 - age) * weight_kg) / (72 * serum_creatinine)
            kadın hastada sonuç * 0.85
    Birim: serum_creatinine mg/dL cinsinden olmalı.

    NOT: Formül sadece "male"/"female" ikili ayrımı için validasyonu
    olan klasik bir klinik formüldür -- sex bu ikisinden farklıysa
    (örn. "unknown"), 0.85 düzeltmesi UYGULANMAZ (erkek-eşdeğeri
    varsayılan), bu bilinçli bir yaklaşıklıktır ve CALIBRATION_REPORT.md
    tarzında belgelenmelidir.
    """
    if age is None or weight_kg is None or serum_creatinine is None or sex is None:
        raise ValueError("Cockcroft-Gault için gerekli tüm parametreler dolu olmalı")
    if serum_creatinine <= 0:
        raise ValueError("serum_creatinine sıfır veya negatif olamaz")
    crcl = ((140 - age) * weight_kg) / (72 * serum_creatinine)
    if sex == "female":
        crcl *= 0.85
    return crcl


def crcl_to_renal_function(crcl_ml_min: float, reference_crcl: float = 100.0) -> float:
    """
    CrCl'yi mevcut sistemin 0-1 renal_function skalasına çevirir.
    Referans: CrCl 100 mL/dk = normal fonksiyon (renal_function=1.0),
    bu değer birçok renal doz ayarlama kılavuzunda (KDIGO, FDA renal
    impairment guidance) "normal" referans klerens olarak kullanılır.
    0'a clamp edilir (negatif/anlamsız değer üretmesin); üst sınırda
    (renal_function>1, süperfonksiyon) CAP YOK -- projenin geri kalanında
    renal_function>1 kullanan bir yer bulunamadı (organ_function_adjusted_ke
    formülü matematiksel olarak >1 girdiyle de tutarlı çalışıyor: ke
    normalden daha hızlı olur), bu yüzden cap'lemeye gerek görülmedi.

    YAKLAŞIKLIK UYARISI: bu ORANSAL/LİNEER bir köprüdür, gerçek
    klerens-fonksiyon ilişkisi kesin lineer olmayabilir (böbrek rezervi
    nedeniyle). İDEAL ALTERNATİF (uygulanmadı, gelecek iş): mevcut
    `renal_function` katmanını tamamen bypass edip CrCl'yi doğrudan
    `Drug.renal_clearance_fraction` ile çarparak `ke`'yi ayarlamak --
    bu, iki ayrı "oransal küçültme" adımını (CrCl->renal_function->ke)
    tek adıma indirger ve muhtemelen daha doğru olur. Şimdilik basit
    köprüyle ilerleniyor (bkz. proje sohbeti, ADIM 0 sonrası karar).
    """
    return max(0.0, crcl_ml_min / reference_crcl)


# ---------------------------------------------------------------------------
# Child-Pugh (karaciğer fonksiyon skoru -- 5 klinik kritere dayanan,
# 5-15 puan arası ordinal bir skor, A/B/C sınıfına ayrılır)
# ---------------------------------------------------------------------------


def _child_pugh_points_bilirubin(bilirubin_mg_dl: float) -> int:
    if bilirubin_mg_dl < 2:
        return 1
    if bilirubin_mg_dl <= 3:
        return 2
    return 3


def _child_pugh_points_albumin(albumin_g_dl: float) -> int:
    if albumin_g_dl > 3.5:
        return 1
    if albumin_g_dl >= 2.8:
        return 2
    return 3


def _child_pugh_points_inr(inr: float) -> int:
    if inr < 1.7:
        return 1
    if inr <= 2.3:
        return 2
    return 3


def _child_pugh_points_ascites(ascites_present: bool) -> int:
    """
    Standart Child-Pugh, asiti "yok / hafif-kontrollü / orta-ağır
    (refrakter)" olarak 3 dereceye ayırır (1/2/3 puan). Şemamızda
    (ADIM 2) sadece BOOLEAN `ascites_present` var -- bu yüzden 3 puanlık
    (ağır/refrakter) uca hiç ulaşamıyoruz, sadece 1 (yok) veya 2
    (var, şiddeti belirsiz) verebiliyoruz. Bu, dürüstçe kabul edilmesi
    gereken bir kapsam sınırı -- gelecekte şemaya bir "severity"
    alanı eklenirse düzeltilebilir.
    """
    return 2 if ascites_present else 1


def _child_pugh_points_encephalopathy(grade: int) -> int:
    if grade == 0:
        return 1
    if grade <= 2:
        return 2
    return 3


def child_pugh_score(
    bilirubin_total: float,
    albumin: float,
    inr: float,
    ascites_present: bool,
    hepatic_encephalopathy_grade: int,
) -> int:
    """
    Standart Child-Pugh puanlaması (5 kriter, her biri 1-3 puan, toplam
    5-15). Beş girdiden HERHANGİ BİRİ None ise hesaplanamaz -- eksik
    veriyle sessizce yarım bir skor üretmez.
    """
    if None in (bilirubin_total, albumin, inr, ascites_present, hepatic_encephalopathy_grade):
        raise ValueError("Child-Pugh skoru için 5 kriterin de dolu olması gerekir")
    return (
        _child_pugh_points_bilirubin(bilirubin_total)
        + _child_pugh_points_albumin(albumin)
        + _child_pugh_points_inr(inr)
        + _child_pugh_points_ascites(ascites_present)
        + _child_pugh_points_encephalopathy(hepatic_encephalopathy_grade)
    )


def child_pugh_class(score: int) -> Literal["A", "B", "C"]:
    """5-6 = A (hafif), 7-9 = B (orta), 10-15 = C (ağır)."""
    if score <= 6:
        return "A"
    if score <= 9:
        return "B"
    return "C"


def child_pugh_to_hepatic_function(score: int) -> float:
    """
    UYARI: Bu, literatürde validasyonu olan bir formül DEĞİL -- Child-Pugh
    sürekli bir klerens ölçümü değil, ordinal bir skor. Bu eşleme,
    mevcut sistemin 0-1 skalasına uyum sağlamak için yapılan
    muhafazakar bir yaklaşıklıktır, klinik kesinlik iddia etmez.
    Sınıf A (5-6 puan): hafif/yok bozukluk → hepatic_function 0.9-1.0
    Sınıf B (7-9 puan): orta bozukluk        → hepatic_function 0.5-0.7
    Sınıf C (10-15 puan): ağır bozukluk       → hepatic_function 0.2-0.4
    Her sınıf içinde puan arttıkça skala içinde lineer enterpolasyon yapılır.
    """
    if score <= 6:
        return 1.0 - (score - 5) * 0.05  # 5->1.0, 6->0.95
    elif score <= 9:
        return 0.7 - (score - 7) * 0.10  # 7->0.7, 9->0.5
    else:
        return max(0.2, 0.4 - (score - 10) * 0.04)  # 10->0.4, 15->0.2


# ---------------------------------------------------------------------------
# Allometrik PK ölçekleme
# ---------------------------------------------------------------------------


def allometric_scale(
    reference_clearance: float, reference_weight_kg: float, patient_weight_kg: float, exponent: float = 0.75
) -> float:
    """
    Standart allometrik ölçekleme: Cl_hasta = Cl_referans * (W_hasta / W_referans) ** exponent
    exponent klerens için genelde 0.75, dağılım hacmi için genelde 1.0 kullanılır --
    hangi parametreyi ölçeklediğine göre exponent'i doğru seç, fonksiyonu
    parametreye özel çağır.

    ENTEGRASYON NOTU: `apply_patient_covariates()` bu fonksiyonu OTOMATİK
    ÇAĞIRMAZ -- çünkü mevcut Drug modeli (bkz. src/worldmodel/pk.py)
    zaten `vd_per_kg`/`ke_mean` üzerinden kiloyla ölçekleniyor
    (plasma_concentration() içinde `vd = vd_per_kg * weight_kg`).
    Bu fonksiyonu OTOMATİK uygulamak, aynı kilo etkisini İKİ KEZ (bir
    kere burada, bir kere pk.py'de) uygulama riski taşır. Bu yüzden
    ayrı bir yardımcı fonksiyon olarak bırakıldı -- ileride bir ilacın
    parametreleri "referans hastaya göre" (per-kg değil, tek bir
    referans ağırlık için) verilirse (örn. pediatrik ekstrapolasyon)
    kullanılabilir, mevcut ilaç setinde (hepsi zaten per-kg) gerekmiyor.
    """
    return reference_clearance * (patient_weight_kg / reference_weight_kg) ** exponent


# ---------------------------------------------------------------------------
# Entegrasyon: PatientProfile -> mevcut Patient parametreleri
# ---------------------------------------------------------------------------


@dataclass
class AdjustedPatientParams:
    """
    src/worldmodel/patient.py > Patient'ın kabul ettiği alanlara
    doğrudan eşlenebilecek, hastaya-özel (ya da hiçbiri bulunamadıysa
    popülasyon-ortalaması varsayılan) parametreler.
    """

    weight_kg: float
    renal_function: float
    hepatic_function: float
    baseline_hr: float
    baseline_sbp: float
    baseline_dbp: float
    potassium_mEqL: float
    calcium_mgdL: float
    adjustment_log: list[str] = field(default_factory=list)


def apply_patient_covariates(profile: PatientProfile, base_pk_params: dict) -> AdjustedPatientParams:
    """
    `base_pk_params` -- popülasyon-ortalaması varsayılanlar (örn.
    `configs/patients.yaml > hasta_a` alanları): weight_kg, renal_function,
    hepatic_function, baseline_hr, baseline_sbp, baseline_dbp,
    potassium_mEqL, calcium_mgdL.

    Her alan için: hasta profilinde YETERLİ veri varsa hastaya-özel
    değer hesaplanır/kullanılır; YOKSA varsayılan (`base_pk_params`)
    değişmeden kullanılır VE bu açıkça `adjustment_log`'a yazılır --
    hiçbir ayarlama sessizce atlanmaz.
    """
    core = profile.core_parameters
    log: list[str] = []

    # --- Doğrudan 1:1 alan eşlemeleri (formül gerektirmez) ---
    def _direct_or_default(extracted_value, default_value, label: str, unit: str = ""):
        if extracted_value is not None:
            log.append(f"{label}: hasta verisinden alındı ({extracted_value}{unit}).")
            return extracted_value
        log.append(f"{label}: hasta verisi yok, popülasyon-ortalaması varsayılan kullanıldı ({default_value}{unit}).")
        return default_value

    weight_kg = _direct_or_default(core.weight_kg.value, base_pk_params["weight_kg"], "weight_kg", " kg")
    baseline_hr = _direct_or_default(
        core.baseline_heart_rate.value, base_pk_params["baseline_hr"], "baseline_hr", " bpm"
    )
    baseline_sbp = _direct_or_default(
        core.baseline_bp_systolic.value, base_pk_params["baseline_sbp"], "baseline_sbp", " mmHg"
    )
    baseline_dbp = _direct_or_default(
        core.baseline_bp_diastolic.value, base_pk_params["baseline_dbp"], "baseline_dbp", " mmHg"
    )
    potassium = _direct_or_default(core.potassium.value, base_pk_params["potassium_mEqL"], "potassium_mEqL", " mEq/L")
    calcium = _direct_or_default(core.calcium.value, base_pk_params["calcium_mgdL"], "calcium_mgdL", " mg/dL")

    # --- Renal: CrCl (Cockcroft-Gault) -> renal_function ---
    if None not in (core.age.value, weight_kg, core.serum_creatinine.value, core.sex.value):
        crcl = cockcroft_gault(core.age.value, weight_kg, core.serum_creatinine.value, core.sex.value)
        renal_function = crcl_to_renal_function(crcl)
        log.append(
            f"renal_function: Cockcroft-Gault CrCl={crcl:.1f} mL/dk'dan türetildi "
            f"(renal_function={renal_function:.2f}, referans CrCl=100 mL/dk)."
        )
    else:
        renal_function = base_pk_params["renal_function"]
        log.append(
            "renal_function: gerekli veri (yaş/kilo/kreatinin/cinsiyet) eksik, "
            f"renal ayarlama uygulanmadı, varsayılan klerens kullanılıyor (renal_function={renal_function})."
        )

    # --- Hepatik: Child-Pugh -> hepatic_function ---
    if None not in (
        core.bilirubin_total.value,
        core.albumin.value,
        core.inr.value,
        core.ascites_present.value,
        core.hepatic_encephalopathy_grade.value,
    ):
        score = child_pugh_score(
            core.bilirubin_total.value,
            core.albumin.value,
            core.inr.value,
            bool(core.ascites_present.value),
            int(core.hepatic_encephalopathy_grade.value),
        )
        hepatic_function = child_pugh_to_hepatic_function(score)
        log.append(
            f"hepatic_function: Child-Pugh skoru={score} (sınıf {child_pugh_class(score)}) "
            f"üzerinden türetildi (hepatic_function={hepatic_function:.2f})."
        )
    else:
        hepatic_function = base_pk_params["hepatic_function"]
        log.append(
            "hepatic_function: gerekli 5 Child-Pugh kriterinden en az biri eksik, "
            f"hepatik ayarlama uygulanmadı, varsayılan kullanılıyor (hepatic_function={hepatic_function})."
        )

    return AdjustedPatientParams(
        weight_kg=weight_kg,
        renal_function=renal_function,
        hepatic_function=hepatic_function,
        baseline_hr=baseline_hr,
        baseline_sbp=baseline_sbp,
        baseline_dbp=baseline_dbp,
        potassium_mEqL=potassium,
        calcium_mgdL=calcium,
        adjustment_log=log,
    )
