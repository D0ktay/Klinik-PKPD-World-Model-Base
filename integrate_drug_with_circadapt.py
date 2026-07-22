"""
İlaç PK/PD Modelini CircAdapt ile Bütünleştirme
================================================

Bu script, src/worldmodel içindeki PK/PD motorunun ürettiği "ilaç
etkisini" (nabız ve tansiyon değişimi), CircAdapt'in gerçek kalp-damar
mekaniği simülasyonuna aktarır.

Akış:
  1. PK modeli -> zaman içinde plazma konsantrasyonu
  2. PD modeli -> konsantrasyondan etki oranına (Emax)
  3. Etki oranı -> nabız ve tansiyon değişimine
  4. Nabız değişimi -> CircAdapt'te her zaman General.t_cycle (kalp
     siklus süresi) üzerinden uygulanır. Tansiyon değişimi ise ilaç
     SINIFINA göre FARKLI bir mekanizmaya bağlanır (bkz.
     apply_drug_effect_to_circadapt):
       - beta_blocker / positive_inotrope -> Patch.Sf_act (ventrikül
         kontraktilitesi) -- yön (artış/azalış) sbp_fraction'ın
         işaretinden otomatik gelir.
       - vasodilator -> ArtVen.p0[0] (CiSy = sistemik dolaşım direnci),
         kontraktiliteye DOKUNULMAZ.
     Bunların hepsi CircAdapt modelinin mutlak nabız/tansiyonu hasta
     profiliyle birebir kalibre olmadığı için göreli (fraksiyonel)
     olarak uygulanıyor -- sadece İLAÇ'IN YARATTIĞI ORANSAL DEĞİŞİM
     aktarılıyor.
  5. İlaçsız (baseline) ve ilaçlı simülasyonların LV basınç/hacim
     eğrileri karşılaştırmalı çizilir.

ÖNEMLİ (CircAdapt API gotcha'sı -- bulmak için deneme gerekti):
  model[bileşen][parametre] bir `circadapt.components.Parameter`
  nesnesi döndürüyor. Bunu ALIP, İÇİNDEKİ elemanı değiştirip
  (`arr[idx] = değer`), SONRA AYNI nesneyi geri atarsan
  (`model[bileşen][parametre] = arr`), değişiklik SESSİZCE GERİ
  ALINIYOR -- model her zaman ilk okunan değere dönüyor. Doğru kullanım:
  SADECE öğe ataması yapıp geri atamayı YAPMAMAK
  (`arr = model[b][p]; arr[idx] = değer` yeterli, üçüncü satır YOK).
  Bu, projenin önceki fazlarındaki Sf_act değişikliklerinin FİİLEN HİÇ
  UYGULANMADIĞI anlamına geliyordu -- bu dosyada düzeltildi (bkz.
  README.md > "Bilinen CircAdapt API Davranışları").
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from worldmodel.pk import plasma_concentration
from worldmodel.pd import (
    emax_effect, apply_effect_to_vitals,
    potassium_av_conduction_factor, calcium_contractility_factor,
)

from circadapt import VanOsta2024

# Ventrikül duvarlarının Patch.Sf_act dizisindeki konumları
# (dizi sırası: pLa0, pRa0, pLv0, pSv0, pRv0 -- kurulum testinde doğrulandı)
VENTRICLE_WALL_INDICES = [2, 3, 4]  # pLv0, pSv0, pRv0

# ArtVen['objects'] = ['Model.CiSy', 'Model.CiPu'] -- yani p0/q0/k
# dizilerinde index 0 = sistemik dolaşım (CiSy), index 1 = pulmoner (CiPu).
# print(model['ArtVen']) ile doğrulandı.
SYSTEMIC_ARTVEN_INDEX = 0


def calibrate_circadapt_to_patient(model, patient):
    """
    DÜZELTME: CircAdapt'e hastanın KENDİ bazal nabzını (patient.baseline_hr,
    configs/patients.yaml'dan) aktarır.

    BULUNAN HATA: bu fonksiyon eklenmeden önce, CircAdapt'in bazal kalp
    hızı HER ZAMAN kendi jenerik varsayılanıydı (General.t_cycle=0.85s,
    yani ~70.6 bpm) -- hangi hasta seçilirse seçilsin (Hasta A: 78 bpm,
    Hasta B: 85 bpm) CircAdapt çıktısındaki "baseline" hep aynı (~71 bpm)
    çıkıyordu, çünkü apply_drug_effect_to_circadapt SADECE ilacın
    FRAKSİYONEL etkisini CircAdapt'in kendi jenerik t_cycle'ının üzerine
    uyguluyordu, hastanın gerçek bazalini hiç okumuyordu. Doğrulandı:
    hasta_a ve hasta_b farklı baseline_hr'a sahip olmalarına rağmen
    run_baseline() ikisinde de 70.6 bpm döndürüyordu.

    DÜZELTME: bu fonksiyon, İLAÇ ETKİSİ UYGULANMADAN ÖNCE çağrılarak
    General.t_cycle'ı `60/patient.baseline_hr` olacak şekilde ayarlıyor
    -- yani CircAdapt önce hastanın KENDİ bazaline kalibre ediliyor,
    ilacın fraksiyonel etkisi SONRA bu kişiselleştirilmiş bazalin
    üzerine bindiriliyor. Sayısal kararlılık, gerçekçi tüm nabız
    aralığında (Streamlit slider sınırı 50-110 bpm) elle doğrulandı --
    CircAdapt'in sabit-süreli (saniye cinsinden, cycle fraksiyonu değil)
    sistol/diyastol zamanlama parametreleri (tr, td, time_act) bu
    aralıkta sorun çıkarmıyor.

    Tüm CircAdapt giriş noktaları (run_baseline, run_with_drug,
    run_with_multiple_drugs) bu fonksiyonu çağırıyor -- yeni bir
    entegrasyon yazan biri de aynı hatayı tekrar etmesin diye.
    """
    model["General"]["t_cycle"] = 60.0 / patient.baseline_hr
    return model


def apply_drug_effect_to_circadapt(model, drug_class: str, hr_fraction: float,
                                    sbp_fraction: float):
    """
    hr_fraction: yeni_nabız / bazal_nabız (örn. 0.8 = %20 nabız düşüşü)
    sbp_fraction: yeni_SBP / bazal_SBP (ilaç sınıfına göre <1 ya da >1 olabilir --
        yön zaten drugs.yaml'daki emax_hr/emax_sbp işaretinden geliyor)

    Nabız her zaman General.t_cycle üzerinden uygulanır. Tansiyon etkisi
    ise ilaç sınıfına göre farklı, gerçek CircAdapt parametrelerine
    bağlanır (isimler tahmin edilmedi, print(model[bileşen]) ile
    keşfedildi):
      - "beta_blocker" / "positive_inotrope": Patch.Sf_act (ventrikül
        kontraktilitesi). sbp_fraction<1 -> kontraktilite azalır
        (beta-bloker), sbp_fraction>1 -> artar (pozitif inotrop) --
        AYNI formül, yön işaretten otomatik gelir.
      - "vasodilator": ArtVen.p0[0] (sistemik dolaşım direncinin
        referans basıncı). PressureFlowControl (PFC) bileşeni normalde
        ArtVen direncini kendi hedefine göre otomatik "fit" ettiği için
        (bkz. circadapt/components/global_functions.py docstring:
        "Pressure Flow Control by fitting ArtVen resistance and total
        volume"), bu simülasyon için PFC geçici olarak devre dışı
        bırakılır -- aksi halde manuel direnç değişikliği
        stabilizasyon sırasında sessizce geri alınır (doğrulandı).
    """
    t_cycle_base = model["General"]["t_cycle"]
    model["General"]["t_cycle"] = t_cycle_base / hr_fraction

    if drug_class in ("beta_blocker", "positive_inotrope"):
        sf_act = model["Patch"]["Sf_act"]
        sf_act[VENTRICLE_WALL_INDICES] = sf_act[VENTRICLE_WALL_INDICES] * sbp_fraction
    elif drug_class == "vasodilator":
        model["PFC"]["is_active"] = False
        p0 = model["ArtVen"]["p0"]
        p0[SYSTEMIC_ARTVEN_INDEX] = p0[SYSTEMIC_ARTVEN_INDEX] * sbp_fraction
    else:
        raise ValueError(f"Bilinmeyen drug_class: {drug_class!r}")

    return model


def apply_patient_electrolytes_to_circadapt(model, patient):
    """
    Hastanın KENDİ elektrolit durumunu (ilaçtan bağımsız) CircAdapt
    modeline uygular:
      - Potasyum -> Timings.c_tau_av1 (AV iletim gecikmesi katsayısı).
        `Timings.tau_av`'ı DOĞRUDAN değiştirmek işe yaramıyor -- tau_av
        her adımda `c_tau_av0 + c_tau_av1 * t_cycle` formülünden (bir
        "law") yeniden hesaplanıyor, bu yüzden manuel atama sessizce
        geri alınıyor (deneyerek bulundu, tıpkı Sf_act/ArtVen bug'ları
        gibi). Doğru kullanım katsayıyı (c_tau_av1) değiştirmek.
      - Kalsiyum -> Patch.Sf_act (ventrikül kontraktilitesi).

    Normal orta noktadaki (K=4.25, Ca=9.5) bir hastada çarpanlar tam 1.0
    olduğu için bu fonksiyon NO-OP'tur -- geriye dönük uyumluluğu bozmaz.
    """
    k_factor = potassium_av_conduction_factor(patient.potassium_mEqL)
    if k_factor != 1.0:
        c_tau_av1 = model["Timings"]["c_tau_av1"]
        c_tau_av1[0] = c_tau_av1[0] * k_factor

    ca_factor = calcium_contractility_factor(patient.calcium_mgdL)
    if ca_factor != 1.0:
        sf_act = model["Patch"]["Sf_act"]
        sf_act[VENTRICLE_WALL_INDICES] = sf_act[VENTRICLE_WALL_INDICES] * ca_factor

    return model


# Hastalık profili çarpanları -- TEMSİLİDİR, belirli bir hasta
# kohortundan/çalışmadan kalibre edilmedi. Yön ve mekanizma gerçek
# fizyolojiye dayanıyor (sistolik KY = azalmış kontraktilite; kronik HT =
# artmış sistemik direnç + yükselmiş basınç setpoint'i).
HEART_FAILURE_CONTRACTILITY_FACTOR = 0.6      # ventrikül Sf_act'i %40 azalır
HYPERTENSION_RESISTANCE_FACTOR = 1.3          # sistemik direnç %30 artar


def apply_comorbidity_to_circadapt(model, comorbidity: str | None):
    """
    Hastanın KRONİK komorbiditesini (ilaçtan/elektrolitten bağımsız,
    hastanın temel kalp/damar durumu) CircAdapt modeline uygular.

    - "heart_failure" (sistolik kalp yetmezliği): ventrikül duvarlarının
      Patch.Sf_act'ini (kontraktilite) düşürür -- sistolik KY'nin TANIMI
      zaten "azalmış kontraktilite/ejeksiyon fraksiyonu" olduğu için bu,
      hastalığın kendisini temsil eden en doğrudan mekanizma.
    - "hypertension" (kronik hipertansiyon): HEM sistemik damar direncini
      (ArtVen.p0[0]) HEM PressureFlowControl'ün kendi hedef basıncını
      (PFC.p0) BİRLİKTE artırır. SADECE ArtVen.p0 artırmak yetmez --
      PFC normalde kendi SABİT hedefine geri "fit" eder (deneyerek
      keşfedildi, doğrulandı: ArtVen.p0 tek başına artırılırsa PFC bunu
      nötralize ediyor). PFC.p0'ı da artırmak, "bu hastanın vücudunun
      YENİ (yüksek) basıncı normal kabul ettiği" KRONİK bir adaptasyonu
      temsil ediyor -- akut vazodilatör senaryosundan (PFC geçici olarak
      DEVRE DIŞI bırakılıyor) kasıtlı olarak FARKLI bir modelleme kararı,
      çünkü burada YENİ bir denge durumuna adapte olmuş bir kalp
      istiyoruz, PFC'siz akut bir perturbasyon değil. Uçtan uca
      doğrulandı: sadece ArtVen.p0 artışı SyArt basıncını değiştirmiyor
      (91.49->91.49 mmHg), ama ikisi birlikte kalıcı bir yükseliş
      üretiyor (91.49->118.94 mmHg).
    """
    if comorbidity == "heart_failure":
        sf_act = model["Patch"]["Sf_act"]
        sf_act[VENTRICLE_WALL_INDICES] = sf_act[VENTRICLE_WALL_INDICES] * HEART_FAILURE_CONTRACTILITY_FACTOR
    elif comorbidity == "hypertension":
        p0_artven = model["ArtVen"]["p0"]
        p0_artven[SYSTEMIC_ARTVEN_INDEX] = p0_artven[SYSTEMIC_ARTVEN_INDEX] * HYPERTENSION_RESISTANCE_FACTOR
        p0_pfc = model["PFC"]["p0"]
        p0_pfc[0] = p0_pfc[0] * HYPERTENSION_RESISTANCE_FACTOR
    elif comorbidity is not None:
        raise ValueError(f"Bilinmeyen comorbidity: {comorbidity!r}")

    return model


def compute_drug_effect(patient, drug, t_peak_hours: float = None):
    """PK/PD zincirini çalıştırır ve pik etkideki nabız/tansiyonu döndürür."""
    t = np.linspace(0, 8.0, 500)
    conc = plasma_concentration(
        t, drug.dose_mg, drug.ka, drug.ke_mean, patient.weight_kg, drug.vd_per_kg,
        dose_mg_per_kg=drug.dose_mg_per_kg,
    )

    if t_peak_hours is None:
        peak_idx = int(np.argmax(conc))
    else:
        peak_idx = int(np.argmin(np.abs(t - t_peak_hours)))

    effect_fraction = emax_effect(conc[peak_idx], drug.ec50, sensitivity=1.0)
    hr_drug, sbp_drug = apply_effect_to_vitals(
        patient.baseline_hr, patient.baseline_sbp, effect_fraction,
        drug.emax_hr, drug.emax_sbp,
    )
    return {
        "t_peak_hours": t[peak_idx],
        "conc_peak": conc[peak_idx],
        "effect_fraction": float(effect_fraction),
        "hr_drug": float(hr_drug),
        "sbp_drug": float(sbp_drug),
    }


def run_baseline(patient=None):
    """
    patient verilirse, "baseline" artık jenerik bir sağlıklı
    kalp değil, BU HASTANIN KENDİ elektrolit durumunu VE kronik
    komorbiditesini yansıtan bir kalp -- ör. hiperkalemik bir hastanın
    "ilaçsız" baseline'ı bile yavaşlamış AV iletim gösterir; kalp
    yetmezliği olan bir hastanın baseline'ı bile düşük kontraktilite
    gösterir. Sağlıklı, normal orta noktadaki (K=4.25, Ca=9.5,
    comorbidity=None) bir hastada bu NO-OP'tur, eski davranış korunur.
    """
    model = VanOsta2024()
    if patient is not None:
        calibrate_circadapt_to_patient(model, patient)
        apply_patient_electrolytes_to_circadapt(model, patient)
        apply_comorbidity_to_circadapt(model, patient.comorbidity)
    model.run(stable=True)
    return model


def run_with_drug(patient, drug, drug_effect):
    model = VanOsta2024()
    calibrate_circadapt_to_patient(model, patient)
    apply_patient_electrolytes_to_circadapt(model, patient)
    apply_comorbidity_to_circadapt(model, patient.comorbidity)

    hr_fraction = drug_effect["hr_drug"] / patient.baseline_hr
    sbp_fraction = drug_effect["sbp_drug"] / patient.baseline_sbp
    # Eski konfigürasyonlarda drug_class olmayabilir --
    # geriye dönük uyumluluk için varsayılan beta_blocker davranışına düş.
    drug_class = drug.drug_class or "beta_blocker"

    apply_drug_effect_to_circadapt(model, drug_class, hr_fraction, sbp_fraction)

    model.run(stable=True)
    return model


def lv_pressure_volume(model):
    p = model["Cavity"]["p"][:, "cLv"] / 133.322
    v = model["Cavity"]["V"][:, "cLv"] * 1e6
    t = model["Solver"]["t"] * 1e3
    return t, p, v


def build_comparison_figure(t_base, p_base, v_base, t_drug, p_drug, v_drug,
                             drug_display_name, hr_base, hr_drug):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t_base, p_base, color="steelblue", label=f"Baseline ({hr_base:.0f} bpm)")
    axes[0].plot(t_drug, p_drug, color="crimson", label=f"{drug_display_name} ({hr_drug:.0f} bpm)")
    axes[0].set_xlabel("Zaman (ms)")
    axes[0].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[0].set_title("LV Basıncı — İlaçsız vs İlaçlı")
    axes[0].legend()

    axes[1].plot(v_base, p_base, color="steelblue", label="Baseline")
    axes[1].plot(v_drug, p_drug, color="crimson", label=drug_display_name)
    axes[1].set_xlabel("Sol karıncık hacmi (mL)")
    axes[1].set_ylabel("Sol karıncık basıncı (mmHg)")
    axes[1].set_title("Basınç-Hacim İlişkisi (PV Loop)")
    axes[1].legend()

    plt.tight_layout()
    return fig


def run_with_multiple_drugs(patient, drugs, drug_effects):
    """
    Birden fazla ilacı AYNI CircAdapt model örneğine, SIRAYLA uygular --
    her ilaç kendi drug_class'ının mekanizmasını hedefler.

    Bu, apply_drug_effect_to_circadapt'i birden fazla kez çağırmaktan
    ibaret ve KASITLI olarak öyle: General.t_cycle her çağrıda kendi
    ÖNCEKİ değerinden bölünür, yani iki ilacın nabız etkisi çarpımsal
    olarak birikir (iki negatif kronotropun etkisi TOPLAMDAN BÜYÜK
    olur -- gerçekçi bir "tehlikeli kombinasyon" davranışı). Aynı şekilde
    iki beta-bloker/pozitif-inotrop art arda Patch.Sf_act'i kendi
    sbp_fraction'ıyla çarpar (birikir). Farklı sınıftaki ilaçlar
    (örn. bir beta-bloker + bir vazodilatör) ise AYNI ANDA İKİ FARKLI
    parametreyi (Sf_act VE ArtVen.p0) etkiler -- her biri sadece kendi
    mekanizmasına dokunur, birbirini geçersiz kılmaz.
    """
    model = VanOsta2024()
    calibrate_circadapt_to_patient(model, patient)
    apply_patient_electrolytes_to_circadapt(model, patient)
    apply_comorbidity_to_circadapt(model, patient.comorbidity)

    for drug, drug_effect in zip(drugs, drug_effects):
        hr_fraction = drug_effect["hr_drug"] / patient.baseline_hr
        sbp_fraction = drug_effect["sbp_drug"] / patient.baseline_sbp
        drug_class = drug.drug_class or "beta_blocker"
        apply_drug_effect_to_circadapt(model, drug_class, hr_fraction, sbp_fraction)

    model.run(stable=True)
    return model


def run_polypharmacy_comparison(patient, drugs):
    """PK/PD + CircAdapt zincirini BİRDEN FAZLA ilaç için uçtan uca çalıştırır."""
    drug_effects = [compute_drug_effect(patient, drug) for drug in drugs]

    baseline_model = run_baseline(patient)
    combo_model = run_with_multiple_drugs(patient, drugs, drug_effects)

    t_base, p_base, v_base = lv_pressure_volume(baseline_model)
    t_combo, p_combo, v_combo = lv_pressure_volume(combo_model)

    hr_base = 60.0 / baseline_model["General"]["t_cycle"]
    hr_combo = 60.0 / combo_model["General"]["t_cycle"]

    return {
        "drug_effects": drug_effects,
        "t_base": t_base, "p_base": p_base, "v_base": v_base,
        "t_drug": t_combo, "p_drug": p_combo, "v_drug": v_combo,
        "hr_base": hr_base, "hr_drug_model": hr_combo,
    }


def run_comparison(patient, drug):
    """PK/PD + CircAdapt zincirini uçtan uca çalıştırır, ham sonuçları döndürür."""
    drug_effect = compute_drug_effect(patient, drug)

    baseline_model = run_baseline(patient)
    drug_model = run_with_drug(patient, drug, drug_effect)

    t_base, p_base, v_base = lv_pressure_volume(baseline_model)
    t_drug, p_drug, v_drug = lv_pressure_volume(drug_model)

    hr_base = 60.0 / baseline_model["General"]["t_cycle"]
    hr_drug_model = 60.0 / drug_model["General"]["t_cycle"]

    return {
        "drug_effect": drug_effect,
        "t_base": t_base, "p_base": p_base, "v_base": v_base,
        "t_drug": t_drug, "p_drug": p_drug, "v_drug": v_drug,
        "hr_base": hr_base, "hr_drug_model": hr_drug_model,
    }


def main():
    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))

    patient = patients["hasta_a"]
    drug = drugs["beta_bloker"]

    print(f"Hasta: {patient.name} | İlaç: {drug.display_name}")

    print("\nCircAdapt baseline (ilaçsız) simülasyonu çalıştırılıyor...")
    print("CircAdapt ilaç-etkili simülasyonu çalıştırılıyor...")
    r = run_comparison(patient, drug)
    drug_effect = r["drug_effect"]

    print(f"Pik etki t={drug_effect['t_peak_hours']:.2f} saat, "
          f"konsantrasyon={drug_effect['conc_peak']:.3f} mg/L, "
          f"etki oranı={drug_effect['effect_fraction']:.2f}")
    print(f"Beklenen nabız: {patient.baseline_hr} -> {drug_effect['hr_drug']:.1f} bpm")
    print(f"Beklenen SBP:   {patient.baseline_sbp} -> {drug_effect['sbp_drug']:.1f} mmHg")

    fig = build_comparison_figure(
        r["t_base"], r["p_base"], r["v_base"],
        r["t_drug"], r["p_drug"], r["v_drug"],
        drug.display_name, r["hr_base"], r["hr_drug_model"],
    )
    out_path = os.path.join(base, "circadapt_drug_integration_sonucu.png")
    fig.savefig(out_path, dpi=150)

    print(f"\nBAŞARILI! Grafik kaydedildi: {out_path}")
    print(f"Baseline maksimum LV basıncı: {r['p_base'].max():.1f} mmHg")
    print(f"İlaçlı maksimum LV basıncı:   {r['p_drug'].max():.1f} mmHg")
    print(f"Baseline LVEDV: {r['v_base'].max():.1f} mL | İlaçlı LVEDV: {r['v_drug'].max():.1f} mL")


if __name__ == "__main__":
    main()
