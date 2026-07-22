"""
Doğrulama Katmanı: Sonuçları Gerçek Klinik Veriyle Kıyaslama
=========================================================================

Şu ana kadar hep "girdi gerçek mi" (parametreler literatürden mi geldi)
diye uğraştık. Bu dosya "çıktı gerçekçi mi" sorusuna dönüyor -- modelin
ÜRETTİĞİ sonuçları, yayınlanmış klinik çalışma verileriyle kıyaslıyor.

ÖNEMLİ: Bu testlerden biri (test_bolus_model_underpredicts_titrated_
infusion_trial) BİLİNÇLİ OLARAK bir UYUMSUZLUĞU doğruluyor, "her şey
mükemmel uyuyor" iddiası yerine. Detaylar için CALIBRATION_REPORT.md'ye
bakın -- hangi karşılaştırmanın geçerli, hangisinin geçersiz (senaryo
uyumsuzluğu) olduğu orada dürüstçe belgelendi.

Kaynaklar (aşağıdaki testlerde tekrar tekrar referans veriliyor):
  [A] Esmolol Research Group (1986), Am J Cardiol -- multicenter SVT
      çalışması, 160 hasta. Postoperatif SVT alt grubunda: bazal HR
      139±12 -> tedavi sırasında 106±21 bpm (yükleme 500mcg/kg/dk 1dk,
      idame 25-300 mcg/kg/dk titre edilmiş, ortalama etkili doz
      97.2±5.5 mcg/kg/dk SÜREKLİ İNFÜZYON).
  [B] Geriatrik kataraki cerrahisi çalışması (esmolol vs labetalol,
      perioperatif hipertansiyon) -- "onset of activity occurs within
      2 minutes, with 90% of steady-state beta-blockade occurring
      within 5 minutes" (500mcg/kg bolus + idame infüzyon).
  [C] Yaşlı hastalarda bolus karşılaştırma çalışması -- 50mg esmolol
      bolusunda etki ~5 dk, 100mg bolusunda ~9.5 dk plasebo altında
      kalıyor (dozla orantılı süre uzaması).
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from worldmodel.patient import load_patients, load_drugs
from worldmodel.pk import plasma_concentration
from worldmodel.pd import emax_effect


def _esmolol_and_patient():
    base = os.path.join(os.path.dirname(__file__), "..")
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))
    return patients["hasta_a"], drugs["beta_bloker"]


# --- Kaynak [B]: onset ~2 dk, %90 etki ~5 dk içinde (GEÇERLİ karşılaştırma --
# bu, esmololün genel farmakolojisi, doz rejiminden bağımsız bir zaman sabiti) ---

def test_esmolol_onset_within_two_minutes_matches_published_pharmacology():
    """Kaynak [B]: 'onset of activity occurs within 2 minutes'."""
    patient, drug = _esmolol_and_patient()
    t = np.linspace(0, 0.5, 2000)  # ilk 30 dk
    conc = plasma_concentration(t, drug.dose_mg, drug.ka, drug.ke_mean,
                                 patient.weight_kg, drug.vd_per_kg,
                                 dose_mg_per_kg=drug.dose_mg_per_kg)
    effect = emax_effect(conc, drug.ec50)

    # "Onset" -- etkinin gözle görülür hale geldiği an (ör. maks etkinin %20'sine ulaşma)
    onset_threshold = 0.2 * effect.max()
    onset_idx = np.argmax(effect >= onset_threshold)
    onset_min = t[onset_idx] * 60

    assert onset_min <= 2.5, (
        f"Model onset={onset_min:.2f} dk veriyor, yayınlanmış '2 dakika içinde' "
        f"eşiğiyle (Kaynak [B]) makul bir tolerans içinde değil."
    )


def test_esmolol_90pct_effect_within_five_minutes_matches_published_pharmacology():
    """Kaynak [B]: '90% of steady-state beta-blockade occurring within 5 minutes'."""
    patient, drug = _esmolol_and_patient()
    t = np.linspace(0, 0.5, 2000)
    conc = plasma_concentration(t, drug.dose_mg, drug.ka, drug.ke_mean,
                                 patient.weight_kg, drug.vd_per_kg,
                                 dose_mg_per_kg=drug.dose_mg_per_kg)
    effect = emax_effect(conc, drug.ec50)

    threshold_90pct = 0.9 * effect.max()
    idx_90pct = np.argmax(effect >= threshold_90pct)
    t_90pct_min = t[idx_90pct] * 60

    assert t_90pct_min <= 6.0, (
        f"Model %90 etkiye {t_90pct_min:.2f} dk'da ulaşıyor, yayınlanmış "
        f"'5 dakika içinde' eşiğiyle (Kaynak [B]) makul bir tolerans içinde değil."
    )


# --- Kaynak [C]: dozla orantılı etki süresi (GEÇERLİ karşılaştırma -- göreli
# bir ilişki, mutlak bir sayı değil, bu yüzden bolus/infüzyon farkından etkilenmiyor) ---

def test_higher_bolus_dose_produces_longer_effect_duration_matches_published_relationship():
    """Kaynak [C]: 50mg dozda etki ~5dk, 100mg dozda ~9.5dk sürüyor -- dozla
    ORANTILI (doğrusala yakın) bir süre uzaması var. Modelin de aynı YÖNDE
    (mutlak dakika değil, göreli ilişki) davranmasını doğruluyoruz."""
    patient, _ = _esmolol_and_patient()
    base_drug = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))["beta_bloker"]

    t = np.linspace(0, 1.0, 5000)

    def duration_above_half_max(dose_mg):
        drug = replace(base_drug, dose_mg=dose_mg, dose_mg_per_kg=None)
        conc = plasma_concentration(t, drug.dose_mg, drug.ka, drug.ke_mean,
                                     patient.weight_kg, drug.vd_per_kg)
        effect = emax_effect(conc, drug.ec50)
        above_half = effect >= 0.5 * effect.max()
        return t[above_half].max() - t[above_half].min() if above_half.any() else 0.0

    duration_50mg_equiv = duration_above_half_max(50.0 * 0.5)   # ölçekli -- referans hasta 76kg'a göre orantılı
    duration_100mg_equiv = duration_above_half_max(100.0 * 0.5)

    assert duration_100mg_equiv > duration_50mg_equiv, (
        "Model, yayınlanmış [C] kaynağındaki 'daha yüksek doz -> daha uzun "
        "etki süresi' ilişkisini göstermiyor."
    )


# --- Kaynak [A]: SÜREKLI İNFÜZYONLA titre edilmiş SVT tedavisi (GEÇERSİZ /
# DOĞRUDAN KARŞILAŞTIRILAMAZ senaryo -- bilinçli olarak burada tutuluyor) ---

def test_model_predicted_reduction_fraction_coincidentally_close_to_svt_trial():
    """
    İLGİNÇ AMA GÜVENİLİR BİR KANIT SAYILMAMALI: Başlangıçta bu karşılaştırmanın
    (bolus modeli vs Kaynak [A]'nın sürekli infüzyon SVT çalışması) büyük bir
    sapma göstermesini BEKLİYORDUK (farklı doz rejimi, farklı bazal fizyoloji).
    Test edildiğinde beklenmedik şekilde YAKIN çıktı: model pik etkide
    ~%27 nabız azalması öngörüyor, çalışma ~%24 azalma bildiriyor (Kaynak [A]:
    139->106 bpm). Bu YANILTICI OLABİLİR -- `emax_hr=25` bu çalışmadan
    DEĞİL, temsili/kalibrasyon gerektiren bir tahminden geldi (bkz.
    configs/drugs.yaml yorumu). Yakınlık muhtemelen TESADÜF; iki senaryo
    (tek bolus vs saatlerce süren titre edilmiş infüzyon, normal bazal nabız
    vs SVT) yapısal olarak hâlâ farklı -- bkz. bir sonraki test (süre
    uyumsuzluğu, ki bu GERÇEK ve sağlam bir kanıt).
    """
    patient, drug = _esmolol_and_patient()
    t = np.linspace(0, 0.5, 2000)
    conc = plasma_concentration(t, drug.dose_mg, drug.ka, drug.ke_mean,
                                 patient.weight_kg, drug.vd_per_kg,
                                 dose_mg_per_kg=drug.dose_mg_per_kg)
    effect = emax_effect(conc, drug.ec50)
    peak_effect_fraction = effect.max()

    published_svt_reduction_fraction = (139 - 106) / 139  # Kaynak [A] -- ~%23.7
    model_predicted_reduction_fraction = peak_effect_fraction * (drug.emax_hr / patient.baseline_hr)

    # Sadece bu rastlantısal yakınlığın kaybolmadığını izlemek için --
    # kırılırsa CALIBRATION_REPORT.md'deki notu güncelleyin, alarma geçmeyin.
    deviation = abs(model_predicted_reduction_fraction - published_svt_reduction_fraction)
    assert deviation < 0.20


def test_bolus_model_concentration_wanes_by_60_minutes_unlike_sustained_infusion_trial():
    """
    BU, GERÇEK VE SAĞLAM BİR KAPSAM SINIRI (yukarıdaki testin aksine).
    Kaynak [A]'daki SVT çalışması, etkiyi İNFÜZYON SÜRESİNCE (saatlerce)
    SÜRDÜRECEK şekilde titre edilmiş bir idame dozuyla yürütüldü. Bizim
    modelimiz ise TEK BİR bolus sonrası doğal PK bozunumunu simüle ediyor
    -- idame infüzyonu YOK (bu proje idame infüzyon PK'sını modellemiyor).

    Esmololün eliminasyon yarı ömrü ~9 dakika (Wiest 1991, ayrıca openFDA
    ile bağımsız doğrulandı) -- 60 dakika, ~6.5 yarı ömre denk gelir, PLAZMA
    KONSANTRASYONUNUN neredeyse tamamı sönmüş olmalı (test edildi: pik
    değerin ~%2'sine iniyor). Bu, published çalışmadaki "infüzyon
    süresince sürdürülen SABİT konsantrasyon" ile YAPISAL OLARAK
    KARŞILAŞTIRILAMAZ bir davranış -- modelin BİLİNEN, dürüstçe
    belgelenen bir kapsam sınırı.

    NOT (test yazılırken keşfedilen ayrı bir nüans): ETKİ (effect_fraction),
    konsantrasyon kadar hızlı sönmüyor (~%11 kalıyor, ~%2 değil) -- çünkü
    Emax modelinde pik etki zaten tam doygunluğa ulaşmıyor (%84.5, %100
    değil; ec50=0.03 pik konsantrasyona (0.163) çok yakın), bu yüzden
    kuyruktaki düşük konsantrasyon bile orantısız derecede "görünür" bir
    etki üretmeye devam ediyor (Emax eğrisinin doygun olmayan kısmında).
    Bu GERÇEK bir PK/PD davranışı (bug değil) -- ama bu yüzden bu testte
    KONSANTRASYONU (ham PK, saturasyon etkisinden arınmış) kullanıyoruz,
    "sürdürülen infüzyon" karşılaştırmasını çarpıtmamak için.
    """
    patient, drug = _esmolol_and_patient()
    t = np.linspace(0, 1.0, 5000)  # ilk 1 saat
    conc = plasma_concentration(t, drug.dose_mg, drug.ka, drug.ke_mean,
                                 patient.weight_kg, drug.vd_per_kg,
                                 dose_mg_per_kg=drug.dose_mg_per_kg)

    peak_conc = conc.max()
    conc_at_60min = conc[-1]
    remaining_fraction = conc_at_60min / peak_conc

    assert remaining_fraction < 0.05, (
        f"60 dk'da hâlâ pik konsantrasyonun %{remaining_fraction*100:.1f}'i kalmış -- "
        f"beklenenden çok daha yavaş sönüyor, ke_mean/t1/2 kalibrasyonu "
        f"gözden geçirilmeli."
    )
