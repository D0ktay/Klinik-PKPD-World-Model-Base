"""
Farmakodinamik (PD / bkz. Sözlük) modülü — "Bu konsantrasyon, vücutta
ne kadar etki yapar?"

Emax modeli kullanıyoruz: etki, konsantrasyon arttıkça belli bir tavana
(Emax -- ilacın yapabileceği MAKSİMUM etkinin büyüklüğü) doğru SATÜRE
olarak artar. Bu gerçek reseptör biyolojisine daha yakın bir davranış
-- lineer bir "doz arttıkça etki de orantılı artar" varsayımı gerçek
hayatta yanlış olurdu (reseptörler doyar).
"""

import numpy as np


def emax_effect(conc: np.ndarray, ec50: float, sensitivity: float = 1.0) -> np.ndarray:
    """
    conc: plazma konsantrasyonu (pk.py'den gelir)
    ec50: yarı-maksimum etki için gereken konsantrasyon
    sensitivity: bireysel duyarlılık çarpanı (Monte Carlo'da rastgele örneklenir)

    Dönüş: 0-1 arası bir "etki oranı" (0 = etki yok, 1 = maksimum etki)
    """
    effect_fraction = sensitivity * conc / (ec50 + conc)
    return np.clip(effect_fraction, 0, 1.3)


def apply_effect_to_vitals(baseline_hr: float, baseline_sbp: float,
                            effect_fraction_hr: np.ndarray,
                            emax_hr: float, emax_sbp: float,
                            effect_fraction_sbp: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Etki oranını gerçek vital bulgulara (nabız, tansiyon) dönüştürür.

    effect_fraction_sbp verilmezse effect_fraction_hr ile aynı kabul
    edilir (geriye dönük uyumluluk -- nabız ve tansiyonun aynı, gecikmesiz
    etki eğrisini paylaştığı eski davranış). Ayrı bir effect_fraction_sbp
    verildiğinde (bkz. effect_compartment_concentration), nabız ve tansiyon
    FARKLI zamanlamalarla etkilenebilir -- gerçekte bir ilacın farklı
    etkileri aynı anda ortaya çıkmaz.
    """
    if effect_fraction_sbp is None:
        effect_fraction_sbp = effect_fraction_hr
    hr = baseline_hr - emax_hr * effect_fraction_hr
    sbp = baseline_sbp - emax_sbp * effect_fraction_sbp
    return hr, sbp


def effect_compartment_concentration(conc: np.ndarray, keo: float,
                                      t_hours: np.ndarray) -> np.ndarray:
    """
    Plazma konsantrasyonundan (conc) "etki bölgesi" (effect-site)
    konsantrasyonunu hesaplar -- dCe/dt = keo * (Cp - Ce) modelinin
    ayrık-zamanlı (zero-order-hold) kapalı-form çözümü:

        Ce[i+1] = Cp[i] + (Ce[i] - Cp[i]) * exp(-keo * dt)

    Bu, plazma konsantrasyonu değişse bile etkinin ona ANINDA değil,
    keo hızıyla "yetişerek" ulaştığı bir gecikme filtresidir. keo büyüdükçe
    gecikme azalır (etki plazmaya hızlı yetişir); keo küçüldükçe gecikme
    artar. PK/PD literatüründe buna "Keo modeli" veya "effect-compartment
    modeli" denir.

    conc: plazma konsantrasyonu dizisi (pk.py'den)
    keo: etki bölgesi denge hızı sabiti (1/saat)
    t_hours: conc ile aynı uzunlukta zaman dizisi (saat)

    Dönüş: etki bölgesi konsantrasyonu (conc ile aynı birimde)
    """
    ce = np.zeros_like(conc)
    for i in range(1, len(t_hours)):
        dt = t_hours[i] - t_hours[i - 1]
        decay = np.exp(-keo * dt)
        ce[i] = conc[i - 1] + (ce[i - 1] - conc[i - 1]) * decay
    return ce


def potassium_av_conduction_factor(potassium_mEqL: float) -> float:
    """
    Hiperkalemi kalp iletim sistemini (özellikle AV düğümü/His-Purkinje)
    YAVAŞLATIR -- azalan dinlenim membran potansiyeli, azalan hızlı Na+
    kanalı kullanılabilirliği nedeniyle (gerçek, iyi bilinen fizyoloji;
    klinikte EKG'de PR uzaması / QRS genişlemesi olarak görülür).
    Normal aralıkta (3.5-5.0 mEq/L) etki yok; hipokalemi bu basit modelde
    iletim hızını değiştirmiyor (asıl riski repolarizasyon/ektopik
    aritmidir, bu kapsamda ayrıca modellenmiyor -- bkz. Patient.
    has_abnormal_electrolytes ve recommend_dose() uyarısı).

    Dönüş: AV iletim gecikmesi çarpanı (1.0 = normal, >1.0 = yavaşlamış
    iletim). CircAdapt'te Timings.c_tau_av1'e uygulanır (bkz.
    integrate_drug_with_circadapt.py > apply_patient_electrolytes_to_circadapt).
    Eğim (0.3) TEMSİLİDİR -- yönü/varlığı gerçek fizyoloji, kesin sayısı
    bir doz-yanıt çalışmasından kalibre edilmedi.
    """
    return 1.0 + 0.3 * max(0.0, potassium_mEqL - 5.0)


def calcium_contractility_factor(calcium_mgdL: float) -> float:
    """
    Kalsiyum, miyokard kontraktilitesiyle DOĞRU orantılıdır --
    eksitasyon-kontraksiyon kenetlenmesinde (excitation-contraction
    coupling) doğrudan rol oynar (gerçek, iyi bilinen fizyoloji).
    Normal aralığın (8.5-10.5 mg/dL) orta noktasına (9.5) göre ölçeklenir.

    Dönüş: kontraktilite çarpanı (1.0 = normal orta nokta, <1.0 =
    hipokalsemi/azalmış kontraktilite, >1.0 = hiperkalsemi/artmış
    kontraktilite). CircAdapt'te Patch.Sf_act'e uygulanır. Eğim (0.08)
    TEMSİLİDİR -- yönü gerçek fizyoloji, kesin sayısı kalibre edilmedi.
    """
    return 1.0 + 0.08 * (calcium_mgdL - 9.5)
