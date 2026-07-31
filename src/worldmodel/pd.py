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


def loewe_combined_effect(concentrations: list[np.ndarray], ec50s: list[float],
                           emaxes: list[float], n_iterations: int = 40) -> np.ndarray:
    """
    Birden fazla ilacın BİRLİKTE ürettiği etkiyi, Loewe additivity (doz
    eşdeğerliği) prensibiyle hesaplar -- pd.py > emax_effect()'in kullandığı
    "oranları topla/çarp" yaklaşımından FARKLI, farmakoloji literatüründe
    (Tallarida, "Quantitative Methods for Assessing Drug Synergism") standart
    kabul edilen bir yöntem.

    NEDEN GEREKLİ: iki ilacın tavan etkisi (Emax) FARKLIYSA (örn. esmolol
    emax_hr=25, digoksin emax_hr=15), etkilerini doğrudan toplamak/çarpmak
    yanıltıcı olabilir -- "eğrisel izobol" sorunu (bkz. Tallarida). Bu
    fonksiyon, her ilacın KENDİ Emax/Hill-1 eğrisini (emax_effect() ile aynı
    matematiksel model: e_i(C) = emax_i * C / (ec50_i + C)) tersine çevirip
    (C_i(e) = ec50_i * e / (emax_i - e) -- belli bir etkiyi üretmek için
    gereken konsantrasyon), şu örtük denklemi sağlayan birleşik etki `e`'yi
    bulur (Grabovsky & Tallarida 2004'ün N-ilaca genelleştirilmiş hâli):

        Σ_i  C_i / C_i(e)  =  1
        ⟺
        Σ_i  C_i * (emax_i - e) / (ec50_i * e)  =  1

    Bu denklemin `e` için kapalı-form çözümü yok (N=1 hariç), ama sol taraf
    `e`'nin MONOTON AZALAN bir fonksiyonu (e→0'da +∞, e→min(emax_i)'de daha
    küçük) -- bu yüzden bisection (ikiye bölme) ile güvenle çözülebilir,
    zaman noktaları üzerinde numpy ile vektörize edilebilir.

    BİLİNEN KAPSAM SINIRI: bu formül birleşik etkiyi min(emax_i) ile
    sınırlıyor -- yüksek tavanlı bir ilaç, düşük tavanlı ortağıyla
    birleşince o düşük tavanı MATEMATİKSEL OLARAK aşamıyor. Grabovsky-
    Tallarida'nın tam-agonist/kısmi-agonist ayrımı bu sınırı kaldırıyor ama
    önemli ölçüde daha karmaşık -- bu proje kapsamında (ilaçların Emax'ları
    aynı büyüklük mertebesinde) bu sınırlama kabul edilebilir bilinçli bir
    basitleştirme.

    concentrations: her ilacın (aynı uzunlukta) zaman dizisi -- HR için
        effect_compartment_concentration(..., keo_hr, ...) çıktısı, SBP için
        keo_sbp çıktısı (ikisi ayrı ayrı çağrılmalı, farklı Emax/keo
        kullandıkları için, bkz. run_polypharmacy_simulation_loewe).
    ec50s, emaxes: concentrations ile AYNI SIRADA, ilaç başına EC50/Emax.

    Dönüş: birleşik etkinin FİZYOLOJİK birimde (örn. doğrudan bpm düşüşü)
    zaman dizisi -- emax_effect()'in döndürdüğü 0-1 fraksiyon DEĞİL, çünkü
    Emax'lar ilaçlar arasında farklı, ortak bir fraksiyon ölçeği anlamsız.

    ÖNEMLİ KISIT: tüm ilaçların Emax'ı AYNI YÖNDE olmalı (hepsi azaltıcı YA
    DA hepsi artırıcı, bkz. drugs.yaml -- emax_hr>0 azaltır, <0 artırır).
    Loewe additivity doz-eşdeğerliği kavramı, ancak aynı yönde etki eden
    ilaçlar için anlamlı -- zıt yönlü bir kombinasyon (örn. bir beta-bloker
    + bir vazodilatörün refleks taşikardisi) izobol modeline uymaz, bu
    durumda run_polypharmacy_simulation()'daki additive model kullanılmalı.
    """
    if any((e >= 0) != (emaxes[0] >= 0) for e in emaxes):
        raise ValueError(
            "loewe_combined_effect: tüm emax değerleri AYNI yönde (hepsi >=0 "
            "ya da hepsi <0) olmalı -- zıt yönlü ilaç kombinasyonları için "
            "Loewe additivity anlamlı değil, additive modeli kullanın."
        )
    sign = 1.0 if emaxes[0] >= 0 else -1.0
    emaxes_abs = [abs(e) for e in emaxes]

    lo = np.zeros_like(concentrations[0])
    hi = np.full_like(concentrations[0], min(emaxes_abs) * (1 - 1e-6))

    for _ in range(n_iterations):
        mid = (lo + hi) / 2
        mid_safe = np.maximum(mid, 1e-12)
        f_mid = sum(
            conc * (emax - mid) / (ec50 * mid_safe)
            for conc, ec50, emax in zip(concentrations, ec50s, emaxes_abs)
        ) - 1.0
        lo = np.where(f_mid > 0, mid, lo)
        hi = np.where(f_mid > 0, hi, mid)

    return sign * (lo + hi) / 2


# İlaç sınıfları -- CircAdapt entegrasyonundaki (integrate_drug_with_circadapt.py
# > apply_drug_effect_to_circadapt) aynı ayrımla TUTARLI: bu iki sınıf
# ventrikül kontraktilitesi/AV iletimi üzerinden etki eder, "vasodilator"
# ise damar direnci üzerinden -- elektrolit duyarlılığı SADECE ilk ikisine
# uygulanır (bkz. electrolyte_adjusted_emax_hr/sbp).
AV_NODE_SENSITIVE_DRUG_CLASSES = ("beta_blocker", "positive_inotrope")


def electrolyte_adjusted_emax_hr(emax_hr: float, drug_class: str | None,
                                  potassium_mEqL: float) -> float:
    """
    Faz 4 doğrulaması sırasında bulunan bir tutarlılık boşluğunu kapatır:
    potassium_av_conduction_factor() ÖNCEDEN sadece CircAdapt tarafında
    (apply_patient_electrolytes_to_circadapt) kullanılıyordu -- istatistiksel
    Monte Carlo motoru (run_monte_carlo/run_polypharmacy_simulation*),
    hastanın potasyum düzeyinden TAMAMEN BAĞIMSIZ çalışıyordu. Gerçek vaka
    raporu kanıtı (bkz. CALIBRATION_REPORT.md §5 -- 82 yaşında, hafif
    hiperkalemik [K=4.90] bir hastada digoksin+beta-bloker kombinasyonunun
    tam AV bloğuna yol açması) bu boşluğun önemli olduğunu gösterdi:
    hiperkalemik bir hasta, AV düğümünü hedefleyen ilaçlara (beta-bloker,
    pozitif inotrop) DAHA DUYARLI olmalı -- vazodilatöre değil (o, damar
    direnci üzerinden etki eder, AV düğümünden bağımsız).

    Dönüş: hiperkalemide (K>5.0) büyütülmüş, normalde (K<=5.0, çarpan=1.0)
    DEĞİŞMEMİŞ bir emax_hr -- normal-elektrolitli hastalarda NO-OP,
    mevcut testleri bozmaz.
    """
    if drug_class not in AV_NODE_SENSITIVE_DRUG_CLASSES:
        return emax_hr
    return emax_hr * potassium_av_conduction_factor(potassium_mEqL)


def electrolyte_adjusted_emax_sbp(emax_sbp: float, drug_class: str | None,
                                   calcium_mgdL: float) -> float:
    """
    electrolyte_adjusted_emax_hr()'ın kalsiyum/kontraktilite karşılığı --
    aynı gerekçe, aynı ilaç sınıfı kısıtı (AV_NODE_SENSITIVE_DRUG_CLASSES).
    Normal kalsiyumda (9.5 mg/dL, çarpan=1.0) NO-OP.
    """
    if drug_class not in AV_NODE_SENSITIVE_DRUG_CLASSES:
        return emax_sbp
    return emax_sbp * calcium_contractility_factor(calcium_mgdL)


def av_conduction_cumulative_multiplier(hr_fraction, av_sensitive_drug_present: bool,
                                         potassium_mEqL: float):
    """
    CircAdapt tarafının `Timings.c_tau_av1` üzerinde biriktirdiği KÜMÜLATİF
    çarpanın (bkz. integrate_drug_with_circadapt.py > apply_patient_
    electrolytes_to_circadapt: `c_tau_av1 *= k_factor`; apply_drug_effect_
    to_circadapt: AV_NODE_SENSITIVE_DRUG_CLASSES için `c_tau_av1 /= hr_fraction`)
    istatistiksel motordaki (bu dosyanın kendi elemanlarıyla kurulmuş) KARŞILIĞI
    -- discrete_av_block_mask() fonksiyonunun eşik kontrolü için kullanılır.

    YAKLAŞIKLIK NOTU: istatistiksel motor CircAdapt gibi ilaçları SIRAYLA
    çarpımsal biriktirmiyor (additive model -- bkz. run_polypharmacy_simulation
    docstring'i); burada TÜM ilaçların BİRLEŞİK (zaten toplanmış) hr_fraction'ı
    tek bir "AV-sensitive ilaç var mı" bayrağıyla birlikte AYNI k_factor/
    hr_fraction formülüne sokuluyor. Bu, CircAdapt'in ilaç-başına çarpımsal
    birikimini BİREBİR TEKRARLAMIYOR (o, ayrı ve çok daha büyük bir mimari
    değişiklik gerektirirdi) -- sadece AYNI fiziksel yönü (AV-sensitive ilaç +
    hiperkalemi -> büyüyen çarpan) aynı formülle, mevcut additive motorun
    ÇIKTISI üzerinden yaklaşık olarak yeniden üretiyor.

    hr_fraction: anlık nabız / bazal nabız (np.ndarray veya float) -- 0'a
        bölünmeyi önlemek için 1e-6'da alt sınırlanır (hr zaten 0'da
        clip'leniyor, ama emniyet için).
    av_sensitive_drug_present: rejimdeki ilaçlardan EN AZ biri
        AV_NODE_SENSITIVE_DRUG_CLASSES'taysa True -- CircAdapt'teki gibi,
        sadece elektrolit varsa (ilaç yok/AV-duyarsız) çarpan k_factor'da
        sabit kalır, hr_fraction'a bölünmez.

    Dönüş: hr_fraction ile aynı şekilde (np.ndarray ya da float) kümülatif çarpan.
    """
    k_factor = potassium_av_conduction_factor(potassium_mEqL)
    if not av_sensitive_drug_present:
        return np.full_like(np.asarray(hr_fraction, dtype=float), k_factor) if isinstance(hr_fraction, np.ndarray) else k_factor
    safe_fraction = np.clip(hr_fraction, 1e-6, None)
    return k_factor / safe_fraction


def discrete_av_block_mask(hr: np.ndarray, baseline_hr: float, av_sensitive_drug_present: bool,
                            potassium_mEqL: float, threshold_multiplier: float) -> np.ndarray:
    """
    hr zaman dizisindeki, av_conduction_cumulative_multiplier()'ın
    threshold_multiplier'ı AŞTIĞI İLK noktadan İTİBAREN (o noktadan sonraki
    TÜM noktalar dahil -- "latch"/mandal davranışı, tek tek titreşen bir
    maske değil) True olan bir boolean maske döndürür.

    Mandal (latch) davranışının gerekçesi: AV bloğu, konsantrasyon azaldıkça
    anlık olarak "açılıp kapanan" bir durum değil -- klinikte bir kez tetiklenen
    ciddi bir iletim bozukluğu, ilacın etkisi azaldıkça KADEMELİ düzelir, ama
    bu basit/lump modelde kademeli düzelmeyi ayrıca modellemek (Wenckebach tipi
    periyodiklik gibi) mevcut soyutlama seviyesinin ötesinde -- bkz.
    CALIBRATION_REPORT.md Gap #3 notu.

    Çağıran, True olan indekslerdeki hr değerlerini AV_BLOCK_ESCAPE_RHYTHM_HR
    ile DEĞİŞTİRİR (bu fonksiyon hr'yi değiştirmez, sadece maskeyi üretir).
    """
    hr_fraction = hr / baseline_hr
    multiplier = av_conduction_cumulative_multiplier(hr_fraction, av_sensitive_drug_present, potassium_mEqL)
    crossed = multiplier >= threshold_multiplier
    mask = np.zeros_like(crossed, dtype=bool)
    if crossed.any():
        first_idx = int(np.argmax(crossed))
        mask[first_idx:] = True
    return mask


# Discrete (all-or-nothing) AV blok -- Gap #3. Sürekli c_tau_av1 çarpanı
# (potassium_av_conduction_factor + ilaç etkisi) belirli bir eşiği aşınca,
# CircAdapt'in kendisi sayısal olarak ÇÖKÜYOR (izole denemeyle doğrulandı --
# 5.0x'te ModelCrashed, 3.0x-5.0x altında stabil) -- yani gerçekte "sürekli
# büyüyen bir gecikme" fizyolojik olarak anlamlı değil, belli bir noktadan
# sonra iletim TAMAMEN KESİLİR (klinikte: tam/3. derece AV blok, ventriküllerin
# kendi -- çok daha yavaş -- kaçış ritmiyle atması).
#
# THRESHOLD_MULTIPLIER=3.0: izole CircAdapt deneyinde gözlemlenen EN DÜŞÜK
# çöküş sınırının (5.0x) ALTINDA bir güvenlik payı -- hastalar arası
# değişkenlik nedeniyle "çöküşün hemen altı" (örn. 4.9x) güvenli sayılmadı.
# Bu, dose_min_mg/scale_min gibi mevcut "keyfi ama belgelenmiş" tarama/eşik
# sabitleriyle AYNI kategoride -- klinik bir kesinlik iddiası taşımaz.
AV_BLOCK_THRESHOLD_MULTIPLIER = 3.0

# ESCAPE_RHYTHM_HR=35.0: idioventriküler kaçış ritmi -- AV düğümü tamamen
# bloklandığında, ventriküllerin (üstten hiç uyarı alamadıkları için) KENDİ
# doğal, çok daha yavaş uyarı üretme yeteneğiyle atması. Standart kardiyoloji
# literatüründe bu ritim 20-40 bpm aralığında kabul edilir; 35, bu aralığın
# temsili orta noktasıdır. ⚠️ TEMSİLİ SABİT -- hasta/vaka-özel bir ölçümden
# gelmiyor, esmolol/nikardipin EC50'lerinde kullanılan ⚠️ temsili kategorisiyle
# AYNI güven seviyesinde.
AV_BLOCK_ESCAPE_RHYTHM_HR = 35.0


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
