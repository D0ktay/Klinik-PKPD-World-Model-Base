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
    Tallarida'nın tam-agonist/kısmi-agonist ayrımı (eğrisel izobol) bu
    sınırı KALDIRMIYOR -- SADECE potens oranı sabit olmadığında additive
    izobolün ŞEKLİNİ (düz yerine eğri) düzeltiyor, ulaşılabilir tavan yine
    min(emax_i)'de kalıyor (bkz. RESEARCH_N_DRUG.md §1.1, ADR-5 --
    literatürde (MuSyC hariç, o da bu proje için kalibrasyon verisi
    eksikliği nedeniyle UYGULANAMAZ, bkz. ADR-1) bu tavanı prensipli
    şekilde kaldıran bir yöntem YOK). N büyüdükçe (N≥3) bu kısıt DAHA SIK
    bağlayıcı hale gelir -- en düşük-Emax'lı TEK ilaç, kaç ilaç eklenirse
    eklensin tüm kombinasyonun tavanını belirler. KARAR: kaldırılmıyor,
    bunun yerine kullanıcıya (streamlit_app.py, ADIM 6) hangi ilacın
    tavanı belirlediği AÇIKÇA gösteriliyor.

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


def mechanistic_fraction_combined_effect(concentrations: list[np.ndarray], ec50s: list[float],
                                          emaxes: list[float], baseline: float) -> np.ndarray:
    """
    Zıt yönlü (bir ilaç azaltır, biri artırır) ilaç kombinasyonlarını,
    integrate_drug_with_circadapt.py > cumulative_parameter_multipliers()'ın
    `General.t_cycle` için kullandığı ile BİREBİR AYNI mekanizmayla
    birleştirir (RESEARCH_N_DRUG.md ADR-7 -- ADR-4'ün "gruplama+fark"
    kararının YERİNE geçti).

    NEDEN BU DAHA İYİ: CircAdapt tarafında nabız, "bpm deltası toplama"
    değil, her ilacın kendi İZOLE `hr_fraction`'ının (yeni_hr/bazal_hr)
    kalp siklus süresi (t_cycle) üzerinde ÇARPIMSAL olarak birikmesiyle
    hesaplanıyor: `t_cycle_multiplier /= hr_fraction` (her ilaç için sırayla).
    HR = 60/t_cycle olduğundan bu, birleşik HR'nin `baseline * Π(hr_fraction_i)`
    olduğu anlamına gelir -- bu formül yönden (fraksiyon 1'in üstünde mi
    altında mı) bağımsız, yani zıt yönlü ilaçlar için HİÇBİR özel gruplama/
    işaret ayrımına ihtiyaç duymuyor, Loewe'nin aksine "aynı yönde olma"
    zorunluluğu da yok. Bu formülün order-independent olduğu N_DRUG_AUDIT.md'de
    ölçülerek doğrulandı (24/24 permütasyon bit-identical).

    HR için bu, CircAdapt'in kendi kanonik formülünün BİREBİR mirror'ı --
    iki motor artık gerçekten aynı matematiği paylaşıyor (AV-blok
    düzeltmesindeki (ADR-3) desenle aynı). SBP için CircAdapt'te doğrudan
    bir karşılığı YOK (SBP orada Sf_act/ArtVen.p0'dan PV-loop simülasyonuyla
    EMERGENT olarak çıkıyor, kapalı bir "SBP fraksiyonu" formülü yok) --
    burada aynı çarpımsal-fraksiyon yaklaşımı HR ile TUTARLILIK ve
    öngörülebilirlik (keyfi bir işaret-gruplaması olmaması) gerekçesiyle
    uygulanıyor; bu kısım ⚠️ CircAdapt mirror'ı DEĞİL, tutarlılık amaçlı bir
    mühendislik genellemesidir.

    concentrations/ec50s/emaxes: loewe_combined_effect() ile AYNI anlamda --
    her ilacın etki-bölgesi konsantrasyonu, EC50'si, Emax'ı (işaretli).
    baseline: hastanın bazal HR/SBP'si (delta'yı fraksiyona/geri çevirmek
    için gerekli -- loewe_combined_effect()'in aksine bu formül baseline'a
    göre TANIMLI, çünkü CircAdapt'teki referans mekanizma da öyle).

    Dönüş: loewe_combined_effect() ile AYNI birimde (fizyolojik, örn. bpm),
    işaretli net delta -- yani `baseline - dönüş_değeri` = birleşik HR/SBP.
    """
    cumulative_fraction = np.ones_like(concentrations[0])
    for conc, ec50, emax in zip(concentrations, ec50s, emaxes):
        isolated_value = baseline - emax_effect(conc, ec50, sensitivity=1.0) * emax
        hr_fraction = np.maximum(isolated_value / baseline, 1e-6)
        cumulative_fraction = cumulative_fraction * hr_fraction
    return baseline - baseline * cumulative_fraction


def grouped_loewe_combined_effect(concentrations: list[np.ndarray], ec50s: list[float],
                                   emaxes: list[float], baseline: float,
                                   n_iterations: int = 40) -> np.ndarray:
    """
    loewe_combined_effect()'in ZIT YÖNLÜ ilaç kombinasyonlarını da kabul
    eden versiyonu (N_DRUG_AUDIT.md Şüphe D, RESEARCH_N_DRUG.md ADR-4/ADR-7).

    TEK GRUP (tüm ilaçlar aynı yönde) durumunda bu fonksiyon
    loewe_combined_effect()'in SAYISAL OLARAK BİREBİR AYNISINI döndürür --
    yani mevcut aynı-yönlü N=1/N=2 senaryolarında (tests/test_no_regression_n_drug.py)
    davranış DEĞİŞMEZ. Zıt yönlü bir kombinasyon verildiğinde ise artık
    ADR-4'ün "gruplama+fark" kararı DEĞİL, mechanistic_fraction_combined_effect()
    (ADR-7 -- CircAdapt'in kendi kanonik t_cycle formülünün mirror'ı)
    kullanılıyor -- bkz. o fonksiyonun docstring'i.

    baseline: hastanın bazal HR/SBP'si -- SADECE zıt yönlü dalda
    (mechanistic_fraction_combined_effect) kullanılır, aynı yönlü dalda
    (loewe_combined_effect) hiç dokunulmaz.

    Dönüş: loewe_combined_effect() ile aynı birimde (fizyolojik, örn. bpm),
    işaretli net delta.
    """
    all_same_sign = all((e >= 0) == (emaxes[0] >= 0) for e in emaxes)
    if all_same_sign:
        return loewe_combined_effect(concentrations, ec50s, emaxes, n_iterations=n_iterations)
    return mechanistic_fraction_combined_effect(concentrations, ec50s, emaxes, baseline)


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


def av_conduction_cumulative_multiplier(av_sensitive_hr_fractions: list, potassium_mEqL: float):
    """
    CircAdapt tarafının `Timings.c_tau_av1` üzerinde biriktirdiği KÜMÜLATİF
    çarpanın (bkz. integrate_drug_with_circadapt.py >
    cumulative_av_conduction_multiplier() -- kanonik/referans formül)
    istatistiksel motordaki KARŞILIĞI -- discrete_av_block_mask()
    fonksiyonunun eşik kontrolü için kullanılır.

    DÜZELTME (N_DRUG_AUDIT.md Şüphe E, RESEARCH_N_DRUG.md ADR-3): bu
    fonksiyon ÖNCEDEN (bkz. git geçmişi) TÜM ilaçların BİRLEŞİK (zaten
    additive/Loewe ile toplanmış) TEK bir hr_fraction'ını, "AV-sensitive
    ilaç var mı" bayrağıyla birlikte, TEK SEFERDE k_factor'a bölüyordu --
    bu SADECE bir YAKLAŞIKLIKTI ve CircAdapt'in gerçek ilaç-başına
    çarpımsal birikiminden N büyüdükçe HIZLA sapıyordu (izole ölçüm: N=2'de
    %1.7, N=5'te %52 fark). Artık integrate_drug_with_circadapt.py >
    cumulative_av_conduction_multiplier() İLE BİREBİR AYNI matematiği
    uyguluyor: k_factor'dan başla, HER AV-duyarlı ilaç için AYRI AYRI,
    SIRAYLA çarpımsal olarak böl -- iki motor artık AYNI formülü paylaşıyor,
    sapma ölçülüp belgelenmek yerine GİDERİLDİ.

    av_sensitive_hr_fractions: AV_NODE_SENSITIVE_DRUG_CLASSES'taki HER
        ilacın KENDİ İZOLE hr_fraction'ı (yeni_nabız/bazal_nabız, o ilaç
        TEK BAŞINA verilseydi ne olurdu) -- listedeki SIRA, çağıranın
        ilaçları uyguladığı sırayla AYNI olmalı (integrate_drug_with_
        circadapt.py > run_with_multiple_drugs ile tutarlılık için, though
        çarpma commutative olduğundan matematiksel sonuç sıradan bağımsız
        -- bkz. N_DRUG_AUDIT.md Şüphe G, çalışma-zamanında doğrulandı).
        AV-duyarsız ilaçlar bu listede YER ALMAZ. Boş liste = rejimde
        AV-duyarlı ilaç yok (CircAdapt'teki "sadece k_factor" durumuyla
        aynı).

    Dönüş: liste elemanlarıyla aynı şekilde (np.ndarray ya da float)
    kümülatif çarpan; liste boşsa skaler k_factor.
    """
    multiplier = potassium_av_conduction_factor(potassium_mEqL)
    for hr_fraction in av_sensitive_hr_fractions:
        safe_fraction = np.clip(hr_fraction, 1e-6, None)
        multiplier = multiplier / safe_fraction
    return multiplier


def discrete_av_block_mask(hr: np.ndarray, baseline_hr: float, av_sensitive_hr_fractions: list,
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

    av_sensitive_hr_fractions: bkz. av_conduction_cumulative_multiplier().

    Çağıran, True olan indekslerdeki hr değerlerini AV_BLOCK_ESCAPE_RHYTHM_HR
    ile DEĞİŞTİRİR (bu fonksiyon hr'yi değiştirmez, sadece maskeyi üretir).
    """
    multiplier = av_conduction_cumulative_multiplier(av_sensitive_hr_fractions, potassium_mEqL)
    multiplier = np.broadcast_to(np.asarray(multiplier, dtype=float), hr.shape)
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
