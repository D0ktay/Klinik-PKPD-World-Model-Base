"""
Simülasyon motoru — tek bir hasta+ilaç kombinasyonunu YÜZLERCE kez
çalıştırıp (Monte Carlo), her seferinde bireysel varyasyonu rastgele
örnekleyerek gerçekçi bir sonuç DAĞILIMI üretir.

Bu dosya, patient.py + pk.py + pd.py'yi bir araya getiren "orkestra
şefi" katmanıdır -- mantığın kendisi değil, mantığı çalıştıran akış.
"""

from dataclasses import dataclass, replace
import numpy as np

from .patient import Patient, Drug
from .pk import (
    plasma_concentration, plasma_concentration_two_compartment, organ_function_adjusted_ke,
    pk_interaction_adjusted_ke, plasma_concentration_infusion,
)
from .pd import (
    emax_effect, apply_effect_to_vitals, effect_compartment_concentration, loewe_combined_effect,
    electrolyte_adjusted_emax_hr, electrolyte_adjusted_emax_sbp,
    AV_NODE_SENSITIVE_DRUG_CLASSES, discrete_av_block_mask,
    AV_BLOCK_THRESHOLD_MULTIPLIER, AV_BLOCK_ESCAPE_RHYTHM_HR,
)


@dataclass
class SimulationResult:
    t: np.ndarray
    hr_runs: np.ndarray    # shape: (n_realizations, n_timepoints)
    sbp_runs: np.ndarray
    conc_runs: np.ndarray
    # Her Monte Carlo denemesinde rastgele örneklenen ke (eliminasyon hızı)
    # ve sensitivity (bireysel duyarlılık) değerleri -- normalde sadece
    # NİHAİ sonuç (hr_runs/sbp_runs) görünür, bu ikisi "perde arkasında"
    # kalır. "Dünya Modelini Gözlemle" sayfasında tek tek denemeleri
    # inceleyebilmek için saklanır. run_polypharmacy_simulation birden
    # fazla ilaç için birden fazla ke/sensitivity örneklediğinden burada
    # None bırakılır (tekil bir "bu deneme" değeri anlamlı değildir).
    ke_values: np.ndarray | None = None
    sensitivity_values: np.ndarray | None = None


def get_plasma_concentration(drug: Drug, t: np.ndarray, weight_kg: float, ke: float) -> np.ndarray:
    """
    Tek-kompartmanlı ilaçlar için (iki-kompartman ayrı, run_monte_carlo/
    run_reference_trace'teki `pk_model="two_compartment"` dalı bunu
    kullanmaz) plazma konsantrasyonunu hesaplayan MERKEZİ giriş noktası --
    dose_mg/dose_mg_per_kg/ka BOLUS yolu (plasma_concentration()) ile
    infusion_rate_mcg_per_kg_min SÜREKLİ İNFÜZYON yolu (plasma_concentration_
    infusion(), bkz. pk.py) arasında dallanır.

    drug.infusion_rate_mcg_per_kg_min DOLU ise (dobutamin/nitroprussid gibi
    klinikte sadece infüzyonla verilen ilaçlar) infüzyon formülü kullanılır
    -- dose_mg/dose_mg_per_kg/ka bu durumda KULLANILMAZ. None ise (VARSAYILAN
    -- tüm diğer ilaçlar) mevcut bolus formülü kullanılır, DAVRANIŞ DEĞİŞMEZ.

    ke: çağıranın (hasta organ fonksiyonu + PK-seviyeli ilaç etkileşimi +
        bireysel varyasyon ile ZATEN ayarlamış olduğu) nihai eliminasyon
        hızı -- bu fonksiyon ke'yi kendisi ayarlamaz, sadece kullanır.
    """
    if drug.infusion_rate_mcg_per_kg_min is not None:
        infusion_rate_mg_hr = drug.infusion_rate_mcg_per_kg_min * weight_kg * 60.0 / 1000.0
        return plasma_concentration_infusion(
            t, infusion_rate_mg_hr, ke, drug.vd_per_kg, weight_kg,
            infusion_duration_hr=drug.infusion_duration_hr,
        )
    return plasma_concentration(
        t, drug.dose_mg, drug.ka, ke, weight_kg, drug.vd_per_kg,
        dose_mg_per_kg=drug.dose_mg_per_kg,
    )


def apply_discrete_av_block(hr: np.ndarray, patient: Patient, av_sensitive_drug_present: bool) -> np.ndarray:
    """
    Discrete (all-or-nothing) AV blok -- Gap #3 (bkz. pd.py > discrete_av_block_mask,
    AV_BLOCK_THRESHOLD_MULTIPLIER, AV_BLOCK_ESCAPE_RHYTHM_HR).

    İKİ ayrı tetikleyici, İKİSİ DE hr'yi AV_BLOCK_ESCAPE_RHYTHM_HR'ye sabitler:
      1. patient.known_av_block_degree == "third" -- hastanın ÖNCEDEN BİLİNEN,
         tanı konmuş tam AV bloğu. Herhangi bir eşik hesabına GİRMEDEN, t=0'dan
         İTİBAREN tüm izi kaçış ritmine sabitler (bu, ilaç/elektrolitten
         bağımsız, kalıcı bir hasta durumu).
      2. Yukarıdaki DEĞİLSE: potassium_av_conduction_factor + (varsa)
         AV-duyarlı ilaç etkisinin kümülatif çarpanı AV_BLOCK_THRESHOLD_
         MULTIPLIER'ı aştığı noktadan itibaren (mandal/latch -- bkz.
         discrete_av_block_mask docstring'i) kaçış ritmine geçer.

    Normal hastalarda (known_av_block_degree None/"none", normal K+,
    AV-duyarlı olmayan/hiç ilaç) HİÇBİR ŞEY DEĞİŞMEZ -- eşik (3.0x), normal
    parametrelerle asla aşılamayacak kadar yüksek, bu yüzden ayrı bir
    "özellik açık mı" bayrağına gerek yok, davranış kendiliğinden geriye
    dönük uyumlu.
    """
    if patient.known_av_block_degree == "third":
        return np.full_like(hr, AV_BLOCK_ESCAPE_RHYTHM_HR)

    mask = discrete_av_block_mask(
        hr, patient.baseline_hr, av_sensitive_drug_present,
        patient.potassium_mEqL, AV_BLOCK_THRESHOLD_MULTIPLIER,
    )
    if mask.any():
        hr = hr.copy()
        hr[mask] = AV_BLOCK_ESCAPE_RHYTHM_HR
    return hr


def run_monte_carlo(patient: Patient, drug: Drug,
                     n_realizations: int = 300,
                     hours: float = 8.0,
                     n_timepoints: int = 200,
                     ke_variation_sigma: float = 0.25,
                     sensitivity_variation_sigma: float = 0.30,
                     measurement_noise_hr: float = 1.2,
                     measurement_noise_sbp: float = 1.5,
                     seed: int = 42,
                     pk_model: str = "one_compartment") -> SimulationResult:
    """
    Aynı hasta+ilaç kombinasyonunu n_realizations kez simüle eder.
    Her denemede:
      - eliminasyon hızı log-normal dağılımdan örneklenir
      - bireysel duyarlılık (sensitivity) log-normal dağılımdan örneklenir
      - ölçüm gürültüsü eklenir (gerçek monitörler de mükemmel değildir)

    pk_model: "one_compartment" (varsayılan, ka/ke_mean/vd_per_kg kullanır)
        veya "two_compartment" (k10/k12/k21/vd_central_per_kg kullanır --
        bu alanları tanımlamayan ilaçlarda ValueError fırlatır).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, hours, n_timepoints)

    hr_runs = np.zeros((n_realizations, n_timepoints))
    sbp_runs = np.zeros((n_realizations, n_timepoints))
    conc_runs = np.zeros((n_realizations, n_timepoints))
    ke_values = np.zeros(n_realizations)
    sensitivity_values = np.zeros(n_realizations)

    if pk_model == "two_compartment" and drug.k10 is None:
        raise ValueError(
            f"{drug.display_name!r} için iki-kompartmanlı PK parametreleri "
            "(k10/k12/k21/vd_central_per_kg) tanımlı değil."
        )

    # Böbrek/karaciğer fonksiyon bozukluğunun eliminasyona etkisi -- ilaca
    # özgü renal_clearance_fraction/hepatic_clearance_fraction=0.0 olan
    # ilaçlarda (örn. esmolol) bu bir NO-OP'tur, ke_mean/k10 değişmeden kalır.
    adjusted_ke_mean = organ_function_adjusted_ke(
        drug.ke_mean, patient.renal_function, patient.hepatic_function,
        drug.renal_clearance_fraction, drug.hepatic_clearance_fraction,
    )
    adjusted_k10 = (
        organ_function_adjusted_ke(
            drug.k10, patient.renal_function, patient.hepatic_function,
            drug.renal_clearance_fraction, drug.hepatic_clearance_fraction,
        ) if drug.k10 is not None else None
    )

    # Hastanın elektrolit durumunun (potasyum/kalsiyum) ilaç etkisini
    # büyütmesi -- CircAdapt tarafında zaten var olan mekanizmanın
    # istatistiksel motordaki karşılığı (bkz. pd.py > electrolyte_adjusted_
    # emax_hr/sbp). Normal elektrolitli hastalarda (K<=5.0, Ca=9.5) NO-OP.
    adjusted_emax_hr = electrolyte_adjusted_emax_hr(drug.emax_hr, drug.drug_class, patient.potassium_mEqL)
    adjusted_emax_sbp = electrolyte_adjusted_emax_sbp(drug.emax_sbp, drug.drug_class, patient.calcium_mgdL)

    for i in range(n_realizations):
        sensitivity = rng.lognormal(mean=0, sigma=sensitivity_variation_sigma)
        sensitivity_values[i] = sensitivity

        if pk_model == "two_compartment":
            # Bireysel varyasyon, tek-kompartmanlı modeldeki gibi eliminasyon
            # hızına (k10) uygulanır -- karşılaştırılabilirlik için tutarlı.
            k10 = adjusted_k10 * rng.lognormal(mean=0, sigma=ke_variation_sigma)
            ke_values[i] = k10
            conc = plasma_concentration_two_compartment(
                t, drug.dose_mg, k10, drug.k12, drug.k21, drug.vd_central_per_kg,
                dose_mg_per_kg=drug.dose_mg_per_kg, weight_kg=patient.weight_kg,
            )
        else:
            ke = adjusted_ke_mean * rng.lognormal(mean=0, sigma=ke_variation_sigma)
            ke_values[i] = ke
            conc = get_plasma_concentration(drug, t, patient.weight_kg, ke)
        # Nabız ve tansiyon etkisi, keo_hr/keo_sbp tanımlıysa FARKLI etki
        # bölgesi (effect-compartment) gecikmeleriyle hesaplanır -- gerçekte
        # bir ilacın iki farklı etkisi aynı anda tepe yapmaz. Tanımlı
        # değilse (None) eski davranışa düşülür: gecikmesiz, doğrudan
        # plazma konsantrasyonundan tek bir etki eğrisi.
        ce_hr = effect_compartment_concentration(conc, drug.keo_hr, t) if drug.keo_hr is not None else conc
        ce_sbp = effect_compartment_concentration(conc, drug.keo_sbp, t) if drug.keo_sbp is not None else conc

        effect_hr = emax_effect(ce_hr, drug.ec50, sensitivity)
        effect_sbp = emax_effect(ce_sbp, drug.ec50, sensitivity)
        hr, sbp = apply_effect_to_vitals(
            patient.baseline_hr, patient.baseline_sbp, effect_hr,
            adjusted_emax_hr, adjusted_emax_sbp, effect_fraction_sbp=effect_sbp,
        )

        hr += rng.normal(0, measurement_noise_hr, size=t.shape)
        sbp += rng.normal(0, measurement_noise_sbp, size=t.shape)

        hr = apply_discrete_av_block(hr, patient, drug.drug_class in AV_NODE_SENSITIVE_DRUG_CLASSES)

        hr_runs[i] = hr
        sbp_runs[i] = sbp
        conc_runs[i] = conc

    return SimulationResult(t=t, hr_runs=hr_runs, sbp_runs=sbp_runs, conc_runs=conc_runs,
                             ke_values=ke_values, sensitivity_values=sensitivity_values)


def run_reference_trace(patient: Patient, drug: Drug, hours: float = 8.0,
                         n_timepoints: int = 200, pk_model: str = "one_compartment") -> dict:
    """
    "Dünya Modelini Gözlemle" sayfası için TEK, gürültüsüz ve rastgele
    örneklenmemiş bir referans iz üretir (ke=ortalama, sensitivity=1.0,
    ölçüm gürültüsü yok). run_monte_carlo yüzlerce RASTGELE denemenin
    dağılımını üretirken, bu fonksiyon "durum + aksiyon -> yeni durum"
    zincirinin TEK bir okunabilir, yeniden üretilebilir örneğini verir --
    her adımın hangi ara değerlerden geçtiğini şeffaf biçimde göstermek
    içindir, istatistiksel bir sonuç değildir.

    Dönüş: {"t", "conc", "effect_hr", "effect_sbp", "hr", "sbp"} -- her biri
    n_timepoints uzunluğunda numpy dizisi.
    """
    t = np.linspace(0, hours, n_timepoints)

    adjusted_ke_mean = organ_function_adjusted_ke(
        drug.ke_mean, patient.renal_function, patient.hepatic_function,
        drug.renal_clearance_fraction, drug.hepatic_clearance_fraction,
    )

    if pk_model == "two_compartment":
        if drug.k10 is None:
            raise ValueError(
                f"{drug.display_name!r} için iki-kompartmanlı PK parametreleri tanımlı değil."
            )
        adjusted_k10 = organ_function_adjusted_ke(
            drug.k10, patient.renal_function, patient.hepatic_function,
            drug.renal_clearance_fraction, drug.hepatic_clearance_fraction,
        )
        conc = plasma_concentration_two_compartment(
            t, drug.dose_mg, adjusted_k10, drug.k12, drug.k21, drug.vd_central_per_kg,
            dose_mg_per_kg=drug.dose_mg_per_kg, weight_kg=patient.weight_kg,
        )
    else:
        conc = plasma_concentration(
            t, drug.dose_mg, drug.ka, adjusted_ke_mean, patient.weight_kg, drug.vd_per_kg,
            dose_mg_per_kg=drug.dose_mg_per_kg,
        )

    ce_hr = effect_compartment_concentration(conc, drug.keo_hr, t) if drug.keo_hr is not None else conc
    ce_sbp = effect_compartment_concentration(conc, drug.keo_sbp, t) if drug.keo_sbp is not None else conc

    effect_hr = emax_effect(ce_hr, drug.ec50, sensitivity=1.0)
    effect_sbp = emax_effect(ce_sbp, drug.ec50, sensitivity=1.0)

    # bkz. run_monte_carlo -- hastanın elektrolit durumu, AV düğümünü/
    # kontraktiliteyi hedefleyen ilaç sınıflarında etkiyi büyütür.
    adjusted_emax_hr = electrolyte_adjusted_emax_hr(drug.emax_hr, drug.drug_class, patient.potassium_mEqL)
    adjusted_emax_sbp = electrolyte_adjusted_emax_sbp(drug.emax_sbp, drug.drug_class, patient.calcium_mgdL)

    hr, sbp = apply_effect_to_vitals(
        patient.baseline_hr, patient.baseline_sbp, effect_hr,
        adjusted_emax_hr, adjusted_emax_sbp, effect_fraction_sbp=effect_sbp,
    )

    return {"t": t, "conc": conc, "effect_hr": effect_hr, "effect_sbp": effect_sbp,
            "hr": hr, "sbp": sbp}


def run_polypharmacy_simulation(patient: Patient, drugs: list[Drug],
                                 n_realizations: int = 300, hours: float = 8.0,
                                 n_timepoints: int = 200,
                                 ke_variation_sigma: float = 0.25,
                                 sensitivity_variation_sigma: float = 0.30,
                                 measurement_noise_hr: float = 1.2,
                                 measurement_noise_sbp: float = 1.5,
                                 seed: int = 42,
                                 interaction_matrix: dict[tuple[int, int], float] | None = None,
                                 drug_keys: list[str] | None = None,
                                 pk_interaction_matrix: dict[tuple[str, str], float] | None = None,
                                 ) -> SimulationResult:
    """
    Birden fazla ilacın AYNI ANDA verildiği bir senaryoyu (polifarmasi)
    simüle eder.

    Kombinasyon mantığı: her ilacın nabız/tansiyon üzerindeki etkisi
    (emax * etki_oranı) TÜM ilaçlar üzerinden TOPLANARAK bazal değerden
    çıkarılır. Delta-tabanlı modelimizde bu, "aynı mekanizmadaki ilaçların
    toplamsal (additive) birleşmesi" ile matematiksel olarak aynı şeydir --
    iki beta-bloker de, bir beta-bloker + bir vazodilatör de aynı toplama
    kuralıyla birleşir. Aralarındaki gerçek FARK, CircAdapt entegrasyonunda
    ortaya çıkar: her `drug_class` farklı bir mekanik parametreyi hedeflediği
    için (bkz. integrate_drug_with_circadapt.py > run_polypharmacy_with_circadapt),
    aynı anda BİRDEN FAZLA CircAdapt parametresi (örn. hem kontraktilite HEM
    damar direnci) değişebilir.

    interaction_matrix: {(i, j): çarpan} -- drugs listesindeki i. ve j.
        ilaç arasında EK bir sinerji/antagonizma terimi uygular
        (etki_i * etki_j * çarpan, hem HR hem SBP'ye ayrı ayrı). Varsayılan
        None = saf toplamsal (ek sinerji yok). Örn. {(0, 1): 0.3} -- 0. ve
        1. ilaç arasında TEMSİLİ bir supra-additive etkileşim (klinikte
        belgelenen, örn. beta-bloker + digoksin kombinasyonunun AV düğümü
        üzerindeki birleşik etkisi gibi bir durumu modellemek için;
        kesin çarpan değeri bir çalışmadan alınmadı, temsilidir).

    drug_keys / pk_interaction_matrix: PD-seviyesindeki interaction_matrix'ten
        AYRI, PK-seviyeli (klerens/AUC) ilaç-ilaç etkileşimi -- bkz. pk.py >
        pk_interaction_adjusted_ke(), configs/drug_pk_interactions.yaml.
        drug_keys, `drugs` listesiyle AYNI SIRADA config anahtarlarını
        taşımalı (örn. ["beta_bloker", "digoxin"]) -- her ilacın hangi
        config kaydına karşılık geldiğini isim üzerinden bulmak için gerekli
        (interaction_matrix'teki index-tabanlı yaklaşımdan farklı olarak, PK
        etkileşimi index'e değil İLACIN KİMLİĞİNE bağlı). pk_interaction_matrix,
        build_pk_interaction_matrix()'in ürettiği {(perpetrator, victim):
        auc_ratio} sözlüğü. İkisi de None ise (varsayılan) HİÇBİR PK
        etkileşimi uygulanmaz -- mevcut davranış DEĞİŞMEZ (regresyon yok).
        Şu an tabloda TEK doğrulanmış kayıt var: esmolol (beta_bloker) ->
        digoxin, Kessler ve ark. 1987 (bkz. CALIBRATION_REPORT.md).

    Fizyolojik üst sınır: toplam etki ne kadar büyük olursa olsun, nabız/
    tansiyon 0'ın altına inemez (np.clip ile garanti edilir) -- toplamsal
    etkiler matematiksel olarak sınırsız büyüyebilse de, sonuç fizyolojik
    olarak anlamlı bir aralıkta kalır.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, hours, n_timepoints)

    hr_runs = np.zeros((n_realizations, n_timepoints))
    sbp_runs = np.zeros((n_realizations, n_timepoints))
    conc_runs = np.zeros((n_realizations, n_timepoints))  # ilk ilacın konsantrasyonu (referans/gösterim için)

    # bkz. run_monte_carlo -- hastanın elektrolit durumu, AV düğümünü/
    # kontraktiliteyi hedefleyen ilaç sınıflarında etkiyi büyütür. İlaç
    # başına, hasta sabit olduğu için realizasyon döngüsünün DIŞINDA bir
    # kez hesaplanır.
    adjusted_emax_hr = [electrolyte_adjusted_emax_hr(d.emax_hr, d.drug_class, patient.potassium_mEqL) for d in drugs]
    adjusted_emax_sbp = [electrolyte_adjusted_emax_sbp(d.emax_sbp, d.drug_class, patient.calcium_mgdL) for d in drugs]
    any_av_sensitive_drug = any(d.drug_class in AV_NODE_SENSITIVE_DRUG_CLASSES for d in drugs)

    for i in range(n_realizations):
        total_hr_delta = np.zeros(n_timepoints)
        total_sbp_delta = np.zeros(n_timepoints)
        effect_hr_list = []
        effect_sbp_list = []

        for d_idx, drug in enumerate(drugs):
            sensitivity = rng.lognormal(mean=0, sigma=sensitivity_variation_sigma)
            ke = organ_function_adjusted_ke(
                drug.ke_mean, patient.renal_function, patient.hepatic_function,
                drug.renal_clearance_fraction, drug.hepatic_clearance_fraction,
            )
            # PK-seviyeli ilaç-ilaç etkileşimi (bkz. pk.py >
            # pk_interaction_adjusted_ke) -- hastanın organ fonksiyonuna göre
            # ayarlanmış ke'nin ÜZERİNE uygulanır. drug_keys/pk_interaction_matrix
            # verilmediyse (varsayılan None) hiçbir şey değişmez.
            if drug_keys is not None and pk_interaction_matrix:
                active_perpetrators = [key for i, key in enumerate(drug_keys) if i != d_idx]
                ke = pk_interaction_adjusted_ke(ke, drug_keys[d_idx], active_perpetrators, pk_interaction_matrix)
            ke = ke * rng.lognormal(mean=0, sigma=ke_variation_sigma)

            conc = get_plasma_concentration(drug, t, patient.weight_kg, ke)
            ce_hr = effect_compartment_concentration(conc, drug.keo_hr, t) if drug.keo_hr is not None else conc
            ce_sbp = effect_compartment_concentration(conc, drug.keo_sbp, t) if drug.keo_sbp is not None else conc

            effect_hr = emax_effect(ce_hr, drug.ec50, sensitivity)
            effect_sbp = emax_effect(ce_sbp, drug.ec50, sensitivity)

            total_hr_delta = total_hr_delta + adjusted_emax_hr[d_idx] * effect_hr
            total_sbp_delta = total_sbp_delta + adjusted_emax_sbp[d_idx] * effect_sbp

            effect_hr_list.append(effect_hr)
            effect_sbp_list.append(effect_sbp)

            if d_idx == 0:
                conc_runs[i] = conc

        if interaction_matrix:
            for (a, b), factor in interaction_matrix.items():
                total_hr_delta = total_hr_delta + factor * adjusted_emax_hr[a] * effect_hr_list[a] * effect_hr_list[b]
                total_sbp_delta = total_sbp_delta + factor * adjusted_emax_sbp[a] * effect_sbp_list[a] * effect_sbp_list[b]

        hr = patient.baseline_hr - total_hr_delta
        sbp = patient.baseline_sbp - total_sbp_delta

        hr += rng.normal(0, measurement_noise_hr, size=t.shape)
        sbp += rng.normal(0, measurement_noise_sbp, size=t.shape)

        hr = np.clip(hr, 0, None)
        hr = apply_discrete_av_block(hr, patient, any_av_sensitive_drug)

        hr_runs[i] = hr
        sbp_runs[i] = np.clip(sbp, 0, None)

    return SimulationResult(t=t, hr_runs=hr_runs, sbp_runs=sbp_runs, conc_runs=conc_runs)


def run_polypharmacy_simulation_loewe(patient: Patient, drugs: list[Drug],
                                       n_realizations: int = 300, hours: float = 8.0,
                                       n_timepoints: int = 200,
                                       ke_variation_sigma: float = 0.25,
                                       sensitivity_variation_sigma: float = 0.30,
                                       measurement_noise_hr: float = 1.2,
                                       measurement_noise_sbp: float = 1.5,
                                       seed: int = 42) -> SimulationResult:
    """
    run_polypharmacy_simulation()'ın Loewe additivity versiyonu -- Monte
    Carlo döngüsü (ke/sensitivity örnekleme, organ fonksiyonu ayarlaması,
    effect-compartment/Keo gecikmesi) AYNI, sadece birleştirme adımı FARKLI:
    additive toplama yerine pd.py > loewe_combined_effect() kullanılır (bkz.
    o fonksiyonun docstring'i -- iki ilacın Emax'ı farklıysa düz toplamanın
    neden yanıltıcı olabileceği, Tallarida'nın "eğrisel izobol" uyarısı).

    interaction_matrix PARAMETRESİ YOK (run_polypharmacy_simulation'daki
    gibi) -- Loewe additivity zaten doz-eşdeğerliğini hesaba kattığı için
    ayrı bir manuel sinerji çarpanına gerek/anlam yok.

    Bireysel duyarlılık (sensitivity), emax_effect()'teki gibi etkiyi
    çarpıp SONRA kırpmak yerine, İLACIN KENDİ EC50'sini ölçekleyerek
    uygulanır (ec50_effective = ec50 / sensitivity) -- çünkü Loewe
    matematiği, Emax'ı hiç aşmayan GEÇERLİ bir Hill eğrisi varsayıyor;
    çarpıp-kırpma bu varsayımı bozardı (0-1.3 aralığına kırpma, izobol
    denkleminin monotonluk garantisini geçersiz kılar).

    UYARI: tüm ilaçların emax_hr'si (ve ayrı ayrı emax_sbp'si) AYNI yönde
    olmalı (bkz. loewe_combined_effect) -- zıt yönlü bir kombinasyon
    (örn. beta-bloker + vazodilatörün refleks taşikardisi) burada
    ValueError fırlatır, bu durumda run_polypharmacy_simulation() kullanın.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, hours, n_timepoints)

    hr_runs = np.zeros((n_realizations, n_timepoints))
    sbp_runs = np.zeros((n_realizations, n_timepoints))
    conc_runs = np.zeros((n_realizations, n_timepoints))
    any_av_sensitive_drug = any(d.drug_class in AV_NODE_SENSITIVE_DRUG_CLASSES for d in drugs)

    for i in range(n_realizations):
        ce_hr_list, ce_sbp_list, ec50_list, emax_hr_list, emax_sbp_list = [], [], [], [], []

        for d_idx, drug in enumerate(drugs):
            sensitivity = rng.lognormal(mean=0, sigma=sensitivity_variation_sigma)
            ke = organ_function_adjusted_ke(
                drug.ke_mean, patient.renal_function, patient.hepatic_function,
                drug.renal_clearance_fraction, drug.hepatic_clearance_fraction,
            ) * rng.lognormal(mean=0, sigma=ke_variation_sigma)

            conc = get_plasma_concentration(drug, t, patient.weight_kg, ke)
            ce_hr = effect_compartment_concentration(conc, drug.keo_hr, t) if drug.keo_hr is not None else conc
            ce_sbp = effect_compartment_concentration(conc, drug.keo_sbp, t) if drug.keo_sbp is not None else conc

            ce_hr_list.append(ce_hr)
            ce_sbp_list.append(ce_sbp)
            ec50_list.append(drug.ec50 / sensitivity)
            # bkz. run_monte_carlo -- hastanın elektrolit durumu, AV
            # düğümünü/kontraktiliteyi hedefleyen ilaç sınıflarında
            # tavan etkiyi (Emax) büyütür.
            emax_hr_list.append(electrolyte_adjusted_emax_hr(drug.emax_hr, drug.drug_class, patient.potassium_mEqL))
            emax_sbp_list.append(electrolyte_adjusted_emax_sbp(drug.emax_sbp, drug.drug_class, patient.calcium_mgdL))

            if d_idx == 0:
                conc_runs[i] = conc

        hr_delta = loewe_combined_effect(ce_hr_list, ec50_list, emax_hr_list)
        sbp_delta = loewe_combined_effect(ce_sbp_list, ec50_list, emax_sbp_list)

        hr = patient.baseline_hr - hr_delta
        sbp = patient.baseline_sbp - sbp_delta

        hr += rng.normal(0, measurement_noise_hr, size=t.shape)
        sbp += rng.normal(0, measurement_noise_sbp, size=t.shape)

        hr = np.clip(hr, 0, None)
        hr = apply_discrete_av_block(hr, patient, any_av_sensitive_drug)

        hr_runs[i] = hr
        sbp_runs[i] = np.clip(sbp, 0, None)

    return SimulationResult(t=t, hr_runs=hr_runs, sbp_runs=sbp_runs, conc_runs=conc_runs)


def build_interaction_matrix(drug_keys: list[str], interactions: list[dict]
                              ) -> dict[tuple[int, int], float]:
    """
    configs/drug_interactions.yaml'dan load_drug_interactions() ile okunan
    ham kayıt listesini, run_polypharmacy_simulation()'ın beklediği
    {(i, j): factor} index-çifti sözlüğüne çevirir.

    drug_keys: kullanıcının SEÇTİĞİ sırayla ilaç anahtarları (örn.
        ["digoxin", "beta_bloker"]) -- interaction kaydındaki drug_a/drug_b
        HANGİ sırada yazıldığından bağımsız olarak, ikisi de drug_keys
        içinde bulunduğu sürece kendi index'lerine doğru eşleştirilir.
        Bir eşleşme bulunmazsa (örn. ilgili çift için henüz kayıt yoksa ya
        da o ilaçlardan biri seçilmediyse) o kayıt sessizce atlanır --
        sonuç boş bir sözlük olabilir, bu da run_polypharmacy_simulation()'ı
        saf toplamsal (additive) davranışına düşürür (mevcut varsayılan,
        regresyon yok).
    """
    key_to_index = {key: i for i, key in enumerate(drug_keys)}
    matrix: dict[tuple[int, int], float] = {}
    for record in interactions:
        a, b = record["drug_a"], record["drug_b"]
        if a in key_to_index and b in key_to_index:
            matrix[(key_to_index[a], key_to_index[b])] = record["factor"]
    return matrix


def build_pk_interaction_matrix(drug_keys: list[str], pk_interactions: list[dict]
                                 ) -> dict[tuple[str, str], float]:
    """
    configs/drug_pk_interactions.yaml'dan load_drug_pk_interactions() ile
    okunan ham kayıt listesini, run_polypharmacy_simulation()'ın beklediği
    {(perpetrator_drug, victim_drug): auc_ratio} sözlüğüne çevirir.

    build_interaction_matrix()'ten (PD-seviyesi) İKİ TEMEL FARKI var:
    1. YÖNLÜ -- (perpetrator, victim) sırası ÖNEMLİ, simetrik DEĞİL
       (esmolol digoksini etkiler, tersi modellenmiyor).
    2. Index yerine İLAÇ ANAHTARI (string) kullanır -- pk_interaction_adjusted_ke()
       bir ilacın ke'sini "hangi perpetrator'lar rejimde aktif" listesine
       bakarak ayarladığı için, isim tabanlı arama index'ten daha doğrudan.

    drug_keys: kullanıcının seçtiği ilaç anahtarları -- hem perpetrator_drug
        hem victim_drug drug_keys içinde bulunmuyorsa (örn. o ilaçlardan
        biri seçilmediyse) o kayıt sessizce atlanır, sonuç boş bir sözlük
        olabilir (hiçbir PK etkileşimi uygulanmaz, mevcut davranış).
    """
    key_set = set(drug_keys)
    matrix: dict[tuple[str, str], float] = {}
    for record in pk_interactions:
        perpetrator, victim = record["perpetrator_drug"], record["victim_drug"]
        if perpetrator in key_set and victim in key_set:
            matrix[(perpetrator, victim)] = record["auc_ratio"]
    return matrix


def summarize(result: SimulationResult) -> dict:
    """Yüzlerce denemeden anlamlı bir klinik özet çıkarır."""
    min_hr_per_run = result.hr_runs.min(axis=1)
    return {
        "mean_min_hr": float(min_hr_per_run.mean()),
        "p5_min_hr": float(np.percentile(min_hr_per_run, 5)),
        "p95_min_hr": float(np.percentile(min_hr_per_run, 95)),
        "pct_bradycardia_risk": float((min_hr_per_run < 50).mean() * 100),
    }


def recommend_dose(patient: Patient, drug: Drug,
                    dose_min_mg: float = 1.0, dose_max_mg: float = 20.0,
                    n_candidates: int = 20, n_realizations: int = 200,
                    max_bradycardia_risk_pct: float = 5.0,
                    circadapt_results: dict | None = None,
                    max_lvedv_increase_pct: float = 20.0,
                    conservative_dose_factor: float = 0.7,
                    polypharmacy_result: SimulationResult | None = None,
                    polypharmacy_description: str = "diğer ilaç(lar)",
                    polypharmacy_risk_multiplier: float = 1.5,
                    **monte_carlo_kwargs) -> dict:
    """
    Aynı ilacın farklı dozlarını tarayıp, bradikardi riskini güvenli sınırda
    (varsayılan %5) tutan EN YÜKSEK dozu önerir -- yani "güvenli sınır
    içinde en fazla tedavi edici etki" mantığı. Hiçbir doz güvenli değilse,
    riski en düşük olan doz önerilir.

    circadapt_results verilirse (integrate_drug_with_circadapt.py'daki
    run_comparison()'ın döndürdüğü sözlük -- "v_base"/"v_drug" içeren),
    istatistiksel risk ayrıca CircAdapt'in ürettiği MEKANİK riskle
    (LVEDV'nin baseline'a göre max_lvedv_increase_pct'ten fazla artması --
    aşırı önyük/preload riski, bkz. Frank-Starling mekanizması)
    birleştirilir. circadapt_results, çağıranın (örn. Streamlit) O ANDA
    seçili dozla zaten çalıştırmış olduğu bir CircAdapt sonucu olmalı --
    bu fonksiyon CircAdapt'i (yavaş olduğu için) kendi taramasında TEKRAR
    çalıştırmaz, sadece dıştan verilen tek bir sonucu değerlendirir.
    Mekanik risk bulunursa, istatistiksel olarak önerilen doz
    conservative_dose_factor ile aşağı çekilir ve gerekçe insan-okunur bir
    metin olarak döndürülür.

    polypharmacy_result verilirse (run_polypharmacy_simulation()'ın
    döndürdüğü bir SimulationResult -- bu ilacın hastanın ZATEN kullandığı
    başka ilaç(lar)la BİRLİKTE verildiği senaryo), o kombinasyonun
    bradikardi riski bu ilacın TEK BAŞINA riskinin polypharmacy_risk_
    multiplier katından fazlaysa (varsayılan 1.5x) "tehlikeli kombinasyon"
    uyarısı eklenir. Bu, dozu OTOMATİK değiştirmez (hangi ilacın dozunun
    düşürüleceği klinik bir karardır, bu fonksiyonun kapsamı dışında) --
    sadece gerekçeye bilgilendirici bir uyarı ekler.
    """
    candidate_doses = np.linspace(dose_min_mg, dose_max_mg, n_candidates)

    candidates = []
    for dose in candidate_doses:
        # dose_mg_per_kg=None: taranan mutlak mg değerinin gerçekten
        # kullanılmasını garantiler -- aksi halde kilo bazlı dozlanan bir
        # ilaçta dose_mg_per_kg her zaman öncelikli olacağından, burada
        # değiştirdiğimiz dose_mg sessizce yok sayılırdı.
        candidate_drug = replace(drug, dose_mg=float(dose), dose_mg_per_kg=None)
        result = run_monte_carlo(patient, candidate_drug, n_realizations=n_realizations,
                                  **monte_carlo_kwargs)
        candidates.append((float(dose), summarize(result)))

    safe = [c for c in candidates if c[1]["pct_bradycardia_risk"] <= max_bradycardia_risk_pct]
    if safe:
        best_dose, best_stats = max(safe, key=lambda c: c[0])
        is_safe = True
    else:
        best_dose, best_stats = min(candidates, key=lambda c: c[1]["pct_bradycardia_risk"])
        is_safe = False

    reasoning_parts = []
    if is_safe:
        reasoning_parts.append(
            f"İstatistiksel PK/PD modeli {best_dose:.1f} mg'ı güvenli buluyor "
            f"(bradikardi riski %{best_stats['pct_bradycardia_risk']:.1f}, "
            f"eşik %{max_bradycardia_risk_pct:.0f})."
        )
    else:
        reasoning_parts.append(
            f"Taranan hiçbir doz %{max_bradycardia_risk_pct:.0f} bradikardi riski "
            f"eşiğinin altına inmiyor; en düşük riskli doz {best_dose:.1f} mg "
            f"(risk %{best_stats['pct_bradycardia_risk']:.1f}) öneriliyor."
        )

    mechanical_risk = False
    lvedv_increase_pct = None
    final_dose = best_dose

    if circadapt_results is not None:
        lvedv_base = float(np.max(circadapt_results["v_base"]))
        lvedv_drug = float(np.max(circadapt_results["v_drug"]))
        lvedv_increase_pct = (lvedv_drug - lvedv_base) / lvedv_base * 100.0
        mechanical_risk = lvedv_increase_pct > max_lvedv_increase_pct

        if mechanical_risk:
            final_dose = best_dose * conservative_dose_factor
            reasoning_parts.append(
                f"UYARI: CircAdapt simülasyonu bu ilacın diyastol sonu hacmi "
                f"(LVEDV) baseline'a göre %{lvedv_increase_pct:.0f} artırdığını "
                f"gösteriyor (eşik %{max_lvedv_increase_pct:.0f}) -- aşırı önyük "
                f"(preload) riski. Bu nedenle istatistiksel öneri yerine daha "
                f"temkinli bir doz olan {final_dose:.1f} mg öneriliyor."
            )
        else:
            reasoning_parts.append(
                f"CircAdapt simülasyonu mekanik risk göstermiyor (LVEDV artışı "
                f"%{lvedv_increase_pct:.0f}, eşik %{max_lvedv_increase_pct:.0f})."
            )

    polypharmacy_risk = False
    polypharmacy_bradycardia_risk_pct = None

    if polypharmacy_result is not None:
        poly_stats = summarize(polypharmacy_result)
        polypharmacy_bradycardia_risk_pct = poly_stats["pct_bradycardia_risk"]
        solo_risk = max(best_stats["pct_bradycardia_risk"], 1e-6)  # sıfıra bölmeyi önle
        polypharmacy_risk = polypharmacy_bradycardia_risk_pct > solo_risk * polypharmacy_risk_multiplier

        if polypharmacy_risk:
            reasoning_parts.append(
                f"TEHLİKELİ KOMBİNASYON UYARISI: {polypharmacy_description} ile "
                f"birlikte verildiğinde bradikardi riski %{best_stats['pct_bradycardia_risk']:.1f}'den "
                f"%{polypharmacy_bradycardia_risk_pct:.1f}'e çıkıyor -- bu ilacın veya "
                f"diğerinin dozunun azaltılması, ya da yakın izlem gerekebilir."
            )
        else:
            reasoning_parts.append(
                f"{polypharmacy_description} ile birlikte verildiğinde bradikardi "
                f"riski anlamlı şekilde artmıyor (%{polypharmacy_bradycardia_risk_pct:.1f})."
            )

    # Hastanın kendi elektrolit durumu normal aralık dışındaysa
    # (ilaçtan bağımsız, sadece hastanın laboratuvar değerlerine dayalı) --
    # bu ilaç sınıfına özel bir risk hesaplamıyoruz, sadece klinisyeni
    # bilgilendiren genel bir uyarı ekliyoruz.
    electrolyte_warning = patient.has_abnormal_electrolytes
    if electrolyte_warning:
        abnormal_notes = []
        if not (3.5 <= patient.potassium_mEqL <= 5.0):
            abnormal_notes.append(f"potasyum {patient.potassium_mEqL:.1f} mEq/L (normal 3.5-5.0)")
        if not (8.5 <= patient.calcium_mgdL <= 10.5):
            abnormal_notes.append(f"kalsiyum {patient.calcium_mgdL:.1f} mg/dL (normal 8.5-10.5)")
        reasoning_parts.append(
            f"LAB UYARISI: Hastanın elektrolit değerleri normal aralık dışında "
            f"({', '.join(abnormal_notes)}) -- bu, kalbin iletim hızını ve/veya "
            f"kontraktilitesini bağımsız olarak etkileyebilir (bkz. CircAdapt "
            f"sonuçları), bu ilacın etkisiyle ETKİLEŞEBİLİR. Yakın EKG/lab izlemi düşünün."
        )

    return {
        "dose_mg": final_dose,
        "statistical_dose_mg": best_dose,
        "stats": best_stats,
        "is_safe": is_safe,
        "max_bradycardia_risk_pct": max_bradycardia_risk_pct,
        "mechanical_risk": mechanical_risk,
        "lvedv_increase_pct": lvedv_increase_pct,
        "polypharmacy_risk": polypharmacy_risk,
        "polypharmacy_bradycardia_risk_pct": polypharmacy_bradycardia_risk_pct,
        "electrolyte_warning": electrolyte_warning,
        "reasoning": " ".join(reasoning_parts),
        "candidates": candidates,
    }


def recommend_polypharmacy_dose_scale(patient: Patient, drugs: list[Drug],
                                       scale_min: float = 0.1, scale_max: float = 3.0,
                                       n_candidates: int = 20, n_realizations: int = 200,
                                       max_bradycardia_risk_pct: float = 5.0,
                                       **monte_carlo_kwargs) -> dict:
    """
    3+ ilaçlı bir kombinasyon için doz önerisi -- recommend_dose()'un
    N-ilaçlı karşılığı. TEK bir ilacın dozunu değil, seçilen TÜM
    ilaçların dozlarını AYNI ORTAK KATSAYI (`scale`) ile ölçekleyip,
    kullanıcının slider'larda seçtiği DOZ ORANINI koruyarak "bu oranı
    kaç katına çıkarırsam güvenli sınırda kalırım" sorusuna cevap verir.

    NEDEN TEK BİR ORTAK KATSAYI (proje sohbetinde tartışılan (a) vs (b)
    kararı): hangi ilacın dozunun sabit kalıp hangisinin değişeceği
    klinik bir tercih -- bu fonksiyon o kararı OTOMATİK vermiyor.
    Kullanıcı zaten her ilaç için ayrı bir doz slider'ına sahip
    (streamlit_app.py > tab_drug); bu fonksiyonun ürettiği `scale`,
    slider'ların ÜSTÜNDE bir bilgi notu olarak gösterilir, DEĞERLERİ
    OTOMATİK DEĞİŞTİRMEZ -- kullanıcı isterse sadece bir ilacı elle
    ayarlayarak kendi (b) senaryosunu zaten üretebilir.

    YÖNTEM: recommend_dose() ile AYNI kalıp -- bisection DEĞİL, grid-scan.
    Gerekçe: risk-scale ilişkisi teorik olarak monoton olsa da, Monte
    Carlo örneklemesinin gürültüsü (özellikle küçük n_realizations'ta)
    bisection'ın gerektirdiği kesin monotonluk varsayımını bozabilir --
    grid-scan + "en yüksek güvenli aday" seçimi buna karşı daha sağlam.

    scale_min/scale_max SINIRLARI HAKKINDA DÜRÜSTLÜK: bu ikisi de,
    recommend_dose()'daki dose_min_mg=1.0/dose_max_mg=20.0 gibi, HİÇBİR
    sayısal ya da klinik gerekçeye dayanmayan, keyfi tarama aralıklarıdır
    -- "0 kat" (ilaçsız, triviyal) ve aşırı büyük/anlamsız katsayılar
    arasında makul bir tarama penceresi açmaktan başka bir amaçları yok.
    Klinik bir doz tavanı/tabanı İDDİA ETMEZLER.

    Dönüş: {"scale": ..., "adjusted_doses": {drug.display_name: mg, ...},
    "stats": ..., "is_safe": ..., "reasoning": ..., "candidates": [...]}
    -- her adayda TÜM ilaçların ayarlanmış dozu candidates içinde saklanır.
    """
    candidate_scales = np.linspace(scale_min, scale_max, n_candidates)

    candidates = []
    for scale in candidate_scales:
        scaled_drugs = [
            replace(
                drug,
                dose_mg=float(drug.dose_mg * scale) if drug.dose_mg_per_kg is None else drug.dose_mg,
                dose_mg_per_kg=(drug.dose_mg_per_kg * scale) if drug.dose_mg_per_kg is not None else None,
            )
            for drug in drugs
        ]
        result = run_polypharmacy_simulation_loewe(patient, scaled_drugs, n_realizations=n_realizations,
                                                     **monte_carlo_kwargs)
        candidates.append((float(scale), scaled_drugs, summarize(result)))

    safe = [c for c in candidates if c[2]["pct_bradycardia_risk"] <= max_bradycardia_risk_pct]
    if safe:
        best_scale, best_drugs, best_stats = max(safe, key=lambda c: c[0])
        is_safe = True
        reasoning = (
            f"İstatistiksel Loewe additivity modeli, mevcut doz ORANINI koruyarak "
            f"{best_scale:.2f} katına kadar çıkarmanın güvenli olduğunu buluyor "
            f"(bradikardi riski %{best_stats['pct_bradycardia_risk']:.1f}, eşik "
            f"%{max_bradycardia_risk_pct:.0f})."
        )
    else:
        best_scale, best_drugs, best_stats = min(candidates, key=lambda c: c[2]["pct_bradycardia_risk"])
        is_safe = False
        reasoning = (
            f"Taranan hiçbir doz ölçeği (x{scale_min:.1f}-x{scale_max:.1f}) "
            f"%{max_bradycardia_risk_pct:.0f} bradikardi riski eşiğinin altına inmiyor; "
            f"en düşük riskli ölçek x{best_scale:.2f} (risk %{best_stats['pct_bradycardia_risk']:.1f}) öneriliyor."
        )

    adjusted_doses = {
        drug.display_name: (drug.dose_mg_per_kg * patient.weight_kg if drug.dose_mg_per_kg is not None else drug.dose_mg)
        for drug in best_drugs
    }

    return {
        "scale": best_scale,
        "adjusted_doses": adjusted_doses,
        "stats": best_stats,
        "is_safe": is_safe,
        "max_bradycardia_risk_pct": max_bradycardia_risk_pct,
        "reasoning": reasoning,
        "candidates": candidates,
    }
