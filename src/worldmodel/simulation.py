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
from .pk import plasma_concentration, plasma_concentration_two_compartment, organ_function_adjusted_ke
from .pd import emax_effect, apply_effect_to_vitals, effect_compartment_concentration


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
            conc = plasma_concentration(
                t, drug.dose_mg, drug.ka, ke, patient.weight_kg, drug.vd_per_kg,
                dose_mg_per_kg=drug.dose_mg_per_kg,
            )
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
            drug.emax_hr, drug.emax_sbp, effect_fraction_sbp=effect_sbp,
        )

        hr += rng.normal(0, measurement_noise_hr, size=t.shape)
        sbp += rng.normal(0, measurement_noise_sbp, size=t.shape)

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

    hr, sbp = apply_effect_to_vitals(
        patient.baseline_hr, patient.baseline_sbp, effect_hr,
        drug.emax_hr, drug.emax_sbp, effect_fraction_sbp=effect_sbp,
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
            ) * rng.lognormal(mean=0, sigma=ke_variation_sigma)

            conc = plasma_concentration(
                t, drug.dose_mg, drug.ka, ke, patient.weight_kg, drug.vd_per_kg,
                dose_mg_per_kg=drug.dose_mg_per_kg,
            )
            ce_hr = effect_compartment_concentration(conc, drug.keo_hr, t) if drug.keo_hr is not None else conc
            ce_sbp = effect_compartment_concentration(conc, drug.keo_sbp, t) if drug.keo_sbp is not None else conc

            effect_hr = emax_effect(ce_hr, drug.ec50, sensitivity)
            effect_sbp = emax_effect(ce_sbp, drug.ec50, sensitivity)

            total_hr_delta = total_hr_delta + drug.emax_hr * effect_hr
            total_sbp_delta = total_sbp_delta + drug.emax_sbp * effect_sbp

            effect_hr_list.append(effect_hr)
            effect_sbp_list.append(effect_sbp)

            if d_idx == 0:
                conc_runs[i] = conc

        if interaction_matrix:
            for (a, b), factor in interaction_matrix.items():
                total_hr_delta = total_hr_delta + factor * drugs[a].emax_hr * effect_hr_list[a] * effect_hr_list[b]
                total_sbp_delta = total_sbp_delta + factor * drugs[a].emax_sbp * effect_sbp_list[a] * effect_sbp_list[b]

        hr = patient.baseline_hr - total_hr_delta
        sbp = patient.baseline_sbp - total_sbp_delta

        hr += rng.normal(0, measurement_noise_hr, size=t.shape)
        sbp += rng.normal(0, measurement_noise_sbp, size=t.shape)

        hr_runs[i] = np.clip(hr, 0, None)
        sbp_runs[i] = np.clip(sbp, 0, None)

    return SimulationResult(t=t, hr_runs=hr_runs, sbp_runs=sbp_runs, conc_runs=conc_runs)


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
    riski en düşük olan doz önerilir. (Bu kısım Faz 2'den beri değişmedi.)

    Faz 6: circadapt_results verilirse (integrate_drug_with_circadapt.py'daki
    run_comparison()'ın döndürdüğü sözlük -- "v_base"/"v_drug" içeren),
    istatistiksel risk ayrıca CircAdapt'in ürettiği MEKANİK riskle
    (LVEDV'nin baseline'a göre max_lvedv_increase_pct'ten fazla artması --
    aşırı önyük/preload riski, bkz. Faz 1-3'teki Frank-Starling bulguları)
    birleştirilir. circadapt_results, çağıranın (örn. Streamlit) O ANDA
    seçili dozla zaten çalıştırmış olduğu bir CircAdapt sonucu olmalı --
    bu fonksiyon CircAdapt'i (yavaş olduğu için) kendi taramasında TEKRAR
    çalıştırmaz, sadece dıştan verilen tek bir sonucu değerlendirir.
    Mekanik risk bulunursa, istatistiksel olarak önerilen doz
    conservative_dose_factor ile aşağı çekilir ve gerekçe insan-okunur bir
    metin olarak döndürülür.

    Faz 10: polypharmacy_result verilirse (run_polypharmacy_simulation()'ın
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

    # Faz 11: hastanın kendi elektrolit durumu normal aralık dışındaysa
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
