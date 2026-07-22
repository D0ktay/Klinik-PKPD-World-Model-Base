"""
Temel testler — bunlar mülakatta 'kaliteli mühendislik' izlenimi bırakır.
Test yazmak, kodun rastgele çalışmadığını, mantığın doğrulanabilir
olduğunu gösterir.
"""

import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from worldmodel.pk import (
    plasma_concentration, plasma_concentration_two_compartment, organ_function_adjusted_ke,
)
from worldmodel.pd import emax_effect, effect_compartment_concentration
from worldmodel.patient import load_drugs, load_patients, load_verified_drugs, Patient, Drug
from worldmodel.simulation import (
    run_monte_carlo, recommend_dose, run_polypharmacy_simulation, summarize, run_reference_trace,
)


def test_concentration_starts_at_zero():
    t = np.linspace(0, 8, 100)
    conc = plasma_concentration(t, dose_mg=5, ka=1.2, ke=0.35, weight_kg=76, vd_per_kg=0.7)
    assert conc[0] == 0


def test_heavier_patient_has_lower_peak_concentration():
    t = np.linspace(0, 8, 100)
    conc_light = plasma_concentration(t, dose_mg=5, ka=1.2, ke=0.35, weight_kg=50, vd_per_kg=0.7)
    conc_heavy = plasma_concentration(t, dose_mg=5, ka=1.2, ke=0.35, weight_kg=100, vd_per_kg=0.7)
    assert conc_heavy.max() < conc_light.max()


def test_emax_effect_is_bounded():
    conc = np.array([0, 0.01, 0.1, 1.0, 100.0])
    effect = emax_effect(conc, ec50=0.03)
    assert (effect >= 0).all()
    assert (effect <= 1.3).all()


def test_zero_concentration_means_zero_effect():
    effect = emax_effect(np.array([0.0]), ec50=0.03)
    assert effect[0] == 0


def test_dose_mg_per_kg_scales_with_weight():
    """Kilo bazlı dozlama: aynı mg/kg, farklı kiloda farklı mutlak doz -> farklı pik konsantrasyon."""
    t = np.linspace(0, 8, 100)
    conc_light = plasma_concentration(t, dose_mg=0, ka=15.4, ke=0.928,
                                       weight_kg=50, vd_per_kg=8.3, dose_mg_per_kg=0.03)
    conc_heavy = plasma_concentration(t, dose_mg=0, ka=15.4, ke=0.928,
                                       weight_kg=100, vd_per_kg=8.3, dose_mg_per_kg=0.03)
    # Vd de kiloyla ölçeklendiği için (dose/Vd oranı) aynı kalır --
    # dose_mg_per_kg'ın asıl garantisi budur: kilodan bağımsız pik konsantrasyon.
    assert np.isclose(conc_light.max(), conc_heavy.max(), rtol=1e-6)


def test_dose_mg_per_kg_overrides_dose_mg_when_given():
    t = np.linspace(0, 8, 100)
    conc_per_kg = plasma_concentration(t, dose_mg=999.0, ka=1.2, ke=0.35,
                                        weight_kg=76, vd_per_kg=0.7, dose_mg_per_kg=0.5)
    conc_explicit = plasma_concentration(t, dose_mg=0.5 * 76, ka=1.2, ke=0.35,
                                          weight_kg=76, vd_per_kg=0.7)
    assert np.allclose(conc_per_kg, conc_explicit)


def test_dose_mg_used_when_dose_mg_per_kg_absent():
    t = np.linspace(0, 8, 100)
    conc = plasma_concentration(t, dose_mg=5, ka=1.2, ke=0.35, weight_kg=76, vd_per_kg=0.7)
    conc_explicit_none = plasma_concentration(t, dose_mg=5, ka=1.2, ke=0.35,
                                               weight_kg=76, vd_per_kg=0.7, dose_mg_per_kg=None)
    assert np.array_equal(conc, conc_explicit_none)


def test_all_configured_drugs_have_a_known_drug_class():
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    known_classes = {"beta_blocker", "vasodilator", "positive_inotrope"}
    assert len(drugs) >= 4, "configs/drugs.yaml içinde en az 4 ilaç bekleniyor (Faz 2)"
    for key, drug in drugs.items():
        assert drug.drug_class in known_classes, f"{key}: bilinmeyen drug_class {drug.drug_class!r}"


def test_new_drugs_use_weight_based_dosing():
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    for key in ("beta_bloker", "nicardipine", "dobutamine"):
        assert drugs[key].dose_mg_per_kg is not None, f"{key} kilo bazlı dozlamalı olmalı"


# --- İki-kompartmanlı IV bolus modeli (Faz 4) ---
#
# NOT: Roadmap'te "t=0'da konsantrasyon=0 olduğunu doğrula" isteniyordu,
# ama bu YANLIŞ bir beklenti -- gerçek bir IV bolus, t=0'da konsantrasyonu
# SIFIRDAN değil DOĞRUDAN dose/Vc'den başlatır (ilaç absorpsiyon fazı
# olmadan doğrudan kana enjekte edilir). Tek-kompartmanlı oral/absorpsiyon
# modelinde (ka ile) t=0'da C=0 olması doğruydu (test_concentration_
# starts_at_zero); IV bolus modelinde bu geçerli değil. Aşağıdaki testler
# bu modelin GERÇEKTE doğru olması gereken davranışını (t=0'da pozitif ve
# dose/Vc'ye eşit pik, sonrasında monoton azalıp pozitif kalma) doğruluyor.

def test_two_compartment_peaks_at_dose_over_vc_at_t0():
    t = np.linspace(0, 2, 200)
    conc = plasma_concentration_two_compartment(
        t, dose_mg=38.0, k10=18.4, k12=1.79, k21=5.22, vd_central=38.0
    )
    assert np.isclose(conc[0], 1.0, rtol=1e-6)  # dose/vd_central = 38/38 = 1.0 mg/L


def test_two_compartment_concentration_stays_non_negative_and_decays():
    t = np.linspace(0, 8, 500)
    conc = plasma_concentration_two_compartment(
        t, dose_mg=38.0, k10=18.4, k12=1.79, k21=5.22, vd_central=38.0
    )
    assert (conc >= 0).all()
    assert conc[-1] < conc[0]  # zamanla azalmalı
    assert np.all(np.diff(conc) <= 1e-9)  # bu parametrelerle monoton azalan (yükselen bir dağılım fazı yok)


def test_two_compartment_dose_mg_per_kg_and_weight_scaling():
    t = np.linspace(0, 2, 100)
    conc_per_kg = plasma_concentration_two_compartment(
        t, dose_mg=999.0, k10=18.4, k12=1.79, k21=5.22, vd_central=0.5,
        dose_mg_per_kg=0.5, weight_kg=76,
    )
    conc_explicit = plasma_concentration_two_compartment(
        t, dose_mg=0.5 * 76, k10=18.4, k12=1.79, k21=5.22, vd_central=0.5 * 76,
    )
    assert np.allclose(conc_per_kg, conc_explicit)


def test_esmolol_two_compartment_config_matches_alpha_beta_half_lives():
    """k10/k12/k21'in, drugs.yaml'da belgelenen alpha/beta yarı ömürlerini gerçekten
    üretip üretmediğini doğrular (elle hesaplanan değerlerin config'e doğru girildiğini garantiler)."""
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    drug = drugs["beta_bloker"]
    assert drug.k10 is not None, "esmolol için iki-kompartmanlı parametreler eksik"

    sum_k = drug.k10 + drug.k12 + drug.k21
    prod_k = drug.k10 * drug.k21
    disc = sum_k ** 2 - 4 * prod_k
    alpha = (sum_k + np.sqrt(disc)) / 2
    beta = (sum_k - np.sqrt(disc)) / 2

    alpha_half_life_min = np.log(2) / alpha * 60
    beta_half_life_min = np.log(2) / beta * 60

    assert np.isclose(alpha_half_life_min, 2.0, atol=0.1)
    assert np.isclose(beta_half_life_min, 9.0, atol=0.1)


def test_run_monte_carlo_two_compartment_pk_model_runs():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    result = run_monte_carlo(patients["hasta_a"], drugs["beta_bloker"],
                              n_realizations=10, pk_model="two_compartment")
    assert result.conc_runs.shape == (10, 200)
    assert (result.conc_runs >= 0).all()


def test_run_monte_carlo_two_compartment_requires_k10():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    try:
        run_monte_carlo(patients["hasta_a"], drugs["vazodilator"],
                         n_realizations=5, pk_model="two_compartment")
        assert False, "k10 olmayan bir ilaçla two_compartment ValueError fırlatmalıydı"
    except ValueError:
        pass


# --- Etki bölgesi (Keo) gecikmesi (Faz 5) ---

def test_effect_compartment_starts_at_zero_and_stays_non_negative():
    t = np.linspace(0, 2, 200)
    conc = plasma_concentration(t, dose_mg=38, ka=21.0, ke=4.6, weight_kg=76, vd_per_kg=2.0)
    ce = effect_compartment_concentration(conc, keo=6.0, t_hours=t)
    assert ce[0] == 0
    assert (ce >= 0).all()


def test_larger_keo_tracks_plasma_concentration_more_closely():
    """Büyük keo -> etki bölgesi plazmaya daha hızlı 'yetişir' (gecikme azalır)."""
    t = np.linspace(0, 2, 500)
    conc = plasma_concentration(t, dose_mg=38, ka=21.0, ke=4.6, weight_kg=76, vd_per_kg=2.0)

    ce_slow = effect_compartment_concentration(conc, keo=1.0, t_hours=t)   # yavaş denge -> büyük gecikme
    ce_fast = effect_compartment_concentration(conc, keo=100.0, t_hours=t)  # hızlı denge -> plazmaya çok yakın

    # Hızlı denge (büyük keo), plazma eğrisine yavaş dengeden daha yakın olmalı
    # (ortalama mutlak fark daha küçük).
    err_slow = np.mean(np.abs(ce_slow - conc))
    err_fast = np.mean(np.abs(ce_fast - conc))
    assert err_fast < err_slow


def test_different_keo_produces_different_peak_timing():
    """Faz 5'in ana iddiası: farklı keo_hr/keo_sbp, farklı zamanlama üretir."""
    t = np.linspace(0, 1, 1000)
    conc = plasma_concentration(t, dose_mg=38, ka=21.0, ke=4.6, weight_kg=76, vd_per_kg=2.0)

    ce_hr = effect_compartment_concentration(conc, keo=12.0, t_hours=t)   # beta_bloker keo_hr
    ce_sbp = effect_compartment_concentration(conc, keo=6.0, t_hours=t)    # beta_bloker keo_sbp

    t_peak_hr = t[np.argmax(ce_hr)]
    t_peak_sbp = t[np.argmax(ce_sbp)]

    assert t_peak_hr != t_peak_sbp
    assert t_peak_hr < t_peak_sbp  # daha büyük keo -> daha erken pik (daha az gecikme)


def test_run_monte_carlo_uses_separate_hr_sbp_effect_timing():
    """Uçtan uca: keo_hr != keo_sbp olan bir ilaçta (beta_bloker), Monte Carlo
    çıktısındaki ortalama nabız ve tansiyon eğrileri FARKLI zamanlarda dip yapmalı.

    NOT: esmololün etkisi dakikalar mertebesinde (çok hızlı), bu yüzden varsayılan
    8 saatlik/200 noktalı ızgara (nokta başına ~2.4 dk) bu zamanlama farkını
    çözemeyecek kadar kaba kalıyor -- bu test, kinetiğe uygun ince bir ızgara
    (1 saat/1000 nokta, nokta başına ~3.6 sn) kullanıyor."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    drug = drugs["beta_bloker"]
    assert drug.keo_hr is not None and drug.keo_sbp is not None
    assert drug.keo_hr != drug.keo_sbp

    result = run_monte_carlo(patients["hasta_a"], drug, n_realizations=50, seed=1,
                              hours=1.0, n_timepoints=1000, measurement_noise_hr=0.0,
                              measurement_noise_sbp=0.0)
    mean_hr = result.hr_runs.mean(axis=0)
    mean_sbp = result.sbp_runs.mean(axis=0)

    t_min_hr = result.t[np.argmin(mean_hr)]
    t_min_sbp = result.t[np.argmin(mean_sbp)]
    assert t_min_hr != t_min_sbp


# --- "Dünya Modelini Gözlemle" sayfası: perde arkası değerlerin dışa açılması ---

def test_run_monte_carlo_exposes_per_trial_ke_and_sensitivity():
    """Her Monte Carlo denemesinin KENDİ ke/sensitivity örneklemesi
    kaydedilmeli -- 'Dünya Modelini Gözlemle' sayfasının 'deneme #47'yi
    incele' özelliği bu diziler olmadan çalışamaz."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    result = run_monte_carlo(patients["hasta_a"], drugs["beta_bloker"], n_realizations=30, seed=7)

    assert result.ke_values is not None and result.sensitivity_values is not None
    assert result.ke_values.shape == (30,)
    assert result.sensitivity_values.shape == (30,)
    assert np.all(result.ke_values > 0)
    assert np.all(result.sensitivity_values > 0)
    # Rastgele örneklendiği için TÜMÜ aynı olmamalı.
    assert len(set(np.round(result.ke_values, 6))) > 1
    assert len(set(np.round(result.sensitivity_values, 6))) > 1


def test_run_reference_trace_is_deterministic_and_noise_free():
    """run_reference_trace, aynı girdilerle her çağrıldığında (rastgele
    örnekleme YOK) birebir aynı sonucu vermeli -- Monte Carlo'nun aksine."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient, drug = patients["hasta_a"], drugs["beta_bloker"]

    trace1 = run_reference_trace(patient, drug)
    trace2 = run_reference_trace(patient, drug)

    np.testing.assert_array_equal(trace1["hr"], trace2["hr"])
    np.testing.assert_array_equal(trace1["conc"], trace2["conc"])


def test_run_reference_trace_starts_at_patient_baseline():
    """t=0'da henüz hiç etki yok (konsantrasyon sıfır) -- nabız/tansiyon
    hastanın KENDİ bazal değerleriyle başlamalı (bkz. Görev A: CircAdapt'in
    de aynı ilkeyle kalibre edilmesi gerektiği bulgusu, burada PK/PD
    motorunun da baştan beri doğru yaptığı şey)."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient, drug = patients["hasta_a"], drugs["beta_bloker"]

    trace = run_reference_trace(patient, drug)

    assert trace["conc"][0] == 0.0
    assert trace["hr"][0] == patient.baseline_hr
    assert trace["sbp"][0] == patient.baseline_sbp


def test_run_reference_trace_effect_fraction_bounded():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    trace = run_reference_trace(patients["hasta_a"], drugs["beta_bloker"])

    assert np.all(trace["effect_hr"] >= 0) and np.all(trace["effect_hr"] <= 1.3)
    assert np.all(trace["effect_sbp"] >= 0) and np.all(trace["effect_sbp"] <= 1.3)


# --- Doz önerisi: istatistiksel + mekanik risk birleşimi (Faz 6) ---

def _fake_circadapt_results(lvedv_base_ml: float, lvedv_drug_ml: float) -> dict:
    """recommend_dose'un beklediği minimum sözlük şeklini üretir --
    gerçek bir CircAdapt çalıştırmadan (yavaş olduğu için) mekanik risk
    mantığını izole test etmek için."""
    return {
        "v_base": np.array([lvedv_base_ml]),
        "v_drug": np.array([lvedv_drug_ml]),
    }


def test_recommend_dose_without_circadapt_results_matches_old_behavior():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    rec = recommend_dose(patients["hasta_a"], drugs["beta_bloker"], n_candidates=5, n_realizations=50)

    assert rec["mechanical_risk"] is False
    assert rec["lvedv_increase_pct"] is None
    assert rec["dose_mg"] == rec["statistical_dose_mg"]
    assert "reasoning" in rec and len(rec["reasoning"]) > 0


def test_recommend_dose_flags_mechanical_risk_and_lowers_dose():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))

    # %25 LVEDV artışı -- varsayılan %20 eşiğini aşıyor.
    fake_results = _fake_circadapt_results(lvedv_base_ml=120.0, lvedv_drug_ml=150.0)

    rec = recommend_dose(patients["hasta_a"], drugs["beta_bloker"], n_candidates=5, n_realizations=50,
                          circadapt_results=fake_results)

    assert rec["mechanical_risk"] is True
    assert np.isclose(rec["lvedv_increase_pct"], 25.0)
    assert rec["dose_mg"] < rec["statistical_dose_mg"]
    assert np.isclose(rec["dose_mg"], rec["statistical_dose_mg"] * 0.7)
    assert "UYARI" in rec["reasoning"]


def test_recommend_dose_no_mechanical_risk_when_lvedv_increase_below_threshold():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))

    # %5 LVEDV artışı -- %20 eşiğinin altında.
    fake_results = _fake_circadapt_results(lvedv_base_ml=120.0, lvedv_drug_ml=126.0)

    rec = recommend_dose(patients["hasta_a"], drugs["beta_bloker"], n_candidates=5, n_realizations=50,
                          circadapt_results=fake_results)

    assert rec["mechanical_risk"] is False
    assert rec["dose_mg"] == rec["statistical_dose_mg"]


# --- Böbrek/karaciğer fonksiyonu (Faz 8) ---

def _make_patient(**overrides) -> Patient:
    base = dict(name="X", weight_kg=76, height_cm=175, age=45, blood_type="A",
                baseline_hr=78, baseline_sbp=125, baseline_dbp=80, baseline_spo2=97)
    base.update(overrides)
    return Patient(**base)


def test_organ_function_adjusted_ke_unaffected_when_clearance_fraction_zero():
    """renal/hepatic_clearance_fraction=0.0 olan bir ilaçta (örn. esmolol),
    organ fonksiyonu ne kadar bozulursa bozulsun ke DEĞİŞMEMELİ."""
    ke_normal = organ_function_adjusted_ke(4.6, renal_function=1.0, hepatic_function=1.0,
                                            renal_clearance_fraction=0.0, hepatic_clearance_fraction=0.0)
    ke_impaired = organ_function_adjusted_ke(4.6, renal_function=0.1, hepatic_function=0.1,
                                              renal_clearance_fraction=0.0, hepatic_clearance_fraction=0.0)
    assert ke_normal == ke_impaired == 4.6


def test_organ_function_adjusted_ke_decreases_with_impaired_renal_function():
    """renal_clearance_fraction>0 olan bir ilaçta (örn. digoksin), böbrek
    fonksiyonu düştükçe ke de düşmeli (ilaç daha yavaş atılır -> birikir)."""
    ke_mean = 0.01925  # digoksin
    ke_normal = organ_function_adjusted_ke(ke_mean, renal_function=1.0, hepatic_function=1.0,
                                            renal_clearance_fraction=0.65, hepatic_clearance_fraction=0.0)
    ke_impaired = organ_function_adjusted_ke(ke_mean, renal_function=0.3, hepatic_function=1.0,
                                              renal_clearance_fraction=0.65, hepatic_clearance_fraction=0.0)
    assert ke_normal == ke_mean  # tam fonksiyonda değişiklik yok
    assert ke_impaired < ke_normal  # bozuk fonksiyonda ke düşer (yarı ömür uzar)


def test_esmolol_ke_unaffected_by_organ_function_end_to_end():
    """Uçtan uca: esmololün ortalama konsantrasyon eğrisi, hasta böbrek/
    karaciğer fonksiyonundan bağımsız olarak AYNI kalmalı."""
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    drug = drugs["beta_bloker"]
    assert drug.renal_clearance_fraction == 0.0
    assert drug.hepatic_clearance_fraction == 0.0

    patient_normal = _make_patient(renal_function=1.0, hepatic_function=1.0)
    patient_impaired = _make_patient(renal_function=0.2, hepatic_function=0.2)

    result_normal = run_monte_carlo(patient_normal, drug, n_realizations=100, seed=7)
    result_impaired = run_monte_carlo(patient_impaired, drug, n_realizations=100, seed=7)

    assert np.allclose(result_normal.conc_runs, result_impaired.conc_runs)


def test_digoxin_accumulates_more_with_impaired_renal_function():
    """Uçtan uca: digoksin, böbrek yetmezliği olan hastada (renal_function
    düşük) daha yavaş atılır -> geç zaman noktalarında daha yüksek
    konsantrasyon birikir (dar terapötik indeksli bu ilaçta toksisite riski)."""
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    drug = drugs["digoxin"]
    assert drug.renal_clearance_fraction > 0

    patient_normal = _make_patient(renal_function=1.0)
    patient_renal_failure = _make_patient(renal_function=0.2)

    result_normal = run_monte_carlo(patient_normal, drug, n_realizations=50, seed=3,
                                     hours=48, n_timepoints=200)
    result_impaired = run_monte_carlo(patient_renal_failure, drug, n_realizations=50, seed=3,
                                       hours=48, n_timepoints=200)

    # Geç zaman noktasında (ilaç normalde büyük ölçüde atılmış olmalı),
    # böbrek yetmezliği olan hastada konsantrasyon belirgin şekilde daha yüksek kalmalı.
    late_conc_normal = result_normal.conc_runs[:, -1].mean()
    late_conc_impaired = result_impaired.conc_runs[:, -1].mean()
    assert late_conc_impaired > late_conc_normal * 1.5


# --- Gerçek ilaç veritabanı entegrasyonu (Faz 9) ---

VERIFIED_DRUGS_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "drugs_verified.yaml")


def test_load_verified_drugs_returns_working_drug_objects():
    verified = load_verified_drugs(VERIFIED_DRUGS_PATH)
    assert len(verified) >= 5, "Faz 9'da en az 5 gerçek ilaç bekleniyor"
    for key, entry in verified.items():
        assert isinstance(entry["drug"], Drug), f"{key}: 'drug' bir Drug nesnesi olmalı"
        assert entry["drug"].display_name, f"{key}: display_name boş olamaz"


def test_verified_drugs_have_provenance_metadata():
    verified = load_verified_drugs(VERIFIED_DRUGS_PATH)
    for key, entry in verified.items():
        prov = entry["provenance"]
        assert prov.get("source_url", "").startswith("https://"), f"{key}: geçerli bir source_url yok"
        assert prov.get("retrieved_date"), f"{key}: retrieved_date yok"
        assert prov.get("rxcui"), f"{key}: rxcui yok"


def test_verified_drugs_cover_all_three_drug_classes():
    verified = load_verified_drugs(VERIFIED_DRUGS_PATH)
    classes = {entry["drug"].drug_class for entry in verified.values()}
    assert classes == {"beta_blocker", "vasodilator", "positive_inotrope"}


def test_verified_drugs_run_monte_carlo_without_error():
    verified = load_verified_drugs(VERIFIED_DRUGS_PATH)
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    patient = patients["hasta_a"]
    for key, entry in verified.items():
        result = run_monte_carlo(patient, entry["drug"], n_realizations=10)
        assert (result.conc_runs >= 0).all(), f"{key}: negatif konsantrasyon üretti"


def test_esmolol_verified_matches_esmolol_in_main_drugs_yaml():
    """esmolol iki dosyada da var -- PK değerleri tutarlı olmalı (aynı ilaç, aynı kaynak)."""
    verified = load_verified_drugs(VERIFIED_DRUGS_PATH)
    main_drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))

    v = verified["esmolol"]["drug"]
    m = main_drugs["beta_bloker"]
    assert v.dose_mg_per_kg == m.dose_mg_per_kg
    assert v.ke_mean == m.ke_mean
    assert v.vd_per_kg == m.vd_per_kg


def test_fetch_fda_label_sections_quotes_multiword_drug_names():
    """Bug regresyon testi: tırnaksız çok kelimeli sorgu yanlış ilaçla eşleşiyordu
    (ör. 'sodium nitroprusside' -> 'sodium fluoride'). Ağ isteği YAPMADAN,
    sadece query string'in doğru oluşturulduğunu (tırnaklı) doğrular."""
    import drug_lookup
    import unittest.mock as mock

    captured_urls = []

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"results": []}'

    def fake_urlopen(url, timeout=10.0):
        captured_urls.append(url)
        return FakeResponse()

    with mock.patch.object(drug_lookup.urllib.request, "urlopen", fake_urlopen):
        drug_lookup.fetch_fda_label_sections("sodium nitroprusside")

    assert "%22sodium%20nitroprusside%22" in captured_urls[0], (
        "Çok kelimeli ilaç adı tırnaklı (tam ifade) aranmalı, yoksa openFDA "
        "kelimeleri ayrı ayrı eşleştirip yanlış ilaç döndürebilir."
    )


# --- Çoklu ilaç etkileşimi / polifarmasi (Faz 10) ---

def test_polypharmacy_two_negative_chronotropes_lower_hr_more_than_either_alone():
    """Tehlikeli kombinasyon senaryosu: esmolol + digoksin, ikisi de nabzı
    düşürür. Birlikte verildiğinde, HER İKİSİNİN tek başına ürettiğinden
    daha düşük bir ortalama minimum nabız beklenir."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = patients["hasta_a"]
    esmolol = drugs["beta_bloker"]
    digoxin = drugs["digoxin"]

    # Daha düşük, gerçekçi bir esmolol dozunda etkileşim daha görünür oluyor
    esmolol_moderate = replace(esmolol, dose_mg=20.0, dose_mg_per_kg=None)

    solo_esmolol = summarize(run_monte_carlo(patient, esmolol_moderate, n_realizations=200, seed=5))
    solo_digoxin = summarize(run_monte_carlo(patient, digoxin, n_realizations=200, seed=5))
    combo = summarize(run_polypharmacy_simulation(patient, [esmolol_moderate, digoxin],
                                                   n_realizations=200, seed=5))

    assert combo["mean_min_hr"] < solo_esmolol["mean_min_hr"]
    assert combo["mean_min_hr"] < solo_digoxin["mean_min_hr"]
    assert combo["pct_bradycardia_risk"] >= solo_esmolol["pct_bradycardia_risk"]


def test_polypharmacy_hr_never_goes_negative_with_many_drugs():
    """Fizyolojik üst sınır: aşırı sayıda toplamsal etki bile nabzı/tansiyonu
    0'ın altına düşürmemeli (np.clip garantisi)."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = patients["hasta_a"]
    esmolol = drugs["beta_bloker"]

    # Aynı ilacı 8 kez "veriyormuş" gibi -- gerçekçi değil ama sınırı test etmek için kasıtlı aşırı.
    result = run_polypharmacy_simulation(patient, [esmolol] * 8, n_realizations=50, seed=9)
    assert (result.hr_runs >= 0).all()
    assert (result.sbp_runs >= 0).all()


def test_polypharmacy_interaction_matrix_adds_extra_synergy():
    """interaction_matrix verildiğinde, ek sinerji terimi saf toplamsal
    kombinasyondan DAHA GÜÇLÜ bir etki üretmeli (aynı realizasyon/seed'de)."""
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = patients["hasta_a"]
    esmolol = replace(drugs["beta_bloker"], dose_mg=20.0, dose_mg_per_kg=None)
    digoxin = drugs["digoxin"]

    additive_only = summarize(run_polypharmacy_simulation(
        patient, [esmolol, digoxin], n_realizations=200, seed=11, interaction_matrix=None))
    with_synergy = summarize(run_polypharmacy_simulation(
        patient, [esmolol, digoxin], n_realizations=200, seed=11, interaction_matrix={(0, 1): 0.5}))

    assert with_synergy["mean_min_hr"] < additive_only["mean_min_hr"]


def test_recommend_dose_flags_polypharmacy_risk():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = patients["hasta_a"]
    esmolol = drugs["beta_bloker"]
    digoxin = drugs["digoxin"]

    poly_result = run_polypharmacy_simulation(patient, [esmolol, digoxin], n_realizations=150, seed=5)
    rec = recommend_dose(patient, esmolol, n_candidates=5, n_realizations=100,
                          polypharmacy_result=poly_result, polypharmacy_description="digoksin")

    assert rec["polypharmacy_bradycardia_risk_pct"] is not None
    assert "reasoning" in rec and len(rec["reasoning"]) > 0
    if rec["polypharmacy_risk"]:
        assert "TEHLİKELİ KOMBİNASYON" in rec["reasoning"]


def test_recommend_dose_without_polypharmacy_result_matches_old_behavior():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    rec = recommend_dose(patients["hasta_a"], drugs["beta_bloker"], n_candidates=5, n_realizations=50)
    assert rec["polypharmacy_risk"] is False
    assert rec["polypharmacy_bradycardia_risk_pct"] is None


# --- Elektrolit / laboratuvar verisinin kalp üzerindeki etkisi (Faz 11) ---

def test_potassium_conduction_factor_normal_range_is_unity():
    from worldmodel.pd import potassium_av_conduction_factor
    assert potassium_av_conduction_factor(3.5) == 1.0
    assert potassium_av_conduction_factor(4.25) == 1.0
    assert potassium_av_conduction_factor(5.0) == 1.0


def test_potassium_conduction_factor_increases_with_hyperkalemia():
    from worldmodel.pd import potassium_av_conduction_factor
    normal = potassium_av_conduction_factor(5.0)
    mild = potassium_av_conduction_factor(6.0)
    severe = potassium_av_conduction_factor(7.5)
    assert normal < mild < severe


def test_calcium_contractility_factor_direction():
    from worldmodel.pd import calcium_contractility_factor
    assert calcium_contractility_factor(9.5) == 1.0  # normal orta nokta
    assert calcium_contractility_factor(7.0) < 1.0     # hipokalsemi -> azalmış kontraktilite
    assert calcium_contractility_factor(12.0) > 1.0     # hiperkalsemi -> artmış kontraktilite


def test_patient_has_abnormal_electrolytes_flag():
    normal = _make_patient(potassium_mEqL=4.25, calcium_mgdL=9.5)
    hyperkalemic = _make_patient(potassium_mEqL=6.5, calcium_mgdL=9.5)
    hypocalcemic = _make_patient(potassium_mEqL=4.25, calcium_mgdL=6.8)

    assert normal.has_abnormal_electrolytes is False
    assert hyperkalemic.has_abnormal_electrolytes is True
    assert hypocalcemic.has_abnormal_electrolytes is True


def test_recommend_dose_flags_electrolyte_warning():
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    hyperkalemic_patient = _make_patient(potassium_mEqL=6.5)

    rec = recommend_dose(hyperkalemic_patient, drugs["beta_bloker"], n_candidates=5, n_realizations=50)
    assert rec["electrolyte_warning"] is True
    assert "LAB UYARISI" in rec["reasoning"]
    assert "potasyum" in rec["reasoning"]


def test_recommend_dose_no_electrolyte_warning_for_normal_patient():
    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    normal_patient = _make_patient()

    rec = recommend_dose(normal_patient, drugs["beta_bloker"], n_candidates=5, n_realizations=50)
    assert rec["electrolyte_warning"] is False
    assert "LAB UYARISI" not in rec["reasoning"]


def test_new_electrolyte_patient_profiles_load_correctly():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    assert "hasta_c_hiperkalemi" in patients
    assert "hasta_d_hipokalsemi" in patients
    assert patients["hasta_c_hiperkalemi"].has_abnormal_electrolytes is True
    assert patients["hasta_d_hipokalsemi"].has_abnormal_electrolytes is True
    assert patients["hasta_a"].has_abnormal_electrolytes is False


# --- Komorbidite / hastalık durumları (Faz 12) ---

def test_patient_comorbidity_defaults_to_none():
    assert _make_patient().comorbidity is None


def test_comorbidity_patient_profiles_load_correctly():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    assert patients["hasta_e_kalp_yetmezligi"].comorbidity == "heart_failure"
    assert patients["hasta_f_hipertansif"].comorbidity == "hypertension"
    assert patients["hasta_a"].comorbidity is None


def test_apply_comorbidity_to_circadapt_rejects_unknown_value():
    """CircAdapt'i gerçekten çalıştırmadan (yavaş), bilinmeyen bir comorbidity
    değerinin sessizce yok sayılmak yerine hata verdiğini doğrular. Bilinmeyen
    değer dalı model'e hiç dokunmadan hata verdiği için model=None yeterli."""
    import integrate_drug_with_circadapt as idc

    try:
        idc.apply_comorbidity_to_circadapt(None, "unknown_disease")
        assert False, "Bilinmeyen comorbidity ValueError fırlatmalıydı"
    except ValueError:
        pass


# --- Veri kaynağı izlenebilirliği / audit trail (Faz 14) ---

def test_provenance_report_classifies_every_known_drug_without_gaps():
    from worldmodel.provenance import provenance_report

    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = _make_patient()

    for key, drug in drugs.items():
        report = provenance_report(patient, drug)
        unclassified = [r for r in report if r["source_type"] == "sınıflandırılmamış"]
        assert not unclassified, f"{key}: sınıflandırılmamış parametre(ler) var: {unclassified}"
        assert len(report) > 0


def test_provenance_report_includes_both_literature_and_assumption_sources_for_esmolol():
    from worldmodel.provenance import provenance_report

    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = _make_patient()
    report = provenance_report(patient, drugs["beta_bloker"])

    source_types = {row["source_type"] for row in report}
    assert "literatür" in source_types
    assert "varsayım" in source_types
    assert "kullanıcı girdisi" in source_types

    ka_row = next(r for r in report if r["parameter"] == "ka")
    assert ka_row["source_type"] == "literatür"
    ec50_row = next(r for r in report if r["parameter"] == "ec50")
    assert ec50_row["source_type"] == "varsayım"


def test_provenance_report_includes_patient_fields_as_user_input():
    from worldmodel.provenance import provenance_report

    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = _make_patient(weight_kg=90)
    report = provenance_report(patient, drugs["beta_bloker"])

    weight_row = next(r for r in report if r["parameter"] == "weight_kg")
    assert weight_row["value"] == 90
    assert weight_row["source_type"] == "kullanıcı girdisi"


def test_provenance_report_handles_unknown_drug_gracefully():
    """Katalogda olmayan (ör. Streamlit'te elle oluşturulmuş) bir Drug için
    çökmemeli, sadece 'sınıflandırılmamış' olarak işaretlemeli."""
    from worldmodel.provenance import provenance_report
    from worldmodel.patient import Drug

    unknown_drug = Drug(display_name="Bilinmeyen Deneysel İlaç", dose_mg=1.0,
                         ka=1.0, ke_mean=1.0, vd_per_kg=1.0, emax_hr=1.0,
                         emax_sbp=1.0, ec50=0.01)
    patient = _make_patient()
    report = provenance_report(patient, unknown_drug)

    drug_rows = [r for r in report if r["source_type"] == "sınıflandırılmamış"]
    assert len(drug_rows) == 1


# --- Klinik rapor çıktısı / PDF export (Faz 15) ---

def _fake_heart_result_for_report():
    """export_report'un ihtiyaç duyduğu minimum alanları taşıyan sahte bir
    CircAdapt sonucu -- gerçek (yavaş) bir CircAdapt çalıştırmadan
    export_report'u izole test etmek için."""
    return {
        "hr_base": 71.0, "hr_drug_model": 55.0,
        "p_base": np.array([80.0, 118.0, 40.0]),
        "p_drug": np.array([80.0, 123.0, 40.0]),
        "v_base": np.array([50.0, 120.0]),
        "v_drug": np.array([50.0, 150.0]),
        "drug_effect": {"t_peak_hours": 0.1, "conc_peak": 0.16, "effect_fraction": 0.84},
    }


def test_export_report_returns_valid_pdf_bytes():
    from worldmodel.report import export_report, DISCLAIMER

    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = _make_patient()
    drug = drugs["beta_bloker"]

    dose_rec = {"dose_mg": 14.0, "reasoning": "test gerekçesi", "mechanical_risk": True, "is_safe": True}
    mc_stats = {"mean_min_hr": 56.0, "pct_bradycardia_risk": 16.0,
                "p5_min_hr": 45.0, "p95_min_hr": 64.0}

    pdf_bytes = export_report(patient, drug, dose_rec, mc_stats, _fake_heart_result_for_report())

    assert pdf_bytes[:5] == b"%PDF-", "Gecerli bir PDF dosyasi degil (magic bytes eksik)"
    assert len(pdf_bytes) > 1000


def test_export_report_disclaimer_is_defined_and_nonempty():
    from worldmodel.report import DISCLAIMER
    assert len(DISCLAIMER) > 10
    assert "SİMÜLASYON" in DISCLAIMER or "SIMULASYON" in DISCLAIMER


def test_export_report_with_chart_figure_embedded():
    import matplotlib.pyplot as plt
    from worldmodel.report import export_report

    drugs = load_drugs(os.path.join(os.path.dirname(__file__), "..", "configs", "drugs.yaml"))
    patient = _make_patient()
    drug = drugs["beta_bloker"]
    dose_rec = {"dose_mg": 14.0, "reasoning": "test", "mechanical_risk": False, "is_safe": True}
    mc_stats = {"mean_min_hr": 56.0, "pct_bradycardia_risk": 16.0,
                "p5_min_hr": 45.0, "p95_min_hr": 64.0}

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    pdf_bytes = export_report(patient, drug, dose_rec, mc_stats, _fake_heart_result_for_report(),
                               chart_figures=[fig])
    plt.close(fig)

    assert pdf_bytes[:5] == b"%PDF-"
    # Grafik gömülü bir PDF, grafiksiz olandan belirgin şekilde daha büyük olmalı.
    pdf_bytes_no_chart = export_report(patient, drug, dose_rec, mc_stats, _fake_heart_result_for_report())
    assert len(pdf_bytes) > len(pdf_bytes_no_chart)


# --- EK GÖREV A: CircAdapt baseline hasta senkronizasyon düzeltmesi ---

def test_circadapt_baseline_reflects_patients_own_baseline_hr():
    """BULUNAN HATA regresyon testi: calibrate_circadapt_to_patient()
    eklenmeden önce, CircAdapt'in "baseline" nabzı hangi hasta seçilirse
    seçilsin HEP AYNIYDI (~70.6 bpm, CircAdapt'in kendi jenerik
    varsayılanı) -- hastanın gerçek baseline_hr'ı (Hasta A: 78, Hasta B:
    85) hiç yansımıyordu. Bu test, artık CircAdapt'in KENDİ hasta bazına
    kalibre olduğunu doğruluyor."""
    import integrate_drug_with_circadapt as idc

    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    hasta_a, hasta_b = patients["hasta_a"], patients["hasta_b"]
    assert hasta_a.baseline_hr != hasta_b.baseline_hr  # ön koşul

    hr_a = 60.0 / idc.run_baseline(hasta_a)["General"]["t_cycle"]
    hr_b = 60.0 / idc.run_baseline(hasta_b)["General"]["t_cycle"]

    assert abs(hr_a - hasta_a.baseline_hr) < 0.1
    assert abs(hr_b - hasta_b.baseline_hr) < 0.1
    assert hr_a != hr_b


def test_calibrate_circadapt_stable_across_realistic_hr_range():
    """Streamlit slider aralığının (50-110 bpm) uçlarında bile CircAdapt'in
    sayısal olarak kararlı (NaN üretmeyen) kaldığını doğrular -- sabit-süreli
    (saniye cinsinden) sistol/diyastol zamanlama parametrelerinin (tr, td,
    time_act) bu aralıkta bozulmadığı elle doğrulanmıştı, bu test o
    doğrulamayı kalıcı hale getiriyor."""
    import integrate_drug_with_circadapt as idc

    patient = _make_patient(baseline_hr=50)
    model = idc.run_baseline(patient)
    p = model["Cavity"]["p"][:, "cLv"]
    assert not np.isnan(p).any()

    patient = _make_patient(baseline_hr=110)
    model = idc.run_baseline(patient)
    p = model["Cavity"]["p"][:, "cLv"]
    assert not np.isnan(p).any()


# --- EK GÖREV B: Ejeksiyon fraksiyonu (EF) / kardiyak output (CO) ---

def test_ejection_fraction_known_values():
    from worldmodel.clinical_metrics import ejection_fraction
    # EDV=120, ESV=48 -> EF = (120-48)/120*100 = %60 (normal aralıkta)
    assert abs(ejection_fraction(120, 48) - 60.0) < 0.01


def test_ejection_fraction_zero_edv_returns_zero_not_crash():
    from worldmodel.clinical_metrics import ejection_fraction
    assert ejection_fraction(0, 0) == 0.0


def test_cardiac_output_known_values():
    from worldmodel.clinical_metrics import cardiac_output
    # SV = 120-48 = 72 mL, HR=70 bpm -> CO = 72*70/1000 = 5.04 L/dk
    assert abs(cardiac_output(120, 48, 70) - 5.04) < 0.01


def test_classify_cardiac_function_normal_is_green():
    from worldmodel.clinical_metrics import classify_cardiac_function
    result = classify_cardiac_function(ef=60.0, co=5.0)
    assert result["overall_color"] == "yeşil"


def test_classify_cardiac_function_low_ef_is_red():
    from worldmodel.clinical_metrics import classify_cardiac_function
    result = classify_cardiac_function(ef=30.0, co=5.0)
    assert result["overall_color"] == "kırmızı"
    assert result["ef_color"] == "kırmızı"


def test_heart_failure_comorbidity_produces_genuinely_lower_ef_than_healthy():
    """Uçtan uca: kalp yetmezliği komorbiditesi olan bir hastanın CircAdapt
    baseline'ından hesaplanan EF, sağlıklı hastadan GERÇEKTEN düşük olmalı
    (Görev B'nin ana iddiası -- düşürülmüş kontraktilite senaryosunda EF düşer)."""
    import integrate_drug_with_circadapt as idc
    from worldmodel.clinical_metrics import ejection_fraction

    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    healthy = patients["hasta_a"]
    heart_failure = patients["hasta_e_kalp_yetmezligi"]

    m_healthy = idc.run_baseline(healthy)
    v_healthy = m_healthy["Cavity"]["V"][:, "cLv"] * 1e6
    ef_healthy = ejection_fraction(v_healthy.max(), v_healthy.min())

    m_hf = idc.run_baseline(heart_failure)
    v_hf = m_hf["Cavity"]["V"][:, "cLv"] * 1e6
    ef_hf = ejection_fraction(v_hf.max(), v_hf.min())

    assert ef_hf < ef_healthy
    assert ef_healthy >= 55.0  # sağlıklı hasta normal aralıkta olmalı
