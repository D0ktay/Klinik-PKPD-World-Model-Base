"""
ADIM 3.7 -- N-ilaç (N=1..8) istatistiksel motor doğruluk testleri.

N_DRUG_AUDIT.md / RESEARCH_N_DRUG.md'de karara bağlanan ADR-1..6'nın
UYGULAMASINI doğrular -- ADIM 2'nin golden-snapshot testlerinden (N=1/2
SAYISAL sabitliği) FARKLI amaç: burada N=1..8 arası matematiksel
ÖZELLİKLER (permütasyon değişmezliği, monotonluk, fizyolojik sınırlar,
Loewe iç tutarlılığı) test ediliyor.
"""
import itertools
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from worldmodel.patient import load_drugs, load_patients, Drug
from worldmodel.pd import loewe_combined_effect, grouped_loewe_combined_effect, mechanistic_fraction_combined_effect
from worldmodel.simulation import (
    run_monte_carlo, run_polypharmacy_simulation, run_polypharmacy_simulation_loewe,
)
import integrate_drug_with_circadapt as idc

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture(scope="module")
def hasta_a():
    return load_patients(os.path.join(CONFIG_DIR, "patients.yaml"))["hasta_a"]


@pytest.fixture(scope="module")
def template_drug():
    return load_drugs(os.path.join(CONFIG_DIR, "drugs.yaml"))["beta_bloker"]


def make_synthetic_drugs(template: Drug, n: int, base_ec50: float = 0.5, direction: float = 1.0) -> list[Drug]:
    """
    N farklı (aynı yönlü, AV-duyarsız -- "vasodilator" sınıfı) sentetik
    ilaç üretir; her biri farklı bir EC50/Emax ile (dejenere/özdeş eğri
    çakışmasından kaçınmak için). drug_class="vasodilator" bilinçli seçim
    -- bu testler saf N-ilaç MATEMATİĞİNİ (Loewe/additive/permütasyon)
    izole etmek istiyor, AV-blok mekanizmasını (ayrı test dosyalarında
    zaten kapsanan) KARIŞTIRMADAN.
    """
    drugs = []
    for i in range(n):
        drugs.append(replace(
            template,
            display_name=f"Sentetik-{i}",
            drug_class="vasodilator",
            dose_mg=10.0 + i, dose_mg_per_kg=None,
            ec50=base_ec50 * (1.0 + 0.15 * i),
            emax_hr=direction * (10.0 + 2.0 * i),
            emax_sbp=direction * (8.0 + 1.5 * i),
            keo_hr=None, keo_sbp=None,
            renal_clearance_fraction=0.0, hepatic_clearance_fraction=0.0,
        ))
    return drugs


ZERO_VARIANCE_KWARGS = dict(
    n_realizations=5, hours=4.0, n_timepoints=20,
    ke_variation_sigma=0.0, sensitivity_variation_sigma=0.0,
    measurement_noise_hr=0.0, measurement_noise_sbp=0.0, seed=7,
)


# --- N=1 tutarlılığı -------------------------------------------------------

def test_n1_polypharmacy_matches_monte_carlo(hasta_a, template_drug):
    """run_polypharmacy_simulation([d]) ≡ run_monte_carlo(d) -- aynı seed,
    aynı rng çağrı SIRASI (sensitivity sonra ke, sonra ölçüm gürültüsü)
    korunduğu sürece N=1'de iki fonksiyon SAYISAL OLARAK örtüşmeli.
    np.clip(hr,0,None) run_polypharmacy_simulation'da var, run_monte_carlo'da
    yok -- bu yüzden hr'nin 0'a yaklaşmadığı normal-doz bir senaryo seçildi."""
    kwargs = dict(n_realizations=10, hours=6.0, n_timepoints=30, seed=11)
    mc = run_monte_carlo(hasta_a, template_drug, **kwargs)
    poly = run_polypharmacy_simulation(hasta_a, [template_drug], **kwargs)

    np.testing.assert_allclose(poly.hr_runs, mc.hr_runs, rtol=0, atol=0)
    np.testing.assert_allclose(poly.sbp_runs, mc.sbp_runs, rtol=0, atol=0)
    np.testing.assert_allclose(poly.conc_runs, mc.conc_runs, rtol=0, atol=0)


# --- Permütasyon değişmezliği (N=2..6) -------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_additive_permutation_invariance_zero_variance(hasta_a, template_drug, n):
    """
    Sıfır-varyanslı (RNG'nin ke/sensitivity/gürültü üzerindeki etkisi
    devre dışı) kwargs ile: toplamsal (additive) motorun sonucu ilaç
    LİSTESİNİN SIRASINDAN bağımsız olmalı -- toplama commutative.

    NEDEN sıfır-varyans gerekli (dürüstlük notu): run_polypharmacy_simulation
    ilaçları SIRAYLA işleyip HER ilaç için AYNI paylaşılan rng akışından
    (sensitivity sonra ke) çekiyor -- yani sigma>0 iken pozisyon DEĞİŞTİĞİNDE
    hangi ilacın hangi rastgele örneği aldığı da değişir, bu da bit-exact
    permütasyon değişmezliğini BOZAR (istatistiksel olarak hâlâ aynı
    DAĞILIM ama aynı DİZİ değil). Bu, CircAdapt tarafının (rng'siz,
    deterministik) permütasyon değişmezliğinden YAPISAL OLARAK farklı bir
    durum -- bkz. N_DRUG_AUDIT.md Şüphe G. sigma=0 iken bu kaynak ortadan
    kalkar (rng.lognormal(sigma=0) her zaman 1.0, rng.normal(sigma=0) her
    zaman 0.0), toplama gerçekten commutative hale gelir.
    """
    drugs = make_synthetic_drugs(template_drug, n)
    reference = run_polypharmacy_simulation(hasta_a, drugs, **ZERO_VARIANCE_KWARGS)

    for perm in list(itertools.permutations(range(n)))[:6]:
        permuted = [drugs[i] for i in perm]
        result = run_polypharmacy_simulation(hasta_a, permuted, **ZERO_VARIANCE_KWARGS)
        np.testing.assert_allclose(result.hr_runs, reference.hr_runs, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result.sbp_runs, reference.sbp_runs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_loewe_permutation_invariance_zero_variance(hasta_a, template_drug, n):
    """grouped_loewe_combined_effect()'in altında yatan bisection, ilaç
    listeleri üzerinden SIRADAN BAĞIMSIZ bir toplam kullanıyor (bkz.
    pd.py > loewe_combined_effect) -- sıfır-varyans kwargs ile Loewe
    yolu da permütasyon-değişmez olmalı."""
    drugs = make_synthetic_drugs(template_drug, n)
    reference = run_polypharmacy_simulation_loewe(hasta_a, drugs, **ZERO_VARIANCE_KWARGS)

    for perm in list(itertools.permutations(range(n)))[:6]:
        permuted = [drugs[i] for i in perm]
        result = run_polypharmacy_simulation_loewe(hasta_a, permuted, **ZERO_VARIANCE_KWARGS)
        np.testing.assert_allclose(result.hr_runs, reference.hr_runs, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(result.sbp_runs, reference.sbp_runs, rtol=1e-9, atol=1e-9)


# --- Monotonluk (N=1..8) ----------------------------------------------------

@pytest.mark.parametrize("n", range(1, 8))
def test_adding_same_direction_drug_increases_combined_hr_drop(hasta_a, template_drug, n):
    """Aynı sınıftan (aynı yönde) bir ilaç daha eklemek, birleşik HR
    düşüşünü ARTIRMALI (additive VE Loewe yollarının ikisinde de) --
    N=1'den N=8'e kadar."""
    drugs_n = make_synthetic_drugs(template_drug, n)
    drugs_n_plus_1 = make_synthetic_drugs(template_drug, n + 1)

    additive_n = run_polypharmacy_simulation(hasta_a, drugs_n, **ZERO_VARIANCE_KWARGS)
    additive_n_plus_1 = run_polypharmacy_simulation(hasta_a, drugs_n_plus_1, **ZERO_VARIANCE_KWARGS)
    assert additive_n_plus_1.hr_runs.mean() <= additive_n.hr_runs.mean() + 1e-9

    loewe_n = run_polypharmacy_simulation_loewe(hasta_a, drugs_n, **ZERO_VARIANCE_KWARGS)
    loewe_n_plus_1 = run_polypharmacy_simulation_loewe(hasta_a, drugs_n_plus_1, **ZERO_VARIANCE_KWARGS)
    assert loewe_n_plus_1.hr_runs.mean() <= loewe_n.hr_runs.mean() + 1e-9


# --- Fizyolojik sınırlar (N=8) ----------------------------------------------

def test_n8_additive_stays_physiologically_bounded(hasta_a, template_drug):
    drugs = make_synthetic_drugs(template_drug, 8)
    result = run_polypharmacy_simulation(hasta_a, drugs, n_realizations=8, hours=6.0, n_timepoints=25, seed=3)
    assert np.isfinite(result.hr_runs).all()
    assert np.isfinite(result.sbp_runs).all()
    assert (result.hr_runs >= 0).all()
    assert (result.sbp_runs >= 0).all()


def test_n8_loewe_stays_physiologically_bounded(hasta_a, template_drug):
    drugs = make_synthetic_drugs(template_drug, 8)
    result = run_polypharmacy_simulation_loewe(hasta_a, drugs, n_realizations=8, hours=6.0, n_timepoints=25, seed=3)
    assert np.isfinite(result.hr_runs).all()
    assert np.isfinite(result.sbp_runs).all()
    assert (result.hr_runs >= 0).all()
    assert (result.sbp_runs >= 0).all()


# --- Sınır durumlar ----------------------------------------------------------

def test_same_drug_selected_twice_does_not_crash_and_stays_bounded(hasta_a, template_drug):
    drugs = [template_drug, template_drug]
    result = run_polypharmacy_simulation(hasta_a, drugs, **ZERO_VARIANCE_KWARGS)
    assert np.isfinite(result.hr_runs).all()
    assert (result.hr_runs >= 0).all()
    # iki katına çıkan doz -- HR düşüşü TEK ilaçtan büyük olmalı
    solo = run_polypharmacy_simulation(hasta_a, [template_drug], **ZERO_VARIANCE_KWARGS)
    assert result.hr_runs.mean() <= solo.hr_runs.mean() + 1e-9


def test_zero_dose_drug_does_not_change_result(hasta_a, template_drug):
    """dose_mg=0 olan bir ilaç eklemek sonucu DEĞİŞTİRMEMELİ -- etkisiz
    (konsantrasyon her zaman 0) bir ilacın toplam etkiye katkısı sıfır
    olmalı."""
    real_drug = template_drug
    zero_dose_drug = replace(template_drug, display_name="Sıfır Doz", dose_mg=0.0, dose_mg_per_kg=None)

    solo = run_polypharmacy_simulation(hasta_a, [real_drug], **ZERO_VARIANCE_KWARGS)
    with_zero = run_polypharmacy_simulation(hasta_a, [real_drug, zero_dose_drug], **ZERO_VARIANCE_KWARGS)

    np.testing.assert_allclose(with_zero.hr_runs, solo.hr_runs, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(with_zero.sbp_runs, solo.sbp_runs, rtol=1e-9, atol=1e-9)


def test_single_drug_plus_n_minus_1_zero_dose_equals_single_drug(hasta_a, template_drug):
    real_drug = template_drug
    zero_dose_drugs = [
        replace(template_drug, display_name=f"Sıfır-{i}", dose_mg=0.0, dose_mg_per_kg=None)
        for i in range(5)
    ]
    solo = run_polypharmacy_simulation(hasta_a, [real_drug], **ZERO_VARIANCE_KWARGS)
    combo = run_polypharmacy_simulation(hasta_a, [real_drug] + zero_dose_drugs, **ZERO_VARIANCE_KWARGS)
    np.testing.assert_allclose(combo.hr_runs, solo.hr_runs, rtol=1e-9, atol=1e-9)


# --- Loewe iç tutarlılığı (izobol denklemi residual'i) ----------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_loewe_isobole_equation_residual_small(template_drug, n):
    """
    loewe_combined_effect()'in bulduğu birleşik etki `e`, tanım gereği
    Σ_i C_i*(emax_i-e)/(ec50_i*e) = 1 denklemini sağlamalı -- bisection'ın
    (n_iterations=40, varsayılan) N=8'e kadar hâlâ yeterince yakınsadığını
    DOĞRUDAN ölçer (varsaymaz).

    ÖNEMLİ KALİBRASYON NOTU (ölçerek bulundu): konsantrasyonlar EC50'lere
    göre yüksek seçilirse (örn. ilk denemede conc~0.3-1.0, ec50~0.4-0.75),
    N≥4'te birleşik etki `min(emax_i)` TAVANINA doğru saturasyona uğruyor
    (bkz. pd.py BİLİNEN KAPSAM SINIRI / ADR-1) -- bu durumda `e` gerçek
    izobol denklemini SAĞLAMAZ (denklemin [0, tavan) aralığında kökü YOK,
    bisection tavana kenetleniyor) -- bu bir bisection HATASI değil, tavan
    kısıtının matematiksel SONUCU (ayrıca bkz.
    test_loewe_saturates_at_ceiling_for_strong_combined_effect). Bu test
    kasıtlı olarak TAVANDAN UZAK (düşük conc/ec50 oranı) parametreler
    kullanıyor ki asıl ölçülen şey (bisection yakınsaması) tavan-kenetlenme
    davranışıyla KARIŞMASIN.
    """
    concentrations = [np.full(5, 0.03 + 0.005 * i) for i in range(n)]
    ec50s = [0.4 + 0.05 * i for i in range(n)]
    emaxes = [15.0 + 3.0 * i for i in range(n)]

    e = loewe_combined_effect(concentrations, ec50s, emaxes, n_iterations=40)

    residual = sum(
        conc * (emax - e) / (ec50 * np.maximum(e, 1e-12))
        for conc, ec50, emax in zip(concentrations, ec50s, emaxes)
    ) - 1.0
    assert np.max(np.abs(residual)) < 1e-6, f"N={n}: bisection residual çok büyük ({np.max(np.abs(residual))})"


def test_loewe_saturates_at_ceiling_for_strong_combined_effect():
    """
    N_DRUG_AUDIT.md Şüphe C / RESEARCH_N_DRUG.md ADR-1,5'in öngördüğü
    davranışı SAYISAL OLARAK doğrular: konsantrasyonlar EC50'lere göre
    güçlüyse (yüksek doz/etki), N büyüdükçe birleşik Loewe etkisi
    min(emax_i) tavanına DAHA HIZLI ulaşır -- N=2'de tavana uzak, N=4+'ta
    pratik olarak tavanda (bisection üst sınırı hi=min(emax)*(1-1e-6)'ya
    kenetlenir). Bu, ADR-5'in "N büyüdükçe tavan daha sık bağlayıcı hale
    gelir" kararının ölçülmüş kanıtı."""
    results = {}
    for n in (2, 3, 4, 6, 8):
        concentrations = [np.full(5, 0.3 + 0.1 * i) for i in range(n)]
        ec50s = [0.4 + 0.05 * i for i in range(n)]
        emaxes = [15.0 + 3.0 * i for i in range(n)]
        e = loewe_combined_effect(concentrations, ec50s, emaxes, n_iterations=40)
        results[n] = float(e[0])

    min_emax = 15.0
    # N arttıkça tavana yaklaşma monoton olmalı
    assert results[2] < results[3] < results[4]
    # N=4 ve sonrası pratik olarak tavanda (1e-3 bpm toleransla)
    for n in (4, 6, 8):
        assert results[n] == pytest.approx(min_emax, abs=1e-2)


# --- Property-based test (hypothesis) ---------------------------------------

@given(
    n=st.integers(min_value=1, max_value=8),
    emax_base=st.floats(min_value=5.0, max_value=40.0),
    ec50_base=st.floats(min_value=0.05, max_value=2.0),
    conc_level=st.floats(min_value=0.01, max_value=5.0),
)
@settings(max_examples=40, deadline=None)
def test_loewe_combined_effect_stays_bounded_and_finite(n, emax_base, ec50_base, conc_level):
    """Rastgele N (1-8), Emax, EC50, konsantrasyon için: loewe_combined_effect
    her zaman sonlu ve |sonuç| <= min(emax_i) olmalı (bkz. pd.py > BİLİNEN
    KAPSAM SINIRI) -- ADR-1'in "tavan kaldırılmıyor" kararının, rastgele
    girdiler altında GERÇEKTEN doğru olduğunu doğrular."""
    concentrations = [np.array([conc_level * (1.0 + 0.1 * i)]) for i in range(n)]
    ec50s = [ec50_base * (1.0 + 0.1 * i) for i in range(n)]
    emaxes = [emax_base + 2.0 * i for i in range(n)]

    result = loewe_combined_effect(concentrations, ec50s, emaxes)

    assert np.isfinite(result).all()
    assert np.all(np.abs(result) <= min(emaxes) + 1e-6)
    assert np.all(result >= -1e-9)  # tüm emax'lar pozitif -> sonuç negatif olmamalı


@given(
    n_positive=st.integers(min_value=0, max_value=4),
    n_negative=st.integers(min_value=0, max_value=4),
)
@settings(max_examples=25, deadline=None)
def test_grouped_loewe_handles_arbitrary_sign_mix(n_positive, n_negative):
    """grouped_loewe_combined_effect (ADR-4/ADR-7), rastgele sayıda pozitif/
    negatif Emax'lı ilaç karışımında HİÇBİR ZAMAN çökmemeli (ValueError/
    exception), her zaman sonlu bir sonuç döndürmeli. Karma yönlü girdide
    ARTIK mechanistic_fraction_combined_effect() (ADR-7) kullanıldığını da
    doğrular -- eski "gruplama+fark" (ADR-4) yöntemine sessizce dönülmediğini
    kanıtlamak için sonuçlar BİREBİR eşleştiriliyor."""
    if n_positive + n_negative == 0:
        return
    baseline = 80.0
    concentrations, ec50s, emaxes = [], [], []
    for i in range(n_positive):
        concentrations.append(np.array([0.5]))
        ec50s.append(0.3 + 0.05 * i)
        emaxes.append(10.0 + i)
    for i in range(n_negative):
        concentrations.append(np.array([0.5]))
        ec50s.append(0.3 + 0.05 * i)
        emaxes.append(-(10.0 + i))

    result = grouped_loewe_combined_effect(concentrations, ec50s, emaxes, baseline)
    assert np.isfinite(result).all()

    if n_positive > 0 and n_negative > 0:
        expected = mechanistic_fraction_combined_effect(concentrations, ec50s, emaxes, baseline)
        assert np.allclose(result, expected)


def test_mechanistic_fraction_combined_effect_matches_circadapt_t_cycle(hasta_a, template_drug):
    """ADR-7'nin asıl iddiası: istatistiksel motorun karma-yönlü HR
    birleştirmesi, CircAdapt tarafının kanonik `t_cycle` formülüyle
    (integrate_drug_with_circadapt.cumulative_parameter_multipliers)
    SAYISAL OLARAK tutarlı -- iki motor artık gerçekten aynı matematiği
    paylaşıyor (AV-blok düzeltmesindeki (ADR-3) doğrulama deseniyle aynı)."""
    drugs = [
        replace(template_drug, display_name="Azaltan", drug_class="vasodilator",
                dose_mg=10.0, dose_mg_per_kg=None, ec50=0.5, emax_hr=15.0,
                keo_hr=None, keo_sbp=None, renal_clearance_fraction=0.0, hepatic_clearance_fraction=0.0),
        replace(template_drug, display_name="Artiran", drug_class="vasodilator",
                dose_mg=12.0, dose_mg_per_kg=None, ec50=0.6, emax_hr=-8.0,
                keo_hr=None, keo_sbp=None, renal_clearance_fraction=0.0, hepatic_clearance_fraction=0.0),
    ]

    drug_effects = [idc.compute_drug_effect(hasta_a, d) for d in drugs]
    multipliers = idc.cumulative_parameter_multipliers(hasta_a, drugs, drug_effects)
    circadapt_hr = hasta_a.baseline_hr / multipliers["t_cycle"]

    concentrations = [np.array([e["conc_peak"]]) for e in drug_effects]
    ec50s = [d.ec50 for d in drugs]
    emaxes = [d.emax_hr for d in drugs]
    hr_delta = mechanistic_fraction_combined_effect(concentrations, ec50s, emaxes, hasta_a.baseline_hr)
    statistical_hr = hasta_a.baseline_hr - hr_delta

    assert np.allclose(statistical_hr, circadapt_hr, rtol=1e-9)
