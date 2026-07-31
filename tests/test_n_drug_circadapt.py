"""
ADIM 4.3 -- N-ilaç (N=1..6) CircAdapt sayısal kararlılık testleri.

N_DRUG_AUDIT.md Şüphe F/G'nin ve ADIM 4.1/4.2'nin (parametre-başına
ölçülmüş çöküş eşiği + genelleştirilmiş ön-kontrol) uygulamasını doğrular.
CircAdapt gerçekten çalıştırıldığı için (`@pytest.mark.slow`) bu dosya
yavaş -- ama CALIBRATION_REPORT.md ve görev talimatı gereği CI'da yine
çalıştırılmalı, atlanmamalı.
"""
import itertools
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from worldmodel.patient import load_drugs, load_patients, Drug
import integrate_drug_with_circadapt as idc

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def hasta_a():
    return load_patients(os.path.join(CONFIG_DIR, "patients.yaml"))["hasta_a"]


@pytest.fixture(scope="module")
def drug_pool():
    drugs = load_drugs(os.path.join(CONFIG_DIR, "drugs.yaml"))
    return [
        replace(drugs["beta_bloker"], dose_mg=15.0, dose_mg_per_kg=None),
        drugs["vazodilator"],
        replace(drugs["dobutamine"], dose_mg=5.0, dose_mg_per_kg=None),
        replace(drugs["digoxin"], dose_mg=0.25, dose_mg_per_kg=None),
    ]


# --- Sıra bağımsızlığı (Şüphe G) -- gerçek çalışma-zamanı doğrulaması ------

def test_run_with_multiple_drugs_permutation_invariance(hasta_a, drug_pool):
    """
    N_DRUG_AUDIT.md Şüphe G: run_with_multiple_drugs() ilaçları sırayla
    uyguluyor -- bu test 4 ilacın 24 permütasyonundan bir örneklemini
    (hız için 8 tanesi) GERÇEK CircAdapt ile çalıştırıp General.t_cycle,
    Patch.Sf_act, ArtVen.p0, Timings.c_tau_av1'in sıradan BAĞIMSIZ
    olduğunu doğrudan ölçer (statik kod incelemesiyle değil).
    """
    drug_effects = [idc.compute_drug_effect(hasta_a, d) for d in drug_pool]

    perms = list(itertools.permutations(range(4)))[:8]
    reference = None
    for perm in perms:
        perm_drugs = [drug_pool[i] for i in perm]
        perm_effects = [drug_effects[i] for i in perm]
        model = idc.run_with_multiple_drugs(hasta_a, perm_drugs, perm_effects)
        snapshot = (
            float(model["General"]["t_cycle"]),
            float(np.asarray(model["Patch"]["Sf_act"]).flatten()[0]),
            float(np.asarray(model["ArtVen"]["p0"]).flatten()[0]),
            float(np.asarray(model["Timings"]["c_tau_av1"]).flatten()[0]),
        )
        if reference is None:
            reference = snapshot
        else:
            for actual, expected in zip(snapshot, reference):
                assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


# --- N=1..6: çökme yok, ya gerçek sonuç ya temiz güvenli dönüş -------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_run_polypharmacy_comparison_never_raises_uncaught_for_mild_doses(hasta_a, drug_pool, n):
    """Ilımlı dozlu N=1..6 kombinasyonu -- CircAdaptException/ModelCrashed
    ASLA çağırana sızmamalı: ya gerçek bir hemodinamik sonuç, ya da
    instability_triggered=True ile temiz bir güvenli dönüş olmalı."""
    drugs = [drug_pool[i % len(drug_pool)] for i in range(n)]
    result = idc.run_polypharmacy_comparison(hasta_a, drugs)

    assert "instability_triggered" in result
    assert "unstable_parameter" in result
    assert "parameter_multipliers" in result

    if result["instability_triggered"] or result["av_block_triggered"]:
        assert np.all(np.isnan(result["p_drug"]))
        assert np.all(np.isnan(result["v_drug"]))
        assert np.isfinite(result["hr_drug_model"])
    else:
        assert np.isfinite(result["p_drug"]).all()
        assert np.isfinite(result["v_drug"]).all()
        assert np.isfinite(result["hr_drug_model"])


# --- Parametre-başına ön-kontrol mantığı (CircAdapt'siz, saf fonksiyon) ---

def test_cumulative_parameter_multipliers_no_drugs_returns_baseline_electrolyte_only(hasta_a):
    multipliers = idc.cumulative_parameter_multipliers(hasta_a, [], [])
    assert multipliers["t_cycle"] == pytest.approx(1.0)
    assert multipliers["Sf_act"] == pytest.approx(1.0)
    assert multipliers["ArtVen.p0"] == pytest.approx(1.0)
    assert multipliers["c_tau_av1"] == pytest.approx(1.0)  # hasta_a normal K+


def test_circadapt_instability_risk_flags_t_cycle_breach():
    multipliers = {"t_cycle": 3.0, "Sf_act": 1.0, "ArtVen.p0": 1.0, "c_tau_av1": 1.0}
    assert idc.circadapt_instability_risk(multipliers) == "t_cycle"


def test_circadapt_instability_risk_flags_sf_act_breach():
    multipliers = {"t_cycle": 1.0, "Sf_act": 50.0, "ArtVen.p0": 1.0, "c_tau_av1": 1.0}
    assert idc.circadapt_instability_risk(multipliers) == "Sf_act"


def test_circadapt_instability_risk_flags_artven_p0_breach():
    multipliers = {"t_cycle": 1.0, "Sf_act": 1.0, "ArtVen.p0": 1000.0, "c_tau_av1": 1.0}
    assert idc.circadapt_instability_risk(multipliers) == "ArtVen.p0"


def test_circadapt_instability_risk_prioritizes_c_tau_av1(idc_module=idc):
    """c_tau_av1 (AV blok, klinik olarak en anlamlı yorum) diğerlerinden
    ÖNCE kontrol edilmeli -- birden fazla parametre aynı anda eşiği
    aşarsa, dönüş 'c_tau_av1' olmalı."""
    multipliers = {"t_cycle": 10.0, "Sf_act": 1.0, "ArtVen.p0": 1.0, "c_tau_av1": 3.0}
    assert idc.circadapt_instability_risk(multipliers) == "c_tau_av1"


def test_circadapt_instability_risk_none_when_all_safe():
    multipliers = {"t_cycle": 1.2, "Sf_act": 0.8, "ArtVen.p0": 1.5, "c_tau_av1": 1.1}
    assert idc.circadapt_instability_risk(multipliers) is None


def test_t_cycle_breach_triggers_clean_return_without_calling_circadapt_run(hasta_a, drug_pool, monkeypatch):
    """t_cycle esigini asan SENTETIK bir senaryoda (asiri guclu negatif
    kronotrop dozu), run_with_multiple_drugs() HIC CAGRILMAMALI -- on-kontrol
    modele hic dokunmadan devreye girmeli (AV blok deseninin t_cycle'a
    genellenmis hali)."""
    called = {"n": 0}
    original = idc.run_with_multiple_drugs

    def spy(*args, **kwargs):
        called["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(idc, "run_with_multiple_drugs", spy)

    # SENTETIK asiri guclu negatif kronotrop -- emax_hr'yi bazal nabza
    # yakin buyuklukte ayarlayip (75 vs baseline_hr=78) yuksek doz/dusuk
    # EC50 ile pik etkiye ulastirarak hr_fraction'i kasitli olarak cok
    # kucultuyoruz (t_cycle_multiplier = 1/hr_fraction esigi -- 2.5 -- acikca asiyor).
    extreme_beta_bloker = replace(
        drug_pool[0], dose_mg=500.0, dose_mg_per_kg=None, ec50=0.01, emax_hr=75.0,
    )
    result = idc.run_polypharmacy_comparison(hasta_a, [extreme_beta_bloker])

    assert called["n"] == 0, "esik asildiginda CircAdapt hic calistirilmamali"
    assert result["instability_triggered"] or result["av_block_triggered"]
