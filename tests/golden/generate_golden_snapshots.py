"""
ADIM 2 -- Altin anlik goruntu (golden snapshot) uretim betigi.

Bu betik pytest tarafindan TOPLANMAZ (dosya adi test_ ile baslamiyor) --
elle, SADECE davranis degisikligi BEKLENMEDIGINDE calistirilir. Amaci,
N-ilac genellemesi calismasina (ADIM 3+) baslamadan ONCE, N=1 ve N=2 icin
BUGUNKU (referans) ciktiyi dondurmak -- boylece tests/test_no_regression_
n_drug.py, sonraki her degisiklikten sonra "hala AYNI mi" diye
karsilastirabilsin.

Kapsam (ADIM 2 talimati + ek 2 gereksinim):
  - N=1 ve N=2 istatistiksel motor ciktilari (additive + Loewe)
  - N=1 ve N=2 CircAdapt ciktilari (run_comparison / run_polypharmacy_comparison)
  - Doz onerisi (recommend_dose, recommend_polypharmacy_dose_scale)
  - AV-duyarli ilac + anormal potasyum (hasta_c_hiperkalemi) N=1 VE N=2 --
    formul hizalamasinin (ADIM 3.5, Supphe E fix) en riskli oldugu nokta.
  - Zit yonlu kombinasyon (beta_bloker emax_hr=+25 + nicardipine emax_hr=-6):
    additive yolun basarili ciktisi + Loewe yolunun ValueError mesaji.

Hiz icin n_realizations/n_candidates/n_timepoints KUCUK tutuldu -- amac
"gercek klinik kullanim" degil, "ayni girdiyle ayni seed -> ayni cikti"
regresyon kilidi. Deterministik oldugu surece kucuk boyut sorun degil.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from worldmodel.patient import (
    load_drugs, load_patients, load_drug_interactions, load_drug_pk_interactions,
)
from worldmodel.simulation import (
    run_monte_carlo, recommend_dose, run_polypharmacy_simulation, run_polypharmacy_simulation_loewe,
    build_interaction_matrix, build_pk_interaction_matrix, recommend_polypharmacy_dose_scale,
)
from integrate_drug_with_circadapt import run_comparison, run_polypharmacy_comparison

GOLDEN_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

# Hiz/determinizm icin kucultulmus ama anlamli parametreler -- TUM
# senaryolarda AYNI (karsilastirilabilirlik icin).
N_REALIZATIONS = 24
N_TIMEPOINTS = 40
HOURS = 6.0
SEED = 42

DOSE_REC_KWARGS = dict(n_candidates=5, n_realizations=16, hours=HOURS, n_timepoints=N_TIMEPOINTS, seed=SEED)


def mc_kwargs():
    return dict(n_realizations=N_REALIZATIONS, hours=HOURS, n_timepoints=N_TIMEPOINTS, seed=SEED)


def array_payload(prefix, result):
    return {
        f"{prefix}__t": result.t,
        f"{prefix}__hr_runs": result.hr_runs,
        f"{prefix}__sbp_runs": result.sbp_runs,
        f"{prefix}__conc_runs": result.conc_runs,
    }


def circadapt_payload(prefix, result):
    payload = {}
    for key in ("t_base", "p_base", "v_base", "t_drug", "p_drug", "v_drug"):
        payload[f"{prefix}__{key}"] = np.asarray(result[key])
    return payload


def circadapt_scalars(result):
    return {
        "hr_base": result["hr_base"],
        "hr_drug_model": result["hr_drug_model"],
        "tau_av_base_ms": result["tau_av_base_ms"],
        "tau_av_drug_ms": result["tau_av_drug_ms"],
        "av_block_triggered": result.get("av_block_triggered", False),
    }


def dose_rec_scalars(rec):
    return {
        "dose_mg": rec["dose_mg"],
        "statistical_dose_mg": rec["statistical_dose_mg"],
        "is_safe": rec["is_safe"],
        "mechanical_risk": rec["mechanical_risk"],
        "polypharmacy_risk": rec["polypharmacy_risk"],
        "electrolyte_warning": rec["electrolyte_warning"],
        "reasoning": rec["reasoning"],
        "stats": rec["stats"],
    }


def scale_rec_scalars(rec):
    return {
        "scale": rec["scale"],
        "adjusted_doses": rec["adjusted_doses"],
        "is_safe": rec["is_safe"],
        "reasoning": rec["reasoning"],
        "stats": rec["stats"],
    }


def main():
    patients = load_patients(os.path.join(CONFIG_DIR, "patients.yaml"))
    drugs = load_drugs(os.path.join(CONFIG_DIR, "drugs.yaml"))
    interactions = load_drug_interactions(os.path.join(CONFIG_DIR, "drug_interactions.yaml"))
    pk_interactions = load_drug_pk_interactions(os.path.join(CONFIG_DIR, "drug_pk_interactions.yaml"))

    hasta_a = patients["hasta_a"]
    hasta_k = patients["hasta_c_hiperkalemi"]
    beta = drugs["beta_bloker"]
    digoxin = drugs["digoxin"]
    nicardipine = drugs["nicardipine"]

    arrays = {}
    scalars = {}

    # --- N=1 -----------------------------------------------------------
    for label, patient in (("hasta_a", hasta_a), ("hasta_k", hasta_k)):
        mc = run_monte_carlo(patient, beta, **mc_kwargs())
        arrays.update(array_payload(f"n1_{label}_mc", mc))

        rec = recommend_dose(patient, beta, **DOSE_REC_KWARGS)
        scalars[f"n1_{label}_recommend_dose"] = dose_rec_scalars(rec)

        cc = run_comparison(patient, beta)
        arrays.update(circadapt_payload(f"n1_{label}_circadapt", cc))
        scalars[f"n1_{label}_circadapt"] = circadapt_scalars(cc)

    # --- N=2: beta_bloker + digoxin (bilinen PK+PD etkilesimli cift) ---
    drug_keys = ["beta_bloker", "digoxin"]
    interaction_matrix = build_interaction_matrix(drug_keys, interactions)
    pk_interaction_matrix = build_pk_interaction_matrix(drug_keys, pk_interactions)

    for label, patient in (("hasta_a", hasta_a), ("hasta_k", hasta_k)):
        combo = [beta, digoxin]

        additive = run_polypharmacy_simulation(
            patient, combo, interaction_matrix=interaction_matrix,
            drug_keys=drug_keys, pk_interaction_matrix=pk_interaction_matrix, **mc_kwargs(),
        )
        arrays.update(array_payload(f"n2_{label}_additive", additive))

        loewe = run_polypharmacy_simulation_loewe(patient, combo, **mc_kwargs())
        arrays.update(array_payload(f"n2_{label}_loewe", loewe))

        dose_scale_rec = recommend_polypharmacy_dose_scale(patient, combo, **DOSE_REC_KWARGS)
        scalars[f"n2_{label}_dose_scale"] = scale_rec_scalars(dose_scale_rec)

        # streamlit_app.py N=2 dali: recommend_dose(drugs[0], polypharmacy_result=additive)
        dose_rec_n2 = recommend_dose(
            patient, beta, polypharmacy_result=additive,
            polypharmacy_description=digoxin.display_name, **DOSE_REC_KWARGS,
        )
        scalars[f"n2_{label}_recommend_dose"] = dose_rec_scalars(dose_rec_n2)

        combo_circadapt = run_polypharmacy_comparison(patient, combo)
        arrays.update(circadapt_payload(f"n2_{label}_circadapt", combo_circadapt))
        scalars[f"n2_{label}_circadapt"] = circadapt_scalars(combo_circadapt)

    # --- N=2: zit yonlu kombinasyon (beta_bloker +25 / nicardipine -6) -
    opposite_combo = [beta, nicardipine]

    additive_opposite = run_polypharmacy_simulation(hasta_a, opposite_combo, **mc_kwargs())
    arrays.update(array_payload("n2_opposite_additive", additive_opposite))

    try:
        run_polypharmacy_simulation_loewe(hasta_a, opposite_combo, **mc_kwargs())
        loewe_opposite_error = None
    except ValueError as e:
        loewe_opposite_error = str(e)
    scalars["n2_opposite_loewe_error"] = loewe_opposite_error

    combo_opposite_circadapt = run_polypharmacy_comparison(hasta_a, opposite_combo)
    arrays.update(circadapt_payload("n2_opposite_circadapt", combo_opposite_circadapt))
    scalars["n2_opposite_circadapt"] = circadapt_scalars(combo_opposite_circadapt)

    np.savez(os.path.join(GOLDEN_DIR, "n1_n2_golden.npz"), **arrays)
    with open(os.path.join(GOLDEN_DIR, "n1_n2_golden_scalars.json"), "w", encoding="utf-8") as f:
        json.dump(scalars, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"{len(arrays)} dizi, {len(scalars)} skaler-blok kaydedildi -> {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
