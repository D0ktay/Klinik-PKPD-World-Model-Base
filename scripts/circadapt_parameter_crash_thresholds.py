"""
ADIM 4.1 -- CircAdapt parametre-basina guvenli aralik deneyi.

N_DRUG_AUDIT.md Supphe F: bugune kadar SADECE Timings.c_tau_av1 icin
(CALIBRATION_REPORT.md Sec 8) bir cokus esigi (5x-7x) izole olcumu vardi.
Bu betik AYNI izolasyon yontemini (baseline modele TEK bir parametreyi,
digerlerine dokunmadan, kademeli buyuk bir carpanla uygulayip cokup
cokmedigini olcmek) Patch.Sf_act, ArtVen.p0[0] ve General.t_cycle icin de
tekrarlar -- N ilacta bu parametreler de carpimsal biriktigi icin
(integrate_drug_with_circadapt.py > apply_drug_effect_to_circadapt), N=4-5
ilacta bunlarin da cokup cokmeyecegini olcmeden bilmiyoruz.

Calistirma: .venv/Scripts/python.exe scripts/circadapt_parameter_crash_thresholds.py
Cikti: konsola tablo + CALIBRATION_REPORT.md'ye eklenecek Markdown tablosu.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from circadapt import VanOsta2024
from circadapt.error import CircAdaptException

from worldmodel.patient import load_patients
from integrate_drug_with_circadapt import (
    run_stable, calibrate_circadapt_to_patient, VENTRICLE_WALL_INDICES, SYSTEMIC_ARTVEN_INDEX,
)

MULTIPLIERS = [1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0, 7.0, 9.0, 15.0, 50.0, 100.0, 200.0, 500.0]


def build_baseline_model(patient):
    model = VanOsta2024()
    calibrate_circadapt_to_patient(model, patient)
    return model


def set_sf_act(model, mult):
    arr = model["Patch"]["Sf_act"]
    arr[VENTRICLE_WALL_INDICES] = arr[VENTRICLE_WALL_INDICES] * mult


def set_artven_p0(model, mult):
    model["PFC"]["is_active"] = False
    arr = model["ArtVen"]["p0"]
    arr[SYSTEMIC_ARTVEN_INDEX] = arr[SYSTEMIC_ARTVEN_INDEX] * mult


def set_t_cycle(model, mult):
    t_cycle = model["General"]["t_cycle"]
    model["General"]["t_cycle"] = t_cycle * mult


def set_c_tau_av1(model, mult):
    arr = model["Timings"]["c_tau_av1"]
    arr[0] = arr[0] * mult


PARAMETERS = {
    "Patch.Sf_act": set_sf_act,
    "ArtVen.p0[0]": set_artven_p0,
    "General.t_cycle": set_t_cycle,
    "Timings.c_tau_av1": set_c_tau_av1,  # kontrol -- bilinen sonucla (5x-7x) karsilastirma icin tekrar olculuyor
}


def probe(patient, setter, multiplier):
    model = build_baseline_model(patient)
    setter(model, multiplier)
    try:
        run_stable(model)
    except CircAdaptException as e:
        return False, type(e).__name__
    except Exception as e:  # noqa -- beklenmeyen bir hata da "cokme" olarak kaydedilsin, sessizce yutulmasin
        return False, f"UNEXPECTED:{type(e).__name__}:{e}"
    return True, None


def main():
    patients = load_patients(os.path.join(os.path.dirname(__file__), "..", "configs", "patients.yaml"))
    patient = patients["hasta_a"]

    results = {}
    for param_name, setter in PARAMETERS.items():
        results[param_name] = {}
        first_crash = None
        for mult in MULTIPLIERS:
            ok, err = probe(patient, setter, mult)
            results[param_name][mult] = {"ok": ok, "error": err}
            status = "STABIL" if ok else f"COKTU ({err})"
            print(f"{param_name:20s} x{mult:>5.1f}: {status}")
            if not ok and first_crash is None:
                first_crash = mult
        results[param_name]["first_crash_multiplier"] = first_crash
        print(f"  -> İLK ÇÖKÜŞ: x{first_crash}" if first_crash else "  -> hiç çökmedi (test edilen aralıkta)")
        print()

    out_path = os.path.join(os.path.dirname(__file__), "..", "logs", "circadapt_parameter_crash_thresholds.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
