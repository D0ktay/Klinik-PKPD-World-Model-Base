"""
ADIM 4.1 (devam) -- yukari yonlu (>1x) carpanlarin yani sira, ASAGI yonlu
(<1x, parametreyi SIFIRA yaklastiran) carpanlarin da cokus esigini olcer.

Gerekce: pozitif inotrop/kronotrop ilaclar (dobutamin gibi, hr_fraction>1)
t_cycle'i KUCULTUR (daha hizli kalp), vazodilatorler/beta-blokerler
ArtVen.p0/Sf_act'i KUCULTUR (sbp_fraction<1) -- yukari yonlu deney
(circadapt_parameter_crash_thresholds.py) bu yonu KAPSAMIYOR.
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

# 1.0'a yakin degerlerden 0'a dogru kademeli kucul -- "carpan" burada
# DOGRUDAN parametreye uygulanan katsayi (0.5 = yariya indirildi, vb.)
MULTIPLIERS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.10, 0.05, 0.02, 0.01]


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


PARAMETERS = {
    "Patch.Sf_act": set_sf_act,
    "ArtVen.p0[0]": set_artven_p0,
    "General.t_cycle": set_t_cycle,
}


def probe(patient, setter, multiplier):
    model = build_baseline_model(patient)
    setter(model, multiplier)
    try:
        run_stable(model)
    except CircAdaptException as e:
        return False, type(e).__name__
    except Exception as e:  # noqa
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
            print(f"{param_name:20s} x{mult:>5.2f}: {status}")
            if not ok and first_crash is None:
                first_crash = mult
        results[param_name]["first_crash_multiplier"] = first_crash
        print(f"  -> İLK ÇÖKÜŞ (aşağı yönde): x{first_crash}" if first_crash else "  -> hiç çökmedi (test edilen aralıkta)")
        print()

    out_path = os.path.join(os.path.dirname(__file__), "..", "logs", "circadapt_parameter_crash_thresholds_downward.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
