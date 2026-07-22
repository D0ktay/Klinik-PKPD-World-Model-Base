"""
CLI giriş noktası.

Kullanım:
    python app.py --patient hasta_a --drug beta_bloker --n 300
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import load_patients, load_drugs
from worldmodel.simulation import run_monte_carlo, summarize
from worldmodel.viz import plot_results


def main():
    parser = argparse.ArgumentParser(description="Mini Klinik Dünya Modeli Simülatörü")
    parser.add_argument("--patient", default="hasta_a", help="configs/patients.yaml içindeki anahtar")
    parser.add_argument("--drug", default="beta_bloker", help="configs/drugs.yaml içindeki anahtar")
    parser.add_argument("--n", type=int, default=300, help="Monte Carlo deneme sayısı")
    parser.add_argument("--hours", type=float, default=8.0, help="Takip süresi (saat)")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))

    patient = patients[args.patient]
    drug = drugs[args.drug]

    print(f"Simüle ediliyor: {patient.name} + {drug.display_name} ({args.n} deneme)")
    result = run_monte_carlo(patient, drug, n_realizations=args.n, hours=args.hours)

    stats = summarize(result)
    print("\n--- Özet ---")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")

    out_path = os.path.join(base, "outputs", f"{args.patient}_{args.drug}.png")
    plot_results(result, patient, drug, save_path=out_path)


if __name__ == "__main__":
    main()
