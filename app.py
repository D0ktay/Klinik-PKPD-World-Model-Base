"""
CLI giriş noktası.

Kullanım:
    python app.py --patient hasta_a --drug beta_bloker --n 300
    python app.py --patient hasta_a --drug beta_bloker digoxin --n 300   # polifarmasi (N ilaç)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worldmodel.patient import (
    load_patients, load_drugs, load_drug_interactions, load_drug_pk_interactions,
)
from worldmodel.simulation import (
    run_monte_carlo, run_polypharmacy_simulation, run_polypharmacy_simulation_loewe,
    build_interaction_matrix, build_pk_interaction_matrix, summarize,
)
from worldmodel.viz import plot_results


def main():
    parser = argparse.ArgumentParser(description="Mini Klinik Dünya Modeli Simülatörü")
    parser.add_argument("--patient", default="hasta_a", help="configs/patients.yaml içindeki anahtar")
    parser.add_argument("--drug", nargs="+", default=["beta_bloker"],
                         help="configs/drugs.yaml içindeki anahtar(lar) -- birden fazla verilirse polifarmasi (N ilaç) modu çalışır")
    parser.add_argument("--n", type=int, default=300, help="Monte Carlo deneme sayısı")
    parser.add_argument("--hours", type=float, default=8.0, help="Takip süresi (saat)")
    parser.add_argument("--interaction-model", choices=["additive", "loewe"], default="additive",
                         help="Polifarmasi (2+ ilaç) birleştirme yöntemi: 'additive' (varsayılan, mevcut "
                              "toplamsal model) veya 'loewe' (Loewe additivity -- doz-eşdeğerliği bazlı, "
                              "bkz. worldmodel/pd.py > loewe_combined_effect). Tek ilaçta etkisizdir.")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    patients = load_patients(os.path.join(base, "configs", "patients.yaml"))
    drugs = load_drugs(os.path.join(base, "configs", "drugs.yaml"))

    patient = patients[args.patient]

    if len(args.drug) == 1:
        # Tek ilaç -- mevcut davranış birebir korunur (regresyon yok).
        drug = drugs[args.drug[0]]

        print(f"Simüle ediliyor: {patient.name} + {drug.display_name} ({args.n} deneme)")
        result = run_monte_carlo(patient, drug, n_realizations=args.n, hours=args.hours)

        stats = summarize(result)
        print("\n--- Özet ---")
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")

        out_path = os.path.join(base, "outputs", f"{args.patient}_{args.drug[0]}.png")
        plot_results(result, patient, drug, save_path=out_path)
    else:
        # Polifarmasi -- N ilaç aynı anda. Grafik üretilmiyor (plot_results
        # tek bir Drug bekliyor); bunun yerine her ilacın tek başına ve
        # BİRLİKTE sonucunu konsola özetliyoruz (bkz. compare_polypharmacy.py).
        drug_list = [drugs[key] for key in args.drug]
        interactions = load_drug_interactions(os.path.join(base, "configs", "drug_interactions.yaml"))
        interaction_matrix = build_interaction_matrix(args.drug, interactions)

        # PK-seviyeli (klerens/AUC) ilaç-ilaç etkileşimi -- PD-seviyesindeki
        # interaction_matrix'ten AYRI, bkz. pk.py > pk_interaction_adjusted_ke().
        pk_interactions = load_drug_pk_interactions(os.path.join(base, "configs", "drug_pk_interactions.yaml"))
        pk_interaction_matrix = build_pk_interaction_matrix(args.drug, pk_interactions)

        print(f"Simüle ediliyor: {patient.name} + {' + '.join(d.display_name for d in drug_list)} ({args.n} deneme)")

        for key, drug in zip(args.drug, drug_list):
            solo_stats = summarize(run_monte_carlo(patient, drug, n_realizations=args.n, hours=args.hours))
            print(f"  {drug.display_name} tek başına: ortalama en düşük nabız={solo_stats['mean_min_hr']:.1f} bpm, "
                  f"bradikardi riski=%{solo_stats['pct_bradycardia_risk']:.1f}")

        combo_result = run_polypharmacy_simulation(
            patient, drug_list, n_realizations=args.n, hours=args.hours, interaction_matrix=interaction_matrix,
            drug_keys=args.drug, pk_interaction_matrix=pk_interaction_matrix,
        )
        combo_stats = summarize(combo_result)
        print(f"\n--- BİRLİKTE (toplamsal/additive model) ---")
        for k, v in combo_stats.items():
            print(f"{k}: {v:.2f}")
        if interaction_matrix:
            print("(bilinen çift etkileşim verisi -- configs/drug_interactions.yaml -- hesaba katıldı)")
        if pk_interaction_matrix:
            for (perpetrator, victim), auc_ratio in pk_interaction_matrix.items():
                record = next(
                    r for r in pk_interactions
                    if r["perpetrator_drug"] == perpetrator and r["victim_drug"] == victim
                )
                print(
                    f"(PK-seviyeli etkileşim: {drugs[perpetrator].display_name} ilacı, "
                    f"{drugs[victim].display_name} klerensini [ilacın vücuttan atılım hızını] "
                    f"~{auc_ratio}x oranında etkiliyor -- kaynak: {record['source']})"
                )

        if args.interaction_model == "loewe":
            # Loewe additivity, sadece TÜM ilaçların emax'ı AYNI yöndeyse
            # (hepsi nabzı düşürüyor ya da hepsi artırıyor) anlamlı --
            # aksi halde loewe_combined_effect() ValueError fırlatır.
            try:
                loewe_result = run_polypharmacy_simulation_loewe(
                    patient, drug_list, n_realizations=args.n, hours=args.hours,
                )
                loewe_stats = summarize(loewe_result)
                print(f"\n--- BİRLİKTE (Loewe additivity -- doz-eşdeğerliği bazlı) ---")
                for k, v in loewe_stats.items():
                    print(f"{k}: {v:.2f}")
                print(
                    "(İki model farklı çıkabilir -- ilaçların Emax'ları [tavan etkisi] birbirinden "
                    "uzaksa düz toplama yanıltıcı olabilir, bkz. Tallarida 'curved isobole' uyarısı.)"
                )
            except ValueError as e:
                print(f"\n--- Loewe additivity hesaplanamadı: {e} ---")


if __name__ == "__main__":
    main()
