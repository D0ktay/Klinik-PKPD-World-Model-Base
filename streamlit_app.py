"""
İnteraktif demo — mülakatta canlı göstermek için.

Çalıştırma:
    streamlit run streamlit_app.py

Slider'larla kilo, doz, deneme sayısını değiştir, grafiğin anlık
değiştiğini göster -- "hastanın kilosu gerçekten sonucu değiştiriyor"
mesajını canlı kanıtlamanın en etkili yolu budur.

Sayfa tek uzun akış değil, 6 sekmeye ayrıldı (Hasta Girişi / İlaç Seçimi /
Simülasyon / CircAdapt Sonuçları / Dünya Modelini Gözlemle / Rapor İndir)
-- her sekmenin widget'ları normal Python değişkenleri döndürür, bu yüzden
sekmeler arasında veri akışı (patient/drug/sim) değişmez.
"""

import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from worldmodel.patient import Patient, load_drugs
from worldmodel.simulation import run_monte_carlo, summarize, recommend_dose, run_reference_trace
from worldmodel.viz import plot_results
from worldmodel.provenance import provenance_report, SOURCE_TYPE_EMOJI
from worldmodel.report import export_report
from worldmodel.clinical_metrics import ejection_fraction, cardiac_output, classify_cardiac_function
from integrate_drug_with_circadapt import run_comparison, build_comparison_figure

st.set_page_config(page_title="Mini Klinik Dünya Modeli", layout="wide", page_icon="🩺")

st.markdown("""
<style>
/* Klinik görünüm: nötr tipografi, düz (gölgesiz) kart yüzeyleri, ölçülü renk paleti. */
html, body, [class*="css"] { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }

.clinical-banner {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 1.1rem 1.4rem; margin-bottom: 1.1rem;
    background: #0E6E73; border-radius: 6px; color: #FFFFFF;
}
.clinical-banner .mark {
    width: 40px; height: 40px; flex-shrink: 0; border-radius: 6px;
    background: rgba(255,255,255,0.15); display: flex; align-items: center;
    justify-content: center; font-size: 1rem; font-weight: 700; letter-spacing: 0.02em;
}
.clinical-banner h1 {
    font-size: 1.35rem; margin: 0; font-weight: 600; letter-spacing: 0.01em;
}
.clinical-banner p {
    margin: 0.15rem 0 0 0; font-size: 0.86rem; color: rgba(255,255,255,0.82);
}
.clinical-tag {
    margin-left: auto; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; background: rgba(255,255,255,0.16);
    padding: 0.25rem 0.6rem; border-radius: 4px; white-space: nowrap;
}

div[data-testid="stMetric"] {
    background: #F0F4F5; border: 1px solid #DCE4E6; border-radius: 6px;
    padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #4C6067; }

h2, h3 { color: #123338; font-weight: 600; }

div[data-testid="stTabs"] button[role="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="clinical-banner">
    <div class="mark">MK</div>
    <div>
        <h1>Mini Klinik Dünya Modeli</h1>
        <p>PK/PD + CircAdapt kalp-damar mekaniği ile sanal hasta simülasyonu</p>
    </div>
    <div class="clinical-tag">Eğitim / Araştırma Amaçlı</div>
</div>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(__file__)
all_drugs = load_drugs(os.path.join(base_dir, "configs", "drugs.yaml"))

DRUG_CLASS_LABELS = {
    "beta_blocker": "Beta-bloker",
    "vasodilator": "Vazodilatör",
    "positive_inotrope": "Pozitif inotrop",
}
COMORBIDITY_OPTIONS = {
    "Yok (sağlıklı)": None,
    "Sistolik kalp yetmezliği": "heart_failure",
    "Kronik hipertansiyon": "hypertension",
}

tab_patient, tab_drug, tab_sim, tab_heart, tab_observe, tab_report = st.tabs([
    "🧑‍⚕️ Hasta Girişi", "💊 İlaç Seçimi", "📊 Simülasyon",
    "🫀 CircAdapt Sonuçları", "🔎 Dünya Modelini Gözlemle", "📄 Rapor İndir",
])

# --- Sekme 1: Hasta Girişi ---
with tab_patient:
    st.subheader("Temel Bilgiler")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Yaş", 18, 95, 45,
                         help="Hastanın yaşı -- rapora ve klinik özet bilgilerine yansır; "
                              "ileri yaş, doz-yanıt yorumunda dikkate alınması gereken bir "
                              "faktördür.")
        weight = st.slider("Kilo (kg)", 40, 150, 76,
                            help="Hastanın kilosu -- ilacın dağılım hacmini (Vd) belirler, "
                                 "bu yüzden aynı doz farklı kilodaki hastalarda farklı "
                                 "konsantrasyon üretir.")
    with c2:
        height = st.slider("Boy (cm)", 140, 210, 175,
                            help="Hastanın boyu -- vücut yüzey alanı hesabında kullanılır.")
        baseline_hr = st.slider("Bazal nabız (bpm)", 50, 110, 78,
                                 help="Nabız: kalbin dakikada kaç kez attığı (bpm = beats per "
                                      "minute). Bu, hastanın ilaçsız/dinlenme halindeki değeri "
                                      "-- CircAdapt simülasyonu da bu değere kalibre edilir.")
    with c3:
        baseline_sbp = st.slider("Bazal sistolik tansiyon (mmHg)", 90, 180, 125,
                                  help="Sistolik tansiyon (SBP): kalbin kasılıp kan pompaladığı "
                                       "an (sistol) ölçülen, tansiyonun YÜKSEK değeri.")

    with st.expander("Gelişmiş: Böbrek/Karaciğer Fonksiyonu, Elektrolitler, Komorbidite"):
        st.caption(
            "Bu alanlar varsayılan (normal) değerlerinde bırakılırsa hiçbir "
            "etkisi olmaz -- sadece anormal bir değer girildiğinde devreye girer."
        )
        e1, e2 = st.columns(2)
        with e1:
            renal_function = st.slider(
                "Böbrek fonksiyonu (1.0=normal, 0=tam yetmezlik)", 0.0, 1.0, 1.0, step=0.05,
                help="Böbrek fonksiyonu bozulduğunda, SADECE böbrekten atılan ilaçların "
                     "(ör. digoksin) eliminasyonu yavaşlar -- böbrekten bağımsız ilaçlar "
                     "(ör. esmolol) etkilenmez.")
            potassium = st.slider(
                "Potasyum (mEq/L, normal 3.5-5.0)", 2.0, 8.0, 4.25, step=0.05,
                help="Kandaki potasyum düzeyi. Hiperkalemi (>5.0, potasyumun normalden "
                     "yüksek olması) kalbin elektrik iletimini (AV düğümü -- üst ve alt "
                     "kalp odacıkları arasındaki sinyal iletimi) yavaşlatır.")
        with e2:
            hepatic_function = st.slider(
                "Karaciğer fonksiyonu (1.0=normal, 0=tam yetmezlik)", 0.0, 1.0, 1.0, step=0.05,
                help="Karaciğer fonksiyonu bozulduğunda, SADECE karaciğerden metabolize "
                     "olan ilaçların eliminasyonu yavaşlar.")
            calcium = st.slider(
                "Kalsiyum (mg/dL, normal 8.5-10.5)", 5.0, 13.0, 9.5, step=0.05,
                help="Kandaki kalsiyum düzeyi. Kalsiyum, kontraktiliteyle (kalp kasının ne "
                     "kadar güçlü kasıldığı) doğru orantılıdır -- hipokalsemi (düşük "
                     "kalsiyum) kontraktiliteyi azaltır.")
        comorbidity_label = st.selectbox(
            "Komorbidite", list(COMORBIDITY_OPTIONS.keys()),
            help="Komorbidite: hastanın, bu simülasyondan BAĞIMSIZ olarak ZATEN sahip "
                 "olduğu kronik bir kalp hastalığı. Kalp yetmezliği = azalmış kontraktilite "
                 "(düşük EF). Hipertansiyon = kronik olarak yüksek damar direnci/basınç.")
        comorbidity = COMORBIDITY_OPTIONS[comorbidity_label]

# --- Sekme 2: İlaç Seçimi ---
with tab_drug:
    st.subheader("İlaç & Simülasyon Ayarları")
    drug_key = st.selectbox(
        "İlaç",
        options=list(all_drugs.keys()),
        format_func=lambda k: (
            f"{all_drugs[k].display_name} "
            f"[{DRUG_CLASS_LABELS.get(all_drugs[k].drug_class, all_drugs[k].drug_class)}]"
        ),
        index=0,
        help="Beta-bloker: nabzı/kontraktiliteyi düşürür. Vazodilatör: damar direncini "
             "düşürür (kontraktiliteye dokunmaz). Pozitif inotrop: kontraktiliteyi artırır.",
    )
    base_drug = all_drugs[drug_key]

    # Her ilacın gerçek referans dozu çok farklı (esmolol ~38mg, nikardipin
    # ~2.3mg) -- slider aralığını seçili ilacın kendi referans dozuna göre
    # dinamik ayarlıyoruz, tek sabit "1-20mg" aralığı tüm ilaçlara uymuyor.
    ref_dose = base_drug.dose_mg_per_kg * weight if base_drug.dose_mg_per_kg else base_drug.dose_mg
    dose = st.slider("Doz (mg)", round(ref_dose * 0.2, 2), round(ref_dose * 3.0, 2),
                      round(ref_dose, 2),
                      help="Verilen ilaç miktarı. Varsayılan değer, seçili ilacın kendi "
                           "referans dozuna (genelde mg/kg cinsinden, hastanın kilosuyla "
                           "ölçeklenmiş) göre ayarlanır.")
    n_runs = st.slider("Monte Carlo deneme sayısı", 50, 1000, 300, step=50,
                        help="Aynı hasta+ilaç kombinasyonunun kaç kez (her seferinde "
                             "bireysel varyasyon rastgele örneklenerek) simüle edileceği "
                             "-- daha fazla deneme, daha kararlı bir istatistik verir.")
    # EC50 ilaca göre çok küçük olabilir (nikardipin: 0.0015 mg/L) --
    # sabit "%.2f" formatı bu durumda "0.00" gösterip değeri gizlerdi.
    ec50_min, ec50_max = float(base_drug.ec50 * 0.2), float(base_drug.ec50 * 5)
    ec50_step = (ec50_max - ec50_min) / 100
    ec50 = st.slider("EC50 (etki eşiği, mg/L)", ec50_min, ec50_max, float(base_drug.ec50),
                      step=ec50_step, format="%.5f",
                      help="EC50: ilacın YARI-MAKSİMUM etkiyi yapması için gereken "
                           "konsantrasyon. Düşük EC50, ilacın az miktarda bile etkili "
                           "olduğu (güçlü/duyarlı) anlamına gelir.")

patient = Patient(
    name="Canlı Demo Hastası", weight_kg=weight, height_cm=height,
    age=age, blood_type="A Rh+", baseline_hr=baseline_hr,
    baseline_sbp=baseline_sbp, baseline_dbp=80, baseline_spo2=97,
    renal_function=renal_function, hepatic_function=hepatic_function,
    potassium_mEqL=potassium, calcium_mgdL=calcium, comorbidity=comorbidity,
)

# dose_mg_per_kg=None: slider'daki mutlak mg değerinin gerçekten
# kullanılmasını garantiler -- aksi halde kilo bazlı dozlanan ilaçlarda
# dose_mg_per_kg her zaman öncelikli olacağından slider sessizce yok sayılırdı.
drug = replace(base_drug, dose_mg=dose, dose_mg_per_kg=None, ec50=ec50)

# Butona basıldığı an geçerli olan hasta+ilaç girdilerinin "parmak izi" --
# sonuçlar gösterilirken, ekrandaki girdiler bundan farklıysa kullanıcıyı uyarmak için.
current_inputs = (
    drug_key, patient.age, patient.weight_kg, patient.height_cm, patient.baseline_hr,
    patient.baseline_sbp, patient.renal_function, patient.hepatic_function,
    patient.potassium_mEqL, patient.calcium_mgdL, patient.comorbidity,
    drug.dose_mg, drug.ec50, n_runs,
)

# --- Sekme 3: Simülasyon ---
with tab_sim:
    if st.button("Simülasyonu Çalıştır", type="primary"):
        with st.spinner("Monte Carlo (PK/PD) simülasyonu çalıştırılıyor..."):
            mc_result = run_monte_carlo(patient, drug, n_realizations=n_runs)
            mc_stats = summarize(mc_result)

        with st.spinner(
            "CircAdapt kalp simülasyonu çalıştırılıyor (baseline + ilaçlı) -- "
            "gerçek kalp-damar mekaniği modeli olduğu için bu adım birkaç saniyeden "
            "birkaç dakikaya kadar sürebilir, lütfen bekleyin..."
        ):
            heart_result = run_comparison(patient, drug)

        with st.spinner("En iyi doz önerisi hesaplanıyor (istatistiksel + mekanik risk)..."):
            dose_rec = recommend_dose(patient, drug, circadapt_results=heart_result)

        st.session_state["sim"] = {
            "dose_rec": dose_rec,
            "mc_result": mc_result,
            "mc_stats": mc_stats,
            "heart_result": heart_result,
            "drug_name": drug.display_name,
            "drug": drug,
            "patient": patient,
            "inputs": current_inputs,
        }

    if "sim" not in st.session_state:
        st.info("Sonuçları görmek için yukarıdaki butona basın.")
    else:
        sim = st.session_state["sim"]

        if sim["inputs"] != current_inputs:
            st.warning(
                "Hasta/ilaç girdilerini değiştirdiniz. Aşağıdaki sonuçlar hâlâ eski "
                "girdilerle hesaplandı -- güncellemek için butona tekrar basın."
            )

        # --- Doz önerisi (istatistiksel + CircAdapt mekanik risk birleşimi) ---
        dose_rec = sim["dose_rec"]
        if dose_rec["mechanical_risk"]:
            st.error(f"⚠️ **Önerilen doz: {dose_rec['dose_mg']:.1f} mg**\n\n{dose_rec['reasoning']}")
        elif dose_rec["is_safe"]:
            st.success(f"💊 **Önerilen en iyi doz: {dose_rec['dose_mg']:.1f} mg**\n\n{dose_rec['reasoning']}")
        else:
            st.warning(f"⚠️ **Önerilen doz: {dose_rec['dose_mg']:.1f} mg**\n\n{dose_rec['reasoning']}")

        # --- PK/PD (Monte Carlo) sonuçları ---
        st.subheader("PK/PD Simülasyonu — Nabız & Tansiyon Dağılımı")
        fig_mc = plot_results(sim["mc_result"], patient, drug)
        st.pyplot(fig_mc)

        st.subheader("Klinik Özet")
        summary_card = st.container(border=True)
        m1, m2, m3 = summary_card.columns(3)
        m1.metric("Ortalama en düşük nabız", f"{sim['mc_stats']['mean_min_hr']:.1f} bpm",
                   help="Yüzlerce sanal denemenin her birinde ulaşılan EN DÜŞÜK nabzın "
                        "ortalaması -- ilacın en güçlü etkisini gösterdiği andaki nabız.")
        m2.metric("5-95 persentil aralığı",
                   f"{sim['mc_stats']['p5_min_hr']:.0f} - {sim['mc_stats']['p95_min_hr']:.0f}",
                   help="En düşük nabzın, denemelerin %90'ının içine düştüğü aralık -- "
                        "hastadan hastaya (bireysel varyasyondan kaynaklanan) belirsizliğin "
                        "ölçüsü.")
        m3.metric("Bradikardi riski (nabız<50)", f"%{sim['mc_stats']['pct_bradycardia_risk']:.1f}",
                   help="Bradikardi: anormal derecede YAVAŞ kalp atışı (<50 bpm eşiği "
                        "kullanılıyor). Bu yüzde, denemelerin kaçında nabzın bu tehlikeli "
                        "eşiğin altına indiğini gösterir.")

# --- Sekme 4: CircAdapt Sonuçları ---
with tab_heart:
    if "sim" not in st.session_state:
        st.info("Önce 'Simülasyon' sekmesinde simülasyonu çalıştırın.")
    else:
        sim = st.session_state["sim"]

        st.subheader("CircAdapt ile Gerçek Kalp Simülasyonu")
        st.caption(
            "PK/PD zincirinin ürettiği nabız/tansiyon etkisini, CircAdapt'in gerçek "
            "kalp-damar mekaniği modeline aktarır -- ilaç sınıfına göre farklı bir "
            "mekanizmaya bağlanır: beta-bloker/pozitif inotrop -> General.t_cycle + "
            "Patch.Sf_act (kontraktilite); vazodilatör -> General.t_cycle + ArtVen "
            "sistemik direnci (kontraktiliteye dokunulmaz). İlaçsız/ilaçlı sol "
            "karıncık basıncını karşılaştırır."
        )

        heart = sim["heart_result"]
        drug_effect = heart["drug_effect"]

        fig_heart = build_comparison_figure(
            heart["t_base"], heart["p_base"], heart["v_base"],
            heart["t_drug"], heart["p_drug"], heart["v_drug"],
            sim["drug_name"], heart["hr_base"], heart["hr_drug_model"],
        )
        st.pyplot(fig_heart)

        st.markdown(
            f"**Pik etki:** t={drug_effect['t_peak_hours']:.2f} saat, "
            f"konsantrasyon={drug_effect['conc_peak']:.3f} mg/L, "
            f"etki oranı={drug_effect['effect_fraction']:.2f}"
        )

        heart_metrics_card = st.container(border=True)
        h1, h2, h3 = heart_metrics_card.columns(3)
        h1.metric(
            "Maksimum LV basıncı",
            f"{heart['p_drug'].max():.1f} mmHg",
            delta=f"{heart['p_drug'].max() - heart['p_base'].max():+.1f} mmHg",
            help="LV (sol karıncık -- kalbin vücuda kan pompalayan ana odacığı) "
                 "basıncının sistol (kalbin kasılıp kan pompaladığı an) sırasındaki "
                 "en yüksek değeri.",
        )
        h2.metric(
            "LVEDV",
            f"{heart['v_drug'].max():.1f} mL",
            delta=f"{heart['v_drug'].max() - heart['v_base'].max():+.1f} mL",
            help="LVEDV / end-diastolic volume (kalbin diyastolde -- gevşeyip "
                 "kanla dolduğu fazda -- ulaştığı en yüksek hacim).",
        )
        h3.metric(
            "CircAdapt nabzı",
            f"{heart['hr_drug_model']:.0f} bpm",
            delta=f"{heart['hr_drug_model'] - heart['hr_base']:+.0f} bpm",
            help="Nabız: kalbin dakikada kaç kez attığı (bpm = beats per minute).",
        )

        # --- Klinik olarak anlamlı metrikler: EF / CO ---
        st.subheader("Kalp Fonksiyonu Değerlendirmesi")
        st.caption(
            "Ham basınç/hacim sayıları yerine, kardiyolojinin gerçekte kullandığı "
            "iki standart metrik: EF (ejeksiyon fraksiyonu -- kalbin her atışta "
            "içindeki kanın yüzde kaçını pompaladığı) ve CO (kardiyak output -- "
            "kalbin dakikada pompaladığı toplam kan miktarı)."
        )

        edv_base, esv_base = heart["v_base"].max(), heart["v_base"].min()
        edv_drug, esv_drug = heart["v_drug"].max(), heart["v_drug"].min()

        ef_base = ejection_fraction(edv_base, esv_base)
        ef_drug = ejection_fraction(edv_drug, esv_drug)
        co_base = cardiac_output(edv_base, esv_base, heart["hr_base"])
        co_drug = cardiac_output(edv_drug, esv_drug, heart["hr_drug_model"])

        cls_base = classify_cardiac_function(ef_base, co_base)
        cls_drug = classify_cardiac_function(ef_drug, co_drug)

        ef_help = (
            "EF / ejeksiyon fraksiyonu (kalbin her atışta içindeki kanın yüzde "
            "kaçını pompaladığı). Normal: %55-70. Hafif azalmış: %40-54. "
            "Düşük (kalp yetmezliği belirtisi): <%40."
        )
        co_help = (
            "CO / kardiyak output (kalp debisi) -- kalbin dakikada pompaladığı "
            "toplam kan miktarı, litre cinsinden. Normal aralık: 4-8 L/dk."
        )

        ef_co_card = st.container(border=True)
        col_base, col_drug = ef_co_card.columns(2)
        with col_base:
            st.markdown(f"**Baseline (ilaçsız)** -- {cls_base['summary']}")
            b1, b2 = st.columns(2)
            b1.metric("EF (baseline)", cls_base["ef_label"], help=ef_help)
            b2.metric("CO (baseline)", cls_base["co_label"], help=co_help)
        with col_drug:
            st.markdown(f"**İlaçlı** -- {cls_drug['summary']}")
            d1, d2 = st.columns(2)
            d1.metric("EF (ilaçlı)", cls_drug["ef_label"],
                      delta=f"{ef_drug - ef_base:+.0f}", help=ef_help)
            d2.metric("CO (ilaçlı)", cls_drug["co_label"],
                      delta=f"{co_drug - co_base:+.1f}", help=co_help)

        color_to_alert = {"yeşil": st.success, "sarı": st.warning, "kırmızı": st.error}
        color_to_alert[cls_drug["overall_color"]](
            f"**Genel değerlendirme (ilaçlı durum):** {cls_drug['summary']} "
            f"({cls_drug['ef_label']}, {cls_drug['co_label']})"
        )

        # --- Veri kaynağı izlenebilirliği ---
        st.divider()
        with st.expander("🔍 Bu sonuç neye dayanıyor?"):
            st.caption(
                "Bu simülasyonda kullanılan her parametrenin kaynağı -- "
                "📚 Literatür (yayınlanmış kaynak) / ⚠️ Varsayım (yönü gerçek "
                "fizyolojiye dayanan ama kalibrasyon gerektiren temsili değer) / "
                "👤 Hasta verisi (bu hasta için girilen değer). Tam detay ve "
                "kaynak referansları için CALIBRATION_REPORT.md."
            )
            report = provenance_report(sim["patient"], sim["drug"])
            st.table([
                {
                    "": SOURCE_TYPE_EMOJI.get(row["source_type"], "❔"),
                    "Parametre": row["parameter"],
                    "Değer": row["value"],
                    "Kaynak": row["source_type"],
                    "Detay": row["detail"],
                }
                for row in report
            ])

# --- Sekme 5: Dünya Modelini Gözlemle ---
# Simülasyonun "kara kutusunu" açıp her adımı okunabilir şekilde göstermek
# için. Bilinçli olarak CircAdapt'ten
# BAĞIMSIZ çalışır (varsayılan motor: run_reference_trace, gürültüsüz/tek
# bir PK/PD izi -- hızlı) -- CircAdapt (yavaş) sadece "Gerçek Kalp
# Modeliyle Göster" butonuna basılırsa, ve sadece İKİ referans an için
# (ilaçsız / pik etki anı) çalıştırılır; 200 PK/PD zaman noktasının HER
# birinde CircAdapt çalıştırmak dakikalar sürerdi.
with tab_observe:
    st.subheader("Dünya Modelini Gözlemle")
    st.caption(
        "Bu sayfa simülasyonun 'kara kutusunu' açar -- her zaman adımında "
        "hesaplanan TÜM ara değerleri (durum + aksiyon -> yeni durum "
        "zincirinin her halkasını) gizlemeden gösterir."
    )

    # --- Akış diyagramı ---
    flow1, flow_arrow1, flow2, flow_arrow2, flow3 = st.columns([4, 1, 4, 1, 4])
    with flow1:
        with st.container(border=True):
            st.markdown("**GİRDİ: Hasta Durumu**")
            st.caption("nabız, tansiyon, kilo, boy")
    with flow_arrow1:
        st.markdown("<div style='text-align:center; padding-top:1.6rem; font-size:1.4rem;'>→</div>",
                     unsafe_allow_html=True)
    with flow2:
        with st.container(border=True):
            st.markdown("**İŞLEME: PK/PD (+ CircAdapt)**")
            st.caption("konsantrasyon hesabı, etki oranı hesabı, (opsiyonel) gerçek kalp simülasyonu")
    with flow_arrow2:
        st.markdown("<div style='text-align:center; padding-top:1.6rem; font-size:1.4rem;'>→</div>",
                     unsafe_allow_html=True)
    with flow3:
        with st.container(border=True):
            st.markdown("**ÇIKTI: Yeni Durum**")
            st.caption("yeni nabız, yeni tansiyon, (opsiyonel) yeni EF/CO")

    st.caption(
        "Bu üç kutu, Vivax'ın Acudx mimarisindeki 'girdi -> evolving patient "
        "state -> çıktı' akışının küçük ölçekli bir versiyonu -- aşağıdaki "
        "tablo ve kontroller bu akışın HER adımını somutlaştırıyor."
    )
    st.divider()

    # --- Referans iz: gürültüsüz, tek, yeniden üretilebilir (varsayılan/hızlı motor) ---
    ref = run_reference_trace(patient, drug)
    n_points = len(ref["t"])

    st.markdown("#### Durum Tablosu")
    st.caption(
        "Her satır, PK/PD motorunun o zaman noktasında hesapladığı TÜM ara "
        "değerleri gösterir -- Monte Carlo'daki gibi gürültü/rastgelelik "
        "YOK, tek ve yeniden üretilebilir bir iz (ke=ortalama, "
        "bireysel duyarlılık=1.0)."
    )
    state_table = pd.DataFrame({
        "Zaman (saat)": ref["t"],
        "Konsantrasyon (mg/L)": ref["conc"],
        "İlaç Etki Oranı — Nabız (0-1)": ref["effect_hr"],
        "İlaç Etki Oranı — Tansiyon (0-1)": ref["effect_sbp"],
        "Kalp Hızı (bpm)": ref["hr"],
        "Sistolik TA (mmHg)": ref["sbp"],
    })
    st.dataframe(
        state_table.style.format({
            "Zaman (saat)": "{:.2f}", "Konsantrasyon (mg/L)": "{:.4f}",
            "İlaç Etki Oranı — Nabız (0-1)": "{:.2f}", "İlaç Etki Oranı — Tansiyon (0-1)": "{:.2f}",
            "Kalp Hızı (bpm)": "{:.1f}", "Sistolik TA (mmHg)": "{:.1f}",
        }),
        use_container_width=True, height=280,
    )

    st.divider()

    # --- Tek Adımı İncele ---
    st.markdown("#### Tek Adımı İncele")
    st.caption(
        "Bir zaman noktası seç -- o ANA ait GİRDİ/AKSİYON/HESAPLAMA/ÇIKTI "
        "zincirinin tamamı açık şekilde gösterilecek."
    )
    step_idx = st.slider(
        "Zaman noktası seç", 0, n_points - 1, min(20, n_points - 1),
        help="Durum Tablosu'ndaki 200 zaman noktasından birini seçer -- "
             "kaydırdıkça altındaki kutu, TAM OLARAK o andaki durumu gösterir.",
    )
    t_sel = ref["t"][step_idx]
    conc_sel = ref["conc"][step_idx]
    effect_sel = ref["effect_hr"][step_idx]
    hr_sel = ref["hr"][step_idx]
    sbp_sel = ref["sbp"][step_idx]
    hr_prev = ref["hr"][step_idx - 1] if step_idx > 0 else patient.baseline_hr
    sbp_prev = ref["sbp"][step_idx - 1] if step_idx > 0 else patient.baseline_sbp

    st.markdown(f"""
```
📍 t = {t_sel:.2f} saat'teki DURUM:

   GİRDİ (bu adıma gelen durum): kalp hızı={hr_prev:.1f} bpm, tansiyon={sbp_prev:.1f} mmHg
   AKSİYON: {drug.display_name} {drug.dose_mg:.1f}mg (t=0'da verildi, şu an konsantrasyon={conc_sel:.4f} mg/L)
   HESAPLAMA: ilaç_etki_oranı = {effect_sel:.2f}  (Emax formülü: sensitivity * conc / (EC50 + conc))
   ÇIKTI (yeni durum): kalp hızı={hr_sel:.1f} bpm, tansiyon={sbp_sel:.1f} mmHg
```
""")

    st.divider()

    # --- Monte Carlo'nun "gizli" değerleri ---
    st.markdown("#### 🎲 Monte Carlo Denemesini İncele")
    st.caption(
        "Aynı hasta+ilaç kombinasyonu neden yüzlerce kez çalıştırılıyor? "
        "Çünkü her denemede ke (eliminasyon hızı) ve bireysel duyarlılık "
        "(sensitivity) RASTGELE örnekleniyor -- gerçek hastalar da ilaca "
        "aynı hızda yanıt vermez. Aşağıda tek bir denemeyi seçip, o "
        "denemede örneklenen değerleri ve sonucu görebilirsin.",
    )
    observe_n_runs = 300
    mc_for_observe = run_monte_carlo(patient, drug, n_realizations=observe_n_runs)

    trial_idx = st.slider(
        "Deneme # seç", 0, observe_n_runs - 1, 0,
        help="300 sanal Monte Carlo denemesinden birini seç -- her deneme, "
             "aynı hasta+ilaç için ke ve bireysel duyarlılığın FARKLI "
             "rastgele örneklenmiş bir kombinasyonunu temsil eder.",
    )
    trial_ke = mc_for_observe.ke_values[trial_idx]
    trial_sens = mc_for_observe.sensitivity_values[trial_idx]
    trial_min_hr = float(mc_for_observe.hr_runs[trial_idx].min())
    population_mean_min_hr = float(mc_for_observe.hr_runs.min(axis=1).mean())

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Bu denemenin ke değeri", f"{trial_ke:.3f} /saat",
               help="ke: ilacın vücuttan ne kadar HIZLI atıldığının ölçüsü. "
                    "Bu deneme için rastgele örneklenen özel değer.")
    tc2.metric("Bu denemenin duyarlılığı", f"{trial_sens:.2f}",
               help="Bireysel duyarlılık (sensitivity): 1.0 = ortalama "
                    "hasta. >1.0 = ilaca ortalamadan daha güçlü yanıt "
                    "veren bir hasta, <1.0 = daha zayıf yanıt.")
    tc3.metric("Bu denemede en düşük nabız", f"{trial_min_hr:.1f} bpm",
               delta=f"{trial_min_hr - population_mean_min_hr:+.1f} bpm (popülasyon ort.)",
               help="Bu TEK denemede ulaşılan en düşük nabız, 300 "
                    "denemenin ortalamasıyla karşılaştırılıyor -- ke/"
                    "duyarlılık ne kadar 'uç' örneklenirse, sonuç o kadar "
                    "ortalamadan sapar.")

    fig_trial, ax_trial = plt.subplots(figsize=(9, 3.2))
    ax_trial.plot(mc_for_observe.t, mc_for_observe.hr_runs.mean(axis=0),
                  color="gray", linewidth=1.5, linestyle="--", label="Popülasyon ortalaması (300 deneme)")
    ax_trial.plot(mc_for_observe.t, mc_for_observe.hr_runs[trial_idx],
                  color="#0E6E73", linewidth=2, label=f"Deneme #{trial_idx}")
    ax_trial.set_xlabel("Zaman (saat)")
    ax_trial.set_ylabel("Nabız (bpm)")
    ax_trial.legend()
    st.pyplot(fig_trial)

    st.divider()

    # --- Gerçek Kalp Modeliyle Göster (opsiyonel, CircAdapt) ---
    st.markdown("#### Gerçek Kalp Modeliyle Göster")
    st.caption(
        "CircAdapt (gerçek kalp-damar mekaniği motoru), yukarıdaki 200 "
        "PK/PD noktasının HER birinde değil -- her çalıştırması birkaç "
        "saniye sürdüğü için -- sadece İKİ referans an için çalıştırılır: "
        "ilaçsız (baseline) ve pik etki anı."
    )
    if st.button("🫀 Gerçek Kalp Modeliyle Göster"):
        with st.spinner("CircAdapt çalıştırılıyor..."):
            st.session_state["observe_heart"] = {
                "result": run_comparison(patient, drug),
                "inputs": current_inputs,
            }

    if "observe_heart" in st.session_state:
        oh = st.session_state["observe_heart"]
        if oh["inputs"] != current_inputs:
            st.warning("Girdiler değişti -- güncel sonuç için butona tekrar basın.")
        oh_heart = oh["result"]
        edv_b, esv_b = oh_heart["v_base"].max(), oh_heart["v_base"].min()
        edv_d, esv_d = oh_heart["v_drug"].max(), oh_heart["v_drug"].min()
        rows = pd.DataFrame([
            {
                "Referans An": "İlaçsız (baseline)",
                "Nabız (bpm)": oh_heart["hr_base"],
                "EDV (mL)": edv_b, "ESV (mL)": esv_b,
                "EF (%)": ejection_fraction(edv_b, esv_b),
            },
            {
                "Referans An": "Pik etki anı",
                "Nabız (bpm)": oh_heart["hr_drug_model"],
                "EDV (mL)": edv_d, "ESV (mL)": esv_d,
                "EF (%)": ejection_fraction(edv_d, esv_d),
            },
        ])
        st.dataframe(
            rows.style.format({
                "Nabız (bpm)": "{:.0f}", "EDV (mL)": "{:.1f}",
                "ESV (mL)": "{:.1f}", "EF (%)": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

# --- Sekme 6: Rapor İndir ---
with tab_report:
    st.subheader("Klinik Rapor Çıktısı (PDF)")
    if "sim" not in st.session_state:
        st.info("Önce 'Simülasyon' sekmesinde simülasyonu çalıştırın.")
    else:
        sim = st.session_state["sim"]
        st.caption(
            "Hasta bilgisi, ilaç bilgisi, simülasyon özeti, CircAdapt grafiği ve "
            "kaynakça içeren bir PDF raporu oluşturur. Rapor, klinik kullanım için "
            "olmadığını belirten silinemez bir uyarıyla her sayfada işaretlenir."
        )
        if sim["inputs"] != current_inputs:
            st.warning(
                "Hasta/ilaç girdilerini değiştirdiniz. Rapor hâlâ eski girdilerle "
                "hesaplanmış son simülasyonu yansıtacak."
            )

        heart = sim["heart_result"]
        fig_heart_report = build_comparison_figure(
            heart["t_base"], heart["p_base"], heart["v_base"],
            heart["t_drug"], heart["p_drug"], heart["v_drug"],
            sim["drug_name"], heart["hr_base"], heart["hr_drug_model"],
        )
        pdf_bytes = export_report(
            sim["patient"], sim["drug"], sim["dose_rec"], sim["mc_stats"],
            sim["heart_result"], chart_figures=[fig_heart_report],
        )
        st.download_button(
            "📥 PDF Raporu İndir",
            data=pdf_bytes,
            file_name=f"simulasyon_raporu_{sim['drug'].display_name.split()[0]}.pdf",
            mime="application/pdf",
            type="primary",
        )
