"""
İnteraktif demo — mülakatta canlı göstermek için.

Çalıştırma:
    streamlit run streamlit_app.py

Slider'larla kilo, doz, deneme sayısını değiştir, grafiğin anlık
değiştiğini göster -- "hastanın kilosu gerçekten sonucu değiştiriyor"
mesajını canlı kanıtlamanın en etkili yolu budur.

Sayfa tek uzun akış değil, 6 sekmeye ayrıldı (Hasta Kaydı / İlaç Seçimi /
Simülasyon / CircAdapt Sonuçları / Dünya Modelini Gözlemle / Rapor İndir)
-- her sekmenin widget'ları normal Python değişkenleri döndürür, bu yüzden
sekmeler arasında veri akışı (patient/drug/sim) değişmez.

"Hasta Kaydı" sekmesi hem hasta SEÇMEYİ (daha önce diskte -- patient_records/
saved_patients.json -- kaydedilmiş bir hastayı tekrar yükleme) hem de yeni
bir hasta OLUŞTURMAYI (PDF'ten otomatik çıkarım VE/YA DA elle slider girişi)
tek bir yerde birleştirir -- önceden ayrı bir "Hasta Girişi" sekmesi vardı,
kullanıcı isteği üzerine kaldırıldı (bkz. patient_registry.py).
"""

import sys
import os
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from worldmodel.patient import Patient, load_drugs, load_drug_interactions, load_drug_pk_interactions
from worldmodel.simulation import (
    run_monte_carlo, run_polypharmacy_simulation, run_polypharmacy_simulation_loewe,
    build_interaction_matrix, build_pk_interaction_matrix, summarize, recommend_dose, run_reference_trace,
    recommend_polypharmacy_dose_scale,
)
from worldmodel.viz import plot_results
from worldmodel.provenance import provenance_report, SOURCE_TYPE_LABEL
from worldmodel.report import export_report
from worldmodel.clinical_metrics import ejection_fraction, cardiac_output, classify_cardiac_function
from integrate_drug_with_circadapt import run_comparison, run_polypharmacy_comparison, build_comparison_figure
from circadapt.error import CircAdaptException
from patient_profile.file_ingestion import ingest_files, FileIngestionError
from patient_profile.llm_extraction import (
    extract_patient_profile_from_text, extract_patient_profile_from_pdf_bytes, LLMExtractionError,
)
from patient_profile.temporal_merge import merge_profiles
from patient_profile.review_data import build_review_screen, confirm_and_apply, REQUIRED_FIELD_NAMES
from patient_profile.schema import PatientCoreParameters, PatientProfile, ExtractedField
from patient_profile.ui_support import (
    widget_kind_for_field, select_options_with_unknown, select_index_for_value,
    hash_uploaded_files, build_confirmed_patient_params, resolve_conflicts_overridden_by_edits, UNKNOWN_OPTION,
)
from patient_profile.patient_registry import load_saved_patients, save_patient_record, delete_patient_record
from generate_dynamics_dataset import resample_trajectory
from transient_integration import run_transient_trajectory, SUPPORTED_DRUG_CLASSES
from worldmodel.learned_dynamics.model import Encoder, Predictor, DecoderHead, predict_next_embedding, get_device
from worldmodel.learned_dynamics.state_repr import (
    build_patient_covariate_vector, build_state_vector, NormStats, SCALAR_TARGET_FIELDS,
)

st.set_page_config(page_title="Medical Simulation", layout="wide")

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
    <div class="mark">MS</div>
    <div>
        <h1>Medical Simulation</h1>
        <p>PK/PD + CircAdapt kalp-damar mekaniği ile sanal hasta simülasyonu</p>
    </div>
    <div class="clinical-tag">Eğitim / Araştırma Amaçlı</div>
</div>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(__file__)
all_drugs = load_drugs(os.path.join(base_dir, "configs", "drugs.yaml"))
all_drug_interactions = load_drug_interactions(os.path.join(base_dir, "configs", "drug_interactions.yaml"))
all_pk_drug_interactions = load_drug_pk_interactions(os.path.join(base_dir, "configs", "drug_pk_interactions.yaml"))

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

# Onay ekranında bazı alanlar için schema.py'deki (uzun, parantez içi
# açıklamalı) description yerine kısa bir görünen ad -- açıklama yine de
# widget'ın help= metninde tam olarak gösterilir, kaybolmaz.
FIELD_DISPLAY_LABELS = {
    "sex": "Cinsiyet",
    "known_av_block_degree": "Bilinen AV Blok Derecesi",
}

# "Hasta Kaydı" sekmesindeki slider'ların MEVCUT (değiştirilmeyen) sabit
# varsayılanlarıyla birebir aynı değerler -- ne bir PDF onayı ne kayıtlı bir
# hasta seçiliyse buraya düşülür. AdjustedPatientParams'ın
# apply_patient_covariates() içindeki "hasta verisi yoksa" fallback'i de
# (base_pk_params) AYNI sayıları kullanır -- tek bir kaynak, iki ayrı yerde
# tutarsız varsayılan riski olmasın diye.
DEFAULT_BASE_PK_PARAMS = {
    "weight_kg": 76.0, "renal_function": 1.0, "hepatic_function": 1.0,
    "baseline_hr": 78.0, "baseline_sbp": 125.0, "baseline_dbp": 80.0,
    "potassium_mEqL": 4.25, "calcium_mgdL": 9.5,
}

# JEPA -- 1560-trajectory veriyle yeniden eğitilmiş, test setinde en iyi
# ölçülen kontrol noktası (bkz. logs/SUMMARY_1560.md, R²: ef=0.99 hr=0.97
# edv=0.94 esv=0.99). models/dynamics_jepa_transient_large (eski, 260-veri
# ile eğitilmiş kontrol noktası) BİLİNÇLİ OLARAK kullanılmıyor -- o dizin
# bu retraining turunda hiç güncellenmedi.
JEPA_MODEL_DIR = os.path.join(base_dir, "models", "dynamics_jepa_transient_1560run_1560data_seed0")
JEPA_WINDOW_MIN = 40.0
JEPA_FRAME_INTERVAL_MIN = 2.5
JEPA_SCALAR_LABELS = {
    "ef": ("Ejeksiyon Fraksiyonu", "%"), "co": ("Kardiyak Debi", "L/dk"),
    "hr": ("Nabız", "bpm"), "edv": ("LVEDV", "mL"), "esv": ("LVESV", "mL"),
}


@st.cache_resource
def load_jepa_bundle():
    """JEPA checkpoint'lerini (encoder/predictor/decoder + norm_stats) bir
    kez yükleyip Streamlit process'inin ömrü boyunca bellekte tutar --
    her buton tıklamasında diskten yeniden okumaz."""
    with open(os.path.join(JEPA_MODEL_DIR, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    device = get_device()
    encoder = Encoder(state_dim=cfg["state_dim"], hidden_dim=cfg["hidden_dim"],
                       embedding_dim=cfg["embedding_dim"]).to(device)
    predictor = Predictor(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    decoder = DecoderHead(embedding_dim=cfg["embedding_dim"], hidden_dim=cfg["hidden_dim"]).to(device)
    encoder.load_state_dict(torch.load(os.path.join(JEPA_MODEL_DIR, "encoder.pt"), map_location=device))
    predictor.load_state_dict(torch.load(os.path.join(JEPA_MODEL_DIR, "predictor.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(JEPA_MODEL_DIR, "decoder.pt"), map_location=device))
    encoder.eval()
    predictor.eval()
    decoder.eval()
    norm_stats = NormStats.load(os.path.join(JEPA_MODEL_DIR, "norm_stats.json"))
    return encoder, predictor, decoder, norm_stats, device


def run_jepa_rollout(frames: list[dict], patient) -> pd.DataFrame:
    """CircAdapt'in ürettiği GERÇEK kareleri (transient_integration.py ::
    run_transient_trajectory çıktısı) JEPA'nın state temsiline çevirir,
    frame 0'ın GERÇEK embedding'inden başlayarak otoregresif rollout
    çalıştırır -- rollout_evaluate.py ile BİREBİR AYNI protokol (gerçek
    veri araya hiç karışmaz, model kendi tahminini besler) -- ve gerçek/
    tahmin değerlerini tek bir DataFrame'de döndürür."""
    encoder, predictor, decoder, norm_stats, device = load_jepa_bundle()

    covariate_row = {
        "age": patient.age, "weight_kg": patient.weight_kg, "height_cm": patient.height_cm,
        "baseline_hr": patient.baseline_hr, "baseline_sbp": patient.baseline_sbp,
        "baseline_dbp": patient.baseline_dbp, "baseline_spo2": patient.baseline_spo2,
        "renal_function": patient.renal_function, "hepatic_function": patient.hepatic_function,
        "potassium_mEqL": patient.potassium_mEqL, "calcium_mgdL": patient.calcium_mgdL,
        "comorbidity": patient.comorbidity,
    }
    covariates = build_patient_covariate_vector(covariate_row)

    rows = []
    with torch.no_grad():
        embedding = None
        for frame in frames:
            p_r = resample_trajectory(frame["t"], frame["p"])
            v_r = resample_trajectory(frame["t"], frame["v"])
            edv_true, esv_true = float(v_r.max()), float(v_r.min())
            ef_true = ejection_fraction(edv_true, esv_true)
            co_true = cardiac_output(edv_true, esv_true, frame["current_hr"])
            true_vals = {"ef": ef_true, "co": co_true, "hr": frame["current_hr"],
                         "edv": edv_true, "esv": esv_true}

            if embedding is None:
                # frame_idx=0: rollout GERÇEK embedding'den başlar (rollout_evaluate.py
                # ile aynı) -- bu ilk karede "tahmin" de gerçek değerin kendisidir.
                state = build_state_vector(p_r, v_r, covariates, current_hr=frame["current_hr"])
                state_norm = norm_stats.normalize_state(state).astype(np.float32)
                embedding = encoder(torch.from_numpy(state_norm).unsqueeze(0).to(device))
                pred_vals = dict(true_vals)
            else:
                action_raw = np.array([[frame["conc_mg_L"]]], dtype=np.float32)
                action_norm = norm_stats.normalize_action(action_raw).astype(np.float32)
                action_t = torch.from_numpy(action_norm).to(device)
                embedding = predict_next_embedding(predictor, embedding, action_t)
                decoded_norm = decoder(embedding).cpu().numpy()[0]
                pred_vals = {
                    field: float(norm_stats.denormalize_scalar(field, decoded_norm[i]))
                    for i, field in enumerate(SCALAR_TARGET_FIELDS)
                }

            row = {"frame_idx": frame["frame_idx"], "elapsed_min": frame["elapsed_min"],
                   "conc_mg_L": frame["conc_mg_L"]}
            for field in SCALAR_TARGET_FIELDS:
                row[f"{field}_true"] = true_vals[field]
                row[f"{field}_pred"] = pred_vals[field]
            rows.append(row)

    return pd.DataFrame(rows)


tab_registration, tab_drug, tab_sim, tab_heart, tab_observe, tab_jepa, tab_report = st.tabs([
    "Hasta Kaydı", "İlaç Seçimi", "Simülasyon",
    "CircAdapt Sonuçları", "Dünya Modelini Gözlemle", "JEPA Dünya Modeli (Deneysel)", "Rapor İndir",
])

# --- Sekme 0: Hasta Kaydı ---
# Hasta SEÇMEYİ (diskte kalıcı -- patient_profile/patient_registry.py) ve
# hasta OLUŞTURMAYI (PDF'ten otomatik çıkarım VE/YA DA elle slider girişi)
# TEK sekmede birleştirir. Önceden ayrı bir "Hasta Girişi" sekmesi vardı --
# kullanıcı isteği üzerine kaldırıldı, slider'lar buraya taşındı.
NEW_PATIENT_OPTION = "+ Yeni Hasta"

with tab_registration:
    st.subheader("Hasta Kaydı")

    # Streamlit, bir widget'ın session_state anahtarının o widget AYNI
    # çalıştırmada zaten oluşturulduktan SONRA değiştirilmesine izin vermez
    # -- bu yüzden "patient_selector" widget'ı OLUŞTURULMADAN ÖNCE, bir
    # ÖNCEKİ rerun'dan (kaydetme/silme sonrası) bekleyen bir seçim varsa
    # burada uygulanır.
    pending_selection = st.session_state.pop("_pending_patient_selection", None)
    if pending_selection is not None:
        st.session_state["patient_selector"] = pending_selection

    saved_patients = load_saved_patients()
    patient_options = [NEW_PATIENT_OPTION] + sorted(saved_patients.keys())

    sel_col, del_col = st.columns([4, 1])
    with sel_col:
        selected_patient_name = st.selectbox(
            "Hasta", patient_options, key="patient_selector",
            help="'+ Yeni Hasta' -- sıfırdan bir hasta oluşturun. Listedeki bir "
                 "isim -- daha önce kaydedilmiş o hastanın TÜM bilgilerini "
                 "(yaş, kilo, elektrolitler, komorbidite vb.) yükler.",
        )
    with del_col:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        delete_clicked = st.button(
            "Seçili Hastayı Sil", disabled=(selected_patient_name == NEW_PATIENT_OPTION),
        )
    if delete_clicked and selected_patient_name != NEW_PATIENT_OPTION:
        delete_patient_record(selected_patient_name)
        st.session_state["_pending_patient_selection"] = NEW_PATIENT_OPTION
        st.success(f"'{selected_patient_name}' silindi.")
        st.rerun()

    # Seçim, bir ÖNCEKİ rerun'a göre DEĞİŞTİYSE, o eski seçimden kalma
    # (henüz kaydedilmemiş) bir PDF-onay override'ı varsa temizlenir --
    # aksi halde farklı bir hastaya geçince eski hastanın taze onayladığı
    # veriler yanlışlıkla yeni seçili hastaya sızardı.
    if st.session_state.get("_last_patient_selector_value") != selected_patient_name:
        st.session_state.pop("active_patient_override", None)
        st.session_state["_last_patient_selector_value"] = selected_patient_name

    if selected_patient_name != NEW_PATIENT_OPTION:
        active_saved_record = saved_patients[selected_patient_name]
        active_token = f"saved::{selected_patient_name}"
        source_label = "PDF çıkarımı" if active_saved_record.get("source") == "pdf_extraction" else "Elle giriş"
        st.caption(
            f"Kaynak: {source_label} -- kaydedilme: {active_saved_record.get('saved_at_utc', '?')}"
        )
    else:
        active_saved_record = None
        active_token = f"new::{st.session_state.get('new_patient_generation', 0)}"

    st.divider()
    st.markdown("#### PDF Hasta Girişi")

    uploaded_files = st.file_uploader(
        "Hasta raporu (PDF)", type=["pdf"], accept_multiple_files=True,
        help="Aynı hastaya ait birden fazla rapor (örn. farklı tarihli laboratuvar "
             "sonuçları) yüklenirse, alana göre en güncel tarihli değer kullanılır -- "
             "fizyolojik olarak anlamlı ölçüde çelişen değerler ayrıca işaretlenir.",
    )

    if not uploaded_files:
        st.info(
            "Rapor yüklenmedi -- bu adım isteğe bağlı, aşağıdaki 'Temel Bilgiler' "
            "alanlarını doğrudan elle doldurup hastayı kaydedebilirsiniz."
        )
    else:
        file_contents = [f.getvalue() for f in uploaded_files]
        current_hash = hash_uploaded_files(file_contents)
        cached = st.session_state.get("patient_profile_raw")

        if cached is None or cached["hash"] != current_hash:
            try:
                with st.spinner("Rapor(lar) işleniyor (metin çıkarımı + LLM analizi)..."):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_paths = []
                        for f in uploaded_files:
                            tmp_path = os.path.join(tmp_dir, f.name)
                            with open(tmp_path, "wb") as out:
                                out.write(f.getvalue())
                            tmp_paths.append(tmp_path)
                        ingested_docs = ingest_files(tmp_paths)

                        profiles = []
                        for doc in ingested_docs:
                            if doc.text is not None:
                                profiles.append(extract_patient_profile_from_text(doc.text, doc.filename))
                            else:
                                profiles.append(extract_patient_profile_from_pdf_bytes(doc.raw_bytes, doc.filename))

                    if len(profiles) > 1:
                        merged_profile, merge_conflicts = merge_profiles(profiles)
                    else:
                        merged_profile, merge_conflicts = profiles[0], []

                st.session_state["patient_profile_raw"] = {
                    "hash": current_hash,
                    "profile": merged_profile,
                    "conflicts": merge_conflicts,
                    "filenames": [f.name for f in uploaded_files],
                }
                # Yeni bir dosya seti -- önceki yüklemenin elle düzeltmeleri
                # bu YENİ hastaya ait olmayabilir, sıfırlanır.
                st.session_state["patient_profile_edits"] = {}
            except (FileIngestionError, LLMExtractionError) as e:
                st.error(f"Rapor işlenemedi: {e}")

        raw = st.session_state.get("patient_profile_raw")
        if raw is not None and raw["hash"] == current_hash:
            edits = st.session_state.get("patient_profile_edits", {})

            # Kullanıcının elle düzelttiği alanlar, ham profilin BİR KOPYASI
            # üzerinde uygulanır -- covariate_mapping'e giden veri, kullanıcı
            # gözden geçirmesinden geçmiş NİHAİ veridir (bkz. review_data.py
            # docstring'i: onaysız veri asla covariate_mapping'e gitmemeli).
            base_core = raw["profile"].core_parameters
            edited_core_fields = {}
            for field_name in PatientCoreParameters.model_fields:
                ef = getattr(base_core, field_name)
                if field_name in edits:
                    edited_core_fields[field_name] = ExtractedField(
                        value=edits[field_name], source_quote=ef.source_quote,
                        source_document=ef.source_document, source_date=ef.source_date,
                        confidence=ef.confidence,
                    )
                else:
                    edited_core_fields[field_name] = ef
            edited_profile = PatientProfile(
                core_parameters=PatientCoreParameters(**edited_core_fields),
                flags=raw["profile"].flags,
                extraction_metadata=raw["profile"].extraction_metadata,
            )

            active_conflicts = resolve_conflicts_overridden_by_edits(raw["conflicts"], edits)
            review_screen = build_review_screen("canli_demo_hastasi", edited_profile, active_conflicts)

            st.caption(f"Yüklenen dosyalar: {', '.join(raw['filenames'])}")

            if review_screen.conflicts:
                with st.container(border=True):
                    st.warning("Kaynaklar arasında çözülmemiş çelişki(ler) bulundu:")
                    for c in review_screen.conflicts:
                        st.markdown(f"- **{c.get('field', c.get('rule', '?'))}**: {c.get('detail', '')}")
                        for cand in c.get("candidates", []):
                            st.caption(
                                f"    {cand.get('value')} -- {cand.get('source_document')} "
                                f"({cand.get('source_date') or 'tarih yok'})"
                            )

            st.markdown("#### Çıkarılan Veriler (Gözden Geçir / Düzelt)")
            new_edits = dict(edits)
            for item in review_screen.items:
                field_info = PatientCoreParameters.model_fields[item.field_name]
                label = FIELD_DISPLAY_LABELS.get(item.field_name, field_info.description or item.field_name)
                kind = widget_kind_for_field(item.field_name)
                widget_key = f"profile_field_{current_hash}_{item.field_name}"
                col_widget, col_status = st.columns([3, 1])
                with col_widget:
                    if kind == "select":
                        options = select_options_with_unknown(item.field_name)
                        index = select_index_for_value(item.field_name, item.value)
                        selected = st.selectbox(label, options, index=index, key=widget_key,
                                                 help=field_info.description)
                        new_edits[item.field_name] = None if selected == UNKNOWN_OPTION else selected
                    elif kind == "checkbox":
                        checked = st.checkbox(
                            label, value=bool(item.value) if item.value is not None else False,
                            key=widget_key, help=field_info.description,
                        )
                        st.caption("Boş/işaretsiz bırakılırsa 'yok' kabul edilir.")
                        new_edits[item.field_name] = checked
                    else:
                        default_val = float(item.value) if isinstance(item.value, (int, float)) else 0.0
                        entered = st.number_input(label, value=default_val, key=widget_key,
                                                   help=field_info.description)
                        # number_input HER ZAMAN sayısal bir değer döndürür --
                        # None'ı temsil edemez. item.value zaten None (rapor bu
                        # alanı hiç içermiyor) VE kullanıcı widget'ı hiç
                        # DEĞİŞTİRMEDİYSE (hâlâ 0.0 seed varsayılanında), bunu
                        # "kullanıcı 0 girdi" ile KARIŞTIRMAYIP alanı missing
                        # bırakıyoruz -- aksi halde HER boş alan sessizce 0.0
                        # olurdu (örn. serum_creatinine=0 gibi geçersiz bir
                        # değerle Cockcroft-Gault'un çökmesine yol açan gerçek
                        # bir bug buydu). Bilinçli sınır: bir kullanıcı,
                        # GERÇEKTEN eksik bir alana kasıtlı olarak tam 0
                        # girmek isterse (örn. hepatic_encephalopathy_grade=0)
                        # bu, "dokunulmadı" ile ayırt edilemez -- alan yine
                        # missing kalır (yanlış-0 kabul etmekten daha güvenli
                        # bir taraf tutma).
                        if item.value is not None or entered != default_val:
                            new_edits[item.field_name] = entered
                    if item.source_quote:
                        st.caption(f"Kaynak: \"{item.source_quote}\" ({item.source_document or 'bilinmiyor'})")
                with col_status:
                    if item.validation_status == "ok":
                        if item.value is not None:
                            st.success("Doğrulandı")
                    elif item.validation_status == "out_of_range":
                        st.warning("Fizyolojik aralık dışı")
                    elif item.validation_status == "conflict":
                        st.warning("Kaynaklar arasında çelişki")
                    else:
                        st.caption("Bulunamadı")
            st.session_state["patient_profile_edits"] = new_edits

            missing_required = [
                name for name in REQUIRED_FIELD_NAMES
                if getattr(edited_profile.core_parameters, name).value is None
            ]
            if not review_screen.ready_to_confirm:
                reasons = []
                if missing_required:
                    reasons.append(f"eksik zorunlu alan(lar): {', '.join(missing_required)}")
                if review_screen.conflicts:
                    reasons.append("çözülmemiş çelişki(ler) var")
                st.caption(f"Onaylamak için: {'; '.join(reasons)}.")

            confirm_clicked = st.button(
                "Onayla", type="primary", disabled=not review_screen.ready_to_confirm,
                key="confirm_patient_profile_button",
            )
            if confirm_clicked:
                adjusted = confirm_and_apply(
                    review_screen, edited_profile, DEFAULT_BASE_PK_PARAMS, user_confirmed=True,
                )
                # active_patient_override: aşağıdaki "Temel Bilgiler" slider'ları
                # için TEK kaynak -- hangi hasta kimliği seçili olursa olsun
                # (yeni ya da kayıtlı bir hasta) bu onay, o kimliğin slider
                # varsayılanlarını GEÇİCİ olarak (kaydedilene kadar) ezer.
                st.session_state["active_patient_override"] = build_confirmed_patient_params(
                    adjusted, edited_profile.core_parameters,
                )
                if selected_patient_name == NEW_PATIENT_OPTION:
                    # Yeni bir hasta kimliği için taze slider widget'ları
                    # üretilsin diye (bkz. active_token), üretim sayacı artırılır.
                    st.session_state["new_patient_generation"] = (
                        st.session_state.get("new_patient_generation", 0) + 1
                    )
                st.success(
                    "Hasta verisi onaylandı -- aşağıdaki 'Temel Bilgiler' alanları artık "
                    "bu değerlerle başlıyor (yine de elle ince ayar yapılabilir, ve bu "
                    "hastayı bir isimle KAYDETMEDEN kalıcı olmaz)."
                )
                st.rerun()

    # active_patient_override (PDF onayından, henüz KAYDEDİLMEMİŞ) varsa
    # aşağıdaki slider varsayılanları için ÖNCELİKLİDİR; yoksa seçili
    # kayıtlı hastanın (varsa) diskteki değerleri kullanılır; o da yoksa
    # MEVCUT (değiştirilmeyen) sabit varsayılanlara düşülür -- hiçbir
    # slider KALDIRILMADI, sadece value= kaynağı koşullu hale geldi.
    override = st.session_state.get("active_patient_override")
    if override is not None:
        effective_defaults = override
        effective_adjustment_log = override.get("adjustment_log", [])
    elif active_saved_record is not None:
        effective_defaults = active_saved_record
        effective_adjustment_log = (active_saved_record.get("extra_metadata") or {}).get("adjustment_log", [])
    else:
        effective_defaults = None
        effective_adjustment_log = []

    if effective_adjustment_log:
        with st.expander("Uygulanan Ayarlamalar (PDF çıkarımından)"):
            for line in effective_adjustment_log:
                st.caption(line)

    def _patient_default(field_name, default, lo, hi, cast=float):
        if effective_defaults is None:
            return default
        val = effective_defaults.get(field_name)
        if val is None:
            return default
        return cast(max(lo, min(hi, val)))

    st.divider()
    st.markdown("#### Temel Bilgiler")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Yaş", 18, 95, _patient_default("age", 45, 18, 95, cast=int),
                         key=f"age_{active_token}",
                         help="Hastanın yaşı -- rapora ve klinik özet bilgilerine yansır; "
                              "ileri yaş, doz-yanıt yorumunda dikkate alınması gereken bir "
                              "faktördür.")
        weight = st.slider("Kilo (kg)", 40, 150, _patient_default("weight_kg", 76, 40, 150, cast=int),
                            key=f"weight_{active_token}",
                            help="Hastanın kilosu -- ilacın dağılım hacmini (Vd) belirler, "
                                 "bu yüzden aynı doz farklı kilodaki hastalarda farklı "
                                 "konsantrasyon üretir.")
    with c2:
        height = st.slider("Boy (cm)", 140, 210, _patient_default("height_cm", 175, 140, 210, cast=int),
                            key=f"height_{active_token}",
                            help="Hastanın boyu -- vücut yüzey alanı hesabında kullanılır.")
        baseline_hr = st.slider("Bazal nabız (bpm)", 50, 110, _patient_default("baseline_hr", 78, 50, 110, cast=int),
                                 key=f"baseline_hr_{active_token}",
                                 help="Nabız: kalbin dakikada kaç kez attığı (bpm = beats per "
                                      "minute). Bu, hastanın ilaçsız/dinlenme halindeki değeri "
                                      "-- CircAdapt simülasyonu da bu değere kalibre edilir.")
    with c3:
        baseline_sbp = st.slider("Bazal sistolik tansiyon (mmHg)", 90, 180,
                                  _patient_default("baseline_sbp", 125, 90, 180, cast=int),
                                  key=f"baseline_sbp_{active_token}",
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
                "Böbrek fonksiyonu (1.0=normal, 0=tam yetmezlik)", 0.0, 1.0,
                _patient_default("renal_function", 1.0, 0.0, 1.0), step=0.05,
                key=f"renal_function_{active_token}",
                help="Böbrek fonksiyonu bozulduğunda, SADECE böbrekten atılan ilaçların "
                     "(ör. digoksin) eliminasyonu yavaşlar -- böbrekten bağımsız ilaçlar "
                     "(ör. esmolol) etkilenmez.")
            potassium = st.slider(
                "Potasyum (mEq/L, normal 3.5-5.0)", 2.0, 8.0,
                _patient_default("potassium_mEqL", 4.25, 2.0, 8.0), step=0.05,
                key=f"potassium_{active_token}",
                help="Kandaki potasyum düzeyi. Hiperkalemi (>5.0, potasyumun normalden "
                     "yüksek olması) kalbin elektrik iletimini (AV düğümü -- üst ve alt "
                     "kalp odacıkları arasındaki sinyal iletimi) yavaşlatır.")
        with e2:
            hepatic_function = st.slider(
                "Karaciğer fonksiyonu (1.0=normal, 0=tam yetmezlik)", 0.0, 1.0,
                _patient_default("hepatic_function", 1.0, 0.0, 1.0), step=0.05,
                key=f"hepatic_function_{active_token}",
                help="Karaciğer fonksiyonu bozulduğunda, SADECE karaciğerden metabolize "
                     "olan ilaçların eliminasyonu yavaşlar.")
            calcium = st.slider(
                "Kalsiyum (mg/dL, normal 8.5-10.5)", 5.0, 13.0,
                _patient_default("calcium_mgdL", 9.5, 5.0, 13.0), step=0.05,
                key=f"calcium_{active_token}",
                help="Kandaki kalsiyum düzeyi. Kalsiyum, kontraktiliteyle (kalp kasının ne "
                     "kadar güçlü kasıldığı) doğru orantılıdır -- hipokalsemi (düşük "
                     "kalsiyum) kontraktiliteyi azaltır.")
        comorbidity_options_list = list(COMORBIDITY_OPTIONS.keys())
        default_comorbidity_value = effective_defaults.get("comorbidity") if effective_defaults else None
        default_comorbidity_label = next(
            (label for label, val in COMORBIDITY_OPTIONS.items() if val == default_comorbidity_value),
            comorbidity_options_list[0],
        )
        comorbidity_label = st.selectbox(
            "Komorbidite", comorbidity_options_list,
            index=comorbidity_options_list.index(default_comorbidity_label),
            key=f"comorbidity_{active_token}",
            help="Komorbidite: hastanın, bu simülasyondan BAĞIMSIZ olarak ZATEN sahip "
                 "olduğu kronik bir kalp hastalığı. Kalp yetmezliği = azalmış kontraktilite "
                 "(düşük EF). Hipertansiyon = kronik olarak yüksek damar direnci/basınç.")
        comorbidity = COMORBIDITY_OPTIONS[comorbidity_label]

    # known_av_block_degree: bu alan için bir slider/widget YOK (bkz. ADIM 0
    # inceleme notu) -- seçili kaynaktan (PDF onayı ya da kayıtlı hasta)
    # DOĞRUDAN (bir widget'tan geçmeden) taşınır, hiçbiri yoksa None kalır.
    active_known_av_block_degree = (
        effective_defaults.get("known_av_block_degree") if effective_defaults else None
    )

    st.divider()
    st.markdown("#### Bu Hastayı Kaydet")
    save_name_default = selected_patient_name if selected_patient_name != NEW_PATIENT_OPTION else ""
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        patient_name_input = st.text_input(
            "Hasta adı/etiketi", value=save_name_default, key=f"patient_name_input_{active_token}",
        )
    with save_col2:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        save_clicked = st.button("Kaydet", key=f"save_patient_button_{active_token}", type="primary")
    if save_clicked:
        if not patient_name_input.strip():
            st.warning("Kaydetmek için bir hasta adı girin.")
        else:
            fields = {
                "age": age, "weight_kg": weight, "height_cm": height,
                "baseline_hr": baseline_hr, "baseline_sbp": baseline_sbp,
                "renal_function": renal_function, "hepatic_function": hepatic_function,
                "potassium_mEqL": potassium, "calcium_mgdL": calcium,
                "comorbidity": comorbidity, "known_av_block_degree": active_known_av_block_degree,
            }
            came_from_pdf = override is not None or (active_saved_record or {}).get("source") == "pdf_extraction"
            save_patient_record(
                patient_name_input.strip(), fields,
                source="pdf_extraction" if came_from_pdf else "manual",
                extra_metadata={"adjustment_log": effective_adjustment_log} if effective_adjustment_log else None,
            )
            st.session_state["_pending_patient_selection"] = patient_name_input.strip()
            st.session_state.pop("active_patient_override", None)
            st.success(f"'{patient_name_input.strip()}' kaydedildi.")
            st.rerun()

# --- Sekme 2: İlaç Seçimi ---
with tab_drug:
    st.subheader("İlaç & Simülasyon Ayarları")
    drug_keys = st.multiselect(
        "İlaç(lar)",
        options=list(all_drugs.keys()),
        format_func=lambda k: (
            f"{all_drugs[k].display_name} "
            f"[{DRUG_CLASS_LABELS.get(all_drugs[k].drug_class, all_drugs[k].drug_class)}]"
        ),
        default=[list(all_drugs.keys())[0]],
        help="Beta-bloker: nabzı/kontraktiliteyi düşürür. Vazodilatör: damar direncini "
             "düşürür (kontraktiliteye dokunmaz). Pozitif inotrop: kontraktiliteyi artırır. "
             "Birden fazla ilaç seçilirse POLİFARMASİ modu çalışır -- ilaçların BİRLİKTE "
             "etkisi hem istatistiksel hem CircAdapt tarafında hesaplanır.",
    )
    if not drug_keys:
        st.warning("Devam etmek için en az bir ilaç seçin.")
        st.stop()

    n_runs = st.slider("Monte Carlo deneme sayısı", 50, 1000, 300, step=50,
                        help="Aynı hasta+ilaç kombinasyonunun kaç kez (her seferinde "
                             "bireysel varyasyon rastgele örneklenerek) simüle edileceği "
                             "-- daha fazla deneme, daha kararlı bir istatistik verir.")

    # Her seçilen ilaç için kendi doz/EC50 ayarları -- 2+ ilaç seçildiğinde
    # her biri kendi expander'ında, tek ilaçta ise doğrudan (eskisiyle aynı
    # görünüm) gösterilir.
    drugs: list = []
    for drug_key in drug_keys:
        base_drug = all_drugs[drug_key]
        container = st.expander(base_drug.display_name, expanded=(len(drug_keys) == 1)) \
            if len(drug_keys) > 1 else st.container()
        with container:
            # Her ilacın gerçek referans dozu çok farklı (esmolol ~38mg,
            # nikardipin ~2.3mg) -- slider aralığını seçili ilacın kendi
            # referans dozuna göre dinamik ayarlıyoruz.
            ref_dose = base_drug.dose_mg_per_kg * weight if base_drug.dose_mg_per_kg else base_drug.dose_mg
            dose = st.slider("Doz (mg)", round(ref_dose * 0.2, 2), round(ref_dose * 3.0, 2),
                              round(ref_dose, 2), key=f"dose_{drug_key}",
                              help="Verilen ilaç miktarı. Varsayılan değer, seçili ilacın kendi "
                                   "referans dozuna (genelde mg/kg cinsinden, hastanın kilosuyla "
                                   "ölçeklenmiş) göre ayarlanır.")
            # EC50 ilaca göre çok küçük olabilir (nikardipin: 0.0015 mg/L) --
            # sabit "%.2f" formatı bu durumda "0.00" gösterip değeri gizlerdi.
            ec50_min, ec50_max = float(base_drug.ec50 * 0.2), float(base_drug.ec50 * 5)
            ec50_step = (ec50_max - ec50_min) / 100
            ec50 = st.slider("EC50 (etki eşiği, mg/L)", ec50_min, ec50_max, float(base_drug.ec50),
                              step=ec50_step, format="%.5f", key=f"ec50_{drug_key}",
                              help="EC50: ilacın YARI-MAKSİMUM etkiyi yapması için gereken "
                                   "konsantrasyon. Düşük EC50, ilacın az miktarda bile etkili "
                                   "olduğu (güçlü/duyarlı) anlamına gelir.")
        # dose_mg_per_kg=None: slider'daki mutlak mg değerinin gerçekten
        # kullanılmasını garantiler -- aksi halde kilo bazlı dozlanan
        # ilaçlarda dose_mg_per_kg her zaman öncelikli olacağından slider
        # sessizce yok sayılırdı.
        drugs.append(replace(base_drug, dose_mg=dose, dose_mg_per_kg=None, ec50=ec50))

    if len(drug_keys) > 1:
        interaction_matrix = build_interaction_matrix(drug_keys, all_drug_interactions)
        if interaction_matrix:
            st.info(
                "Seçilen ilaçlar arasında bilinen bir etkileşim kaydı bulundu -- "
                "saf toplamsal (additive) modelin ÜSTÜNE ek bir sinerji terimi "
                "uygulanacak (bkz. configs/drug_interactions.yaml)."
            )

        # PK-seviyeli (klerens/AUC) ilaç-ilaç etkileşimi -- PD-seviyesindeki
        # interaction_matrix'ten AYRI, bkz. pk.py > pk_interaction_adjusted_ke().
        pk_interaction_matrix = build_pk_interaction_matrix(drug_keys, all_pk_drug_interactions)
        if pk_interaction_matrix:
            for (perpetrator, victim), auc_ratio in pk_interaction_matrix.items():
                record = next(
                    r for r in all_pk_drug_interactions
                    if r["perpetrator_drug"] == perpetrator and r["victim_drug"] == victim
                )
                st.info(
                    f"**PK-seviyeli (farmakokinetik) ilaç etkileşimi tespit edildi:** "
                    f"{all_drugs[perpetrator].display_name} ilacı, {all_drugs[victim].display_name} "
                    f"klerensini (ilacın vücuttan atılım hızını) ~{auc_ratio}x oranında etkiliyor "
                    f"(AUC [eğrinin altındaki alan -- toplam ilaç maruziyeti] artışı) -- "
                    f"kaynak: {record['source']}."
                )

        compare_with_loewe = st.checkbox(
            "Loewe additivity ile de karşılaştır (bilimsel literatürden, deneysel)",
            value=False,
            help="Varsayılan (toplamsal/additive) modelin YANINA, farmakoloji "
                 "literatüründe standart kabul edilen Loewe additivity (doz-"
                 "eşdeğerliği bazlı) bir hesap da çalıştırır ve iki sonucu "
                 "karşılaştırır. Seçilen ilaçların emax'ı (nabzı düşürme/"
                 "artırma yönü) AYNI olmalı -- aksi halde bu hesap atlanır.",
        )

        # ADIM 5 (N_DRUG_AUDIT.md/RESEARCH_N_DRUG.md): "hangi ilacın dozu
        # optimize edilsin" sorusu N=2'de de, N≥3'te de aynı klinik
        # belirsizlik -- eskiden SADECE N=2'de (drugs[0] sabit varsayımıyla)
        # cevaplanıyordu, N≥3'te bu panel TAMAMEN devre dışı kalıyordu.
        # Artık kullanıcı hangi ilacı optimize etmek istediğini AÇIKÇA
        # seçiyor, diğer TÜM ilaçlar "zaten kullanılan diğer ilaç(lar)"
        # rolünde sabit kalıyor -- recommend_dose()'un polypharmacy_result
        # mekanizması zaten N ilaca açık (bkz. simulation.py).
        target_drug_key = st.selectbox(
            "Doz önerisi hangi ilaç için hesaplansın?",
            options=drug_keys,
            format_func=lambda k: all_drugs[k].display_name,
            help="Seçilen ilacın dozu, DİĞER seçili ilaç(lar) sabit/mevcut "
                 "haliyle kullanılıyor varsayılarak optimize edilir -- "
                 "'hangi ilacın dozu ayarlansın' klinik bir tercih olduğu "
                 "için otomatik seçilmiyor.",
        )
    else:
        interaction_matrix = None
        pk_interaction_matrix = None
        compare_with_loewe = False
        target_drug_key = drug_keys[0]

patient = Patient(
    name="Canlı Demo Hastası", weight_kg=weight, height_cm=height,
    age=age, blood_type="A Rh+", baseline_hr=baseline_hr,
    baseline_sbp=baseline_sbp, baseline_dbp=80, baseline_spo2=97,
    renal_function=renal_function, hepatic_function=hepatic_function,
    potassium_mEqL=potassium, calcium_mgdL=calcium, comorbidity=comorbidity,
    known_av_block_degree=active_known_av_block_degree,
)

# Tek ilaç seçiliyken mevcut (eski) tek-ilaç kod yolu birebir korunur --
# `drug` değişkeni bu durumda hâlâ tek bir Drug nesnesi.
drug = drugs[0]

# Butona basıldığı an geçerli olan hasta+ilaç girdilerinin "parmak izi" --
# sonuçlar gösterilirken, ekrandaki girdiler bundan farklıysa kullanıcıyı uyarmak için.
current_inputs = (
    tuple(drug_keys), patient.age, patient.weight_kg, patient.height_cm, patient.baseline_hr,
    patient.baseline_sbp, patient.renal_function, patient.hepatic_function,
    patient.potassium_mEqL, patient.calcium_mgdL, patient.comorbidity,
    tuple((d.dose_mg, d.ec50) for d in drugs), n_runs, compare_with_loewe, target_drug_key,
)

# --- Sekme 3: Simülasyon ---
with tab_sim:
    if st.button("Simülasyonu Çalıştır", type="primary"):
        with st.spinner("Monte Carlo (PK/PD) simülasyonu çalıştırılıyor..."):
            if len(drugs) == 1:
                mc_result = run_monte_carlo(patient, drug, n_realizations=n_runs)
            else:
                mc_result = run_polypharmacy_simulation(
                    patient, drugs, n_realizations=n_runs, interaction_matrix=interaction_matrix,
                    drug_keys=drug_keys, pk_interaction_matrix=pk_interaction_matrix,
                )
            mc_stats = summarize(mc_result)

            # Loewe additivity -- kullanıcı checkbox'ı işaretlediyse, mevcut
            # (additive) sonucun YANINA gerçek doz-eşdeğerliği bazlı bir
            # hesap daha çalıştırılır (bkz. pd.py > loewe_combined_effect).
            # Sadece TÜM ilaçların emax'ı aynı yöndeyse anlamlı --
            # aksi halde ValueError yakalanıp kullanıcıya nazikçe gösterilir.
            loewe_stats = None
            loewe_error = None
            if len(drugs) > 1 and compare_with_loewe:
                try:
                    loewe_result = run_polypharmacy_simulation_loewe(patient, drugs, n_realizations=n_runs)
                    loewe_stats = summarize(loewe_result)
                except ValueError as e:
                    loewe_error = str(e)

        try:
            with st.spinner(
                "CircAdapt kalp simülasyonu çalıştırılıyor (baseline + ilaçlı) -- "
                "gerçek kalp-damar mekaniği modeli olduğu için bu adım birkaç saniyeden "
                "birkaç dakikaya kadar sürebilir, lütfen bekleyin..."
            ):
                if len(drugs) == 1:
                    heart_result = run_comparison(patient, drug)
                else:
                    heart_result = run_polypharmacy_comparison(patient, drugs)

            with st.spinner("En iyi doz önerisi hesaplanıyor (istatistiksel + mekanik risk)..."):
                if len(drugs) == 1:
                    dose_rec = recommend_dose(patient, drug, circadapt_results=heart_result)
                else:
                    # N≥2 GENEL hali (ADIM 5 -- eskiden SADECE N=2 çalışırdı):
                    # kullanıcının seçtiği hedef ilacın dozu, DİĞER TÜM
                    # seçili ilaç(lar) sabit/mevcut haliyle kullanılıyor
                    # varsayılarak öneriliyor -- recommend_dose() zaten
                    # "bu ilaç + diğerlerinin BİRLEŞİK riski" mantığıyla
                    # çalışıyor (polypharmacy_result=mc_result, N ilacın
                    # TAMAMININ birleşik simülasyonu -- N=2'de eski davranışla
                    # BİREBİR AYNI, sadece "diğer ilaç" artık TEK bir isim
                    # değil, hedef DIŞINDAKİ TÜM ilaçların isim listesi).
                    target_idx = drug_keys.index(target_drug_key)
                    target_drug = drugs[target_idx]
                    other_names = ", ".join(
                        d.display_name for i, d in enumerate(drugs) if i != target_idx
                    )
                    dose_rec = recommend_dose(
                        patient, target_drug, circadapt_results=heart_result,
                        polypharmacy_result=mc_result, polypharmacy_description=other_names,
                    )

            polypharmacy_scale_rec = None
            polypharmacy_scale_error = None
            if len(drugs) >= 2:
                # Hedef-ilaç önerisinin YANINDA, TÜM dozları kullanıcının
                # SEÇTİĞİ ORANI koruyarak ortak bir katsayıyla ölçekleyen
                # tamamlayıcı bir görünüm (bkz. simulation.py >
                # recommend_polypharmacy_dose_scale) -- N=2'de de N≥3'te de
                # yararlı bir ikinci bakış açısı, slider değerlerini
                # otomatik DEĞİŞTİRMEZ.
                with st.spinner("N-ilaçlı doz ölçeği önerisi hesaplanıyor (Loewe additivity)..."):
                    try:
                        polypharmacy_scale_rec = recommend_polypharmacy_dose_scale(
                            patient, drugs, n_realizations=150,
                            drug_keys=drug_keys, pk_interaction_matrix=pk_interaction_matrix,
                        )
                    except ValueError as e:
                        polypharmacy_scale_error = str(e)

            st.session_state["sim"] = {
                "dose_rec": dose_rec,
                "polypharmacy_scale_rec": polypharmacy_scale_rec,
                "polypharmacy_scale_error": polypharmacy_scale_error,
                "mc_result": mc_result,
                "mc_stats": mc_stats,
                "loewe_stats": loewe_stats,
                "loewe_error": loewe_error,
                "heart_result": heart_result,
                "drug_names": ", ".join(d.display_name for d in drugs),
                "drugs": drugs,
                "patient": patient,
                "inputs": current_inputs,
            }
        except CircAdaptException:
            st.error(
                "CircAdapt (gerçek kalp-damar mekaniği motoru), bu hasta/ilaç "
                "kombinasyonunda sayısal olarak kararsız hale geldi (fiziksel "
                "olarak aşırı bir senaryo -- ör. çok yüksek doz, uç bir "
                "komorbidite/elektrolit kombinasyonu). Bu, CircAdapt'in kendi "
                "çözücüsünün bir sınırı -- lütfen dozu, hastayı ya da ilacı "
                "değiştirip tekrar deneyin."
            )

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
        # ADIM 5: dose_rec artık N=1..8 ilaçta HER ZAMAN hesaplanıyor (kullanıcının
        # seçtiği hedef ilaç için, diğerleri sabit varsayılarak) -- eskiden N>2'de
        # None kalıp bu panel tamamen devre dışı kalıyordu.
        dose_rec = sim["dose_rec"]
        polypharmacy_scale_rec = sim.get("polypharmacy_scale_rec")
        polypharmacy_scale_error = sim.get("polypharmacy_scale_error")

        if dose_rec["mechanical_risk"]:
            st.error(f"**Önerilen doz: {dose_rec['dose_mg']:.1f} mg**\n\n{dose_rec['reasoning']}")
        elif dose_rec["is_safe"]:
            st.success(f"**Önerilen en iyi doz: {dose_rec['dose_mg']:.1f} mg**\n\n{dose_rec['reasoning']}")
        else:
            st.warning(f"**Önerilen doz: {dose_rec['dose_mg']:.1f} mg**\n\n{dose_rec['reasoning']}")

        # Ortak-ölçek görünümü (N≥2) -- hedef-ilaç önerisinin TAMAMLAYICISI,
        # ayrı/çakışan bir öneri değil: "tüm dozları AYNI oranda birlikte
        # ölçeklersem ne kadar güvenli" sorusuna cevap verir.
        if polypharmacy_scale_rec is not None:
            adjusted_lines = "\n".join(
                f"- **{name}**: {mg:.2f} mg" for name, mg in polypharmacy_scale_rec["adjusted_doses"].items()
            )
            box = st.success if polypharmacy_scale_rec["is_safe"] else st.warning
            with st.expander("Alternatif görünüm: tüm ilaçları ORTAK bir ölçekle ayarlamak", expanded=False):
                box(
                    f"**Önerilen doz ölçeği: mevcut oranın x{polypharmacy_scale_rec['scale']:.2f} katı**\n\n"
                    f"{polypharmacy_scale_rec['reasoning']}\n\n"
                    f"Buna göre önerilen dozlar (SLIDER'LAR OTOMATİK DEĞİŞMEDİ, bilgi amaçlıdır):\n{adjusted_lines}"
                )
        elif polypharmacy_scale_error is not None:
            st.info(
                f"Ortak-ölçek doz önerisi hesaplanamadı: {polypharmacy_scale_error} "
                "(muhtemelen ilaçların etki yönleri -- Emax işaretleri -- birbirinden farklı, "
                "bkz. pd.py > loewe_combined_effect / grouped_loewe_combined_effect)."
            )

        # --- PK/PD (Monte Carlo) sonuçları ---
        st.subheader("PK/PD Simülasyonu — Nabız & Tansiyon Dağılımı")
        fig_mc = plot_results(sim["mc_result"], patient, sim["drugs"][0], label=sim["drug_names"])
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

        # --- Loewe additivity karşılaştırması (opsiyonel, checkbox işaretliyse) ---
        if sim.get("loewe_error"):
            st.warning(
                f"Loewe additivity hesaplanamadı: {sim['loewe_error']}"
            )
        elif sim.get("loewe_stats") is not None:
            st.subheader("Toplamsal (Additive) vs. Loewe Additivity Karşılaştırması")
            st.caption(
                "İki farklı bilimsel varsayım, iki farklı sonuç üretebilir: 'Toplamsal' "
                "model her ilacın etkisini bağımsız hesaplayıp toplar/çarpar (basit, "
                "varsayılan). 'Loewe additivity' ise ilaçların doz-eşdeğerliğini "
                "hesaba katan, farmakoloji literatüründe standart kabul edilen bir "
                "yöntem -- özellikle ilaçların tavan etkisi (Emax) birbirinden "
                "UZAKSA, iki model FARKLI sonuç verebilir (bkz. Tallarida, "
                "'Quantitative Methods for Assessing Drug Synergism')."
            )
            cmp_card = st.container(border=True)
            c1, c2 = cmp_card.columns(2)
            c1.metric("Toplamsal model — ort. en düşük nabız",
                      f"{sim['mc_stats']['mean_min_hr']:.1f} bpm")
            c2.metric("Loewe additivity — ort. en düşük nabız",
                      f"{sim['loewe_stats']['mean_min_hr']:.1f} bpm",
                      delta=f"{sim['loewe_stats']['mean_min_hr'] - sim['mc_stats']['mean_min_hr']:+.1f} bpm")

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

        # run_comparison() (tek ilaç) "drug_effect" (tekil) döndürür,
        # run_polypharmacy_comparison() (N ilaç) "drug_effects" (liste) --
        # ikisini de tek bir listeye normalize edip aynı şekilde gösteriyoruz.
        drug_effects = [heart["drug_effect"]] if "drug_effect" in heart else heart["drug_effects"]
        for d, eff in zip(sim["drugs"], drug_effects):
            st.markdown(
                f"**{d.display_name} -- pik etki:** t={eff['t_peak_hours']:.2f} saat, "
                f"konsantrasyon={eff['conc_peak']:.3f} mg/L, "
                f"etki oranı={eff['effect_fraction']:.2f}"
            )

        if heart.get("av_block_triggered", False):
            # bkz. integrate_drug_with_circadapt.py > run_comparison/
            # run_polypharmacy_comparison: eşik aşıldığında CircAdapt hiç
            # çalıştırılmıyor (çökeceği için), p_drug/v_drug np.nan --
            # burada .max()/.min() ÇAĞRILMAZ, onun yerine bu durumu
            # açıklayan bir mesaj gösterilir.
            st.error(
                "**AV blok tetiklendi -- hemodinamik iz mevcut değil "
                f"(HR={heart['hr_drug_model']:.0f} bpm, kaçış ritmi).** "
                "Kümülatif AV iletim gecikmesi çarpanı, CircAdapt'in sayısal "
                "olarak çökeceği eşiği aştığı için gerçek bir basınç/hacim "
                "simülasyonu ÇALIŞTIRILMADI -- bkz. CALIBRATION_REPORT.md §8."
            )
        else:
            fig_heart = build_comparison_figure(
                heart["t_base"], heart["p_base"], heart["v_base"],
                heart["t_drug"], heart["p_drug"], heart["v_drug"],
                sim["drug_names"], heart["hr_base"], heart["hr_drug_model"],
            )
            st.pyplot(fig_heart)

            heart_metrics_card = st.container(border=True)
            h1, h2, h3, h4 = heart_metrics_card.columns(4)
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
            if "tau_av_drug_ms" in heart and heart["tau_av_drug_ms"] is not None:
                h4.metric(
                    "AV gecikmesi (PR aralığı benzeri)",
                    f"{heart['tau_av_drug_ms']:.0f} ms",
                    delta=f"{heart['tau_av_drug_ms'] - heart['tau_av_base_ms']:+.0f} ms",
                    help="AV düğümü (kalbin üst/alt odacıkları arasındaki elektrik "
                         "sinyal) iletim gecikmesi -- gerçek klinikte EKG'de PR "
                         "aralığı olarak görülür. Beta-bloker/pozitif inotrop "
                         "ilaçlar VE hiperkalemi bu AYNI parametreyi (Faz 5) hedefler "
                         "-- ikisi birlikteyken bu değer, ikisinin tek başına "
                         "ürettiğinden daha fazla artar. Aşırı uzaması (klinikte "
                         "genelde >200ms 1. derece AV blok sınırı kabul edilir) "
                         "gerçek hayatta AV blok riskinin erken uyarı işaretidir "
                         "(bkz. CALIBRATION_REPORT.md §5-6).",
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
        with st.expander("Bu sonuç neye dayanıyor?"):
            st.caption(
                "Bu simülasyonda kullanılan her parametrenin kaynağı -- "
                "Literatür (yayınlanmış kaynak) / Varsayım (yönü gerçek "
                "fizyolojiye dayanan ama kalibrasyon gerektiren temsili değer) / "
                "Hasta verisi (bu hasta için girilen değer). Tam detay ve "
                "kaynak referansları için CALIBRATION_REPORT.md."
            )
            st.table([
                {
                    "Kaynak Türü": SOURCE_TYPE_LABEL.get(row["source_type"], "Sınıflandırılmamış"),
                    "İlaç": d.display_name,
                    "Parametre": row["parameter"],
                    "Değer": row["value"],
                    "Kaynak": row["source_type"],
                    "Detay": row["detail"],
                }
                for d in sim["drugs"]
                for row in provenance_report(sim["patient"], d)
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
    if len(drugs) > 1:
        st.info(
            f"{len(drugs)} ilaç seçili -- bu sayfa (adım-adım tek iz gösterimi) "
            f"şu an sadece İLK seçilen ilaç için çalışıyor: **{drug.display_name}**. "
            "Kombinasyonun BİRLEŞİK sonucunu görmek için Simülasyon/CircAdapt "
            "sekmelerine bakın."
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
t = {t_sel:.2f} saat'teki DURUM:

   GİRDİ (bu adıma gelen durum): kalp hızı={hr_prev:.1f} bpm, tansiyon={sbp_prev:.1f} mmHg
   AKSİYON: {drug.display_name} {drug.dose_mg:.1f}mg (t=0'da verildi, şu an konsantrasyon={conc_sel:.4f} mg/L)
   HESAPLAMA: ilaç_etki_oranı = {effect_sel:.2f}  (Emax formülü: sensitivity * conc / (EC50 + conc))
   ÇIKTI (yeni durum): kalp hızı={hr_sel:.1f} bpm, tansiyon={sbp_sel:.1f} mmHg
```
""")

    st.divider()

    # --- Monte Carlo'nun "gizli" değerleri ---
    st.markdown("#### Monte Carlo Denemesini İncele")
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
    if st.button("Gerçek Kalp Modeliyle Göster"):
        try:
            with st.spinner("CircAdapt çalıştırılıyor..."):
                st.session_state["observe_heart"] = {
                    "result": run_comparison(patient, drug),
                    "inputs": current_inputs,
                }
        except CircAdaptException:
            st.error(
                "CircAdapt bu hasta/ilaç kombinasyonunda sayısal olarak "
                "kararsız hale geldi -- lütfen dozu, hastayı ya da ilacı "
                "değiştirip tekrar deneyin."
            )

    if "observe_heart" in st.session_state:
        oh = st.session_state["observe_heart"]
        if oh["inputs"] != current_inputs:
            st.warning("Girdiler değişti -- güncel sonuç için butona tekrar basın.")
        oh_heart = oh["result"]
        if oh_heart.get("av_block_triggered", False):
            st.error(
                "**AV blok tetiklendi -- hemodinamik iz mevcut değil "
                f"(HR={oh_heart['hr_drug_model']:.0f} bpm, kaçış ritmi).** "
                "Pik etki anı için gerçek bir basınç/hacim simülasyonu "
                "ÇALIŞTIRILMADI -- bkz. CALIBRATION_REPORT.md §8."
            )
        else:
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

# --- Sekme 5.5: JEPA Dünya Modeli (Deneysel) ---
# "Dünya Modelini Gözlemle" sekmesinden BİLİNÇLİ OLARAK AYRI: o sekme
# CircAdapt'ten TAMAMEN bağımsız, mekanistik PK/PD akışını açıklıyor
# (bkz. o sekmenin başındaki yorum). Bu sekme ise GERÇEKTEN öğrenilmiş
# (JEPA -- bkz. proje_detayli_anlatim.html Bölüm 8) bir sinir ağının
# tahminini, AYNI trajectory için CircAdapt'in GERÇEK sonucuyla
# karşılaştırıyor -- CircAdapt'in yerini ALMIYOR, sadece yanında,
# açıkça "deneysel" etiketiyle duruyor.
with tab_jepa:
    st.subheader("Öğrenilmiş (JEPA) Dünya Modeli — Deneysel")
    st.caption(
        "CircAdapt'in ÜRETTİĞİ verilerle eğitilmiş, atım-atım otoregresif "
        "bir sinir ağı (JEPA) -- kalbin bir sonraki anını, altındaki fiziği "
        "hiç bilmeden, sadece ÖRÜNTÜDEN öğrenerek tahmin etmeye çalışıyor. "
        "Aşağıda bu tahmin, AYNI trajectory için CircAdapt'in gerçekten "
        "hesapladığı değerle yan yana gösteriliyor."
    )
    st.warning(
        "**Bu model henüz üretim/karar-destek amaçlı DEĞİL.** Test setinde "
        "R² yüksek (EF/HR/EDV/ESV'de 0.93-0.99) ama mutlak hatada (MAE) "
        "çoğu metrikte 'hiçbir şey değişmedi' varsayımını (persistence "
        "baseline) henüz geçemiyor -- yani hastalar-arası GÖRELİ farkı iyi "
        "yakalıyor, ama CircAdapt'in yerini alacak kadar isabetli değil. "
        "Sadece araştırma/gösterim amaçlıdır. Detaylı deney sonuçları için "
        "proje_detayli_anlatim.html Bölüm 8 ve logs/SUMMARY_1560.md."
    )

    if drug.drug_class not in SUPPORTED_DRUG_CLASSES:
        st.info(
            f"JEPA modeli şu an sadece **beta-bloker** ve **pozitif inotrop** "
            f"sınıfı ilaçlarla eğitildi -- seçili ilaç "
            f"({DRUG_CLASS_LABELS.get(drug.drug_class, drug.drug_class)}) "
            "desteklenmiyor. 'İlaç Seçimi' sekmesinden uygun bir ilaç seçin."
        )
    else:
        if len(drugs) > 1:
            st.info(
                f"{len(drugs)} ilaç seçili -- bu sekme şu an sadece TEK ilaç için "
                f"çalışıyor: **{drug.display_name}**."
            )

        if st.button("JEPA ile 40 Dakikalık Tahmini Çalıştır", type="primary"):
            with st.spinner(
                "CircAdapt'ten gerçek, 40 dakikalık atım-atım referans trajectory "
                "üretiliyor (16 kare, her kare birkaç CircAdapt atımı) -- bu "
                "birkaç saniyeden bir dakikaya kadar sürebilir..."
            ):
                try:
                    traj_result = run_transient_trajectory(
                        patient, drug, window_min=JEPA_WINDOW_MIN,
                        frame_interval_min=JEPA_FRAME_INTERVAL_MIN,
                    )
                except CircAdaptException:
                    st.error(
                        "CircAdapt bu hasta/ilaç kombinasyonunda sayısal olarak "
                        "kararsız hale geldi -- lütfen dozu veya hastayı değiştirip "
                        "tekrar deneyin."
                    )
                    traj_result = None

            if traj_result is not None:
                if len(traj_result.frames) < 2:
                    st.error(
                        "Yeterli kare üretilemedi (en az 2 kare -- baseline + 1 "
                        "adım -- gerekli), rollout gösterilemiyor."
                    )
                else:
                    if traj_result.truncated:
                        st.warning(
                            "Trajectory, CircAdapt sayısal kararsızlığı nedeniyle "
                            "erken kesildi -- gösterilen adımlar tam 40 dakikayı "
                            "kapsamayabilir."
                        )
                    with st.spinner("JEPA modeliyle otoregresif rollout çalıştırılıyor..."):
                        chart_df = run_jepa_rollout(traj_result.frames, patient)
                    st.session_state["jepa_result"] = {
                        "chart_df": chart_df, "truncated": traj_result.truncated,
                        "patient_name": patient.name, "drug_name": drug.display_name,
                    }

    if "jepa_result" in st.session_state:
        jr = st.session_state["jepa_result"]
        chart_df = jr["chart_df"]

        st.divider()
        st.markdown(f"#### {jr['drug_name']} — {jr['patient_name']} için sonuçlar")

        plot_cols = st.columns(2)
        for i, field in enumerate(SCALAR_TARGET_FIELDS):
            label, unit = JEPA_SCALAR_LABELS[field]
            with plot_cols[i % 2]:
                fig, ax = plt.subplots(figsize=(5.5, 3.2))
                ax.plot(chart_df["elapsed_min"], chart_df[f"{field}_true"], "o-",
                        color="steelblue", label="CircAdapt (gerçek)")
                ax.plot(chart_df["elapsed_min"], chart_df[f"{field}_pred"], "s--",
                        color="crimson", label="JEPA (tahmin)")
                ax.set_xlabel("Zaman (dk)")
                ax.set_ylabel(f"{label} ({unit})")
                ax.set_title(label)
                ax.legend(fontsize=8)
                st.pyplot(fig)

        mae_summary = {
            field: float(np.mean(np.abs(chart_df[f"{field}_true"] - chart_df[f"{field}_pred"])))
            for field in SCALAR_TARGET_FIELDS
        }
        st.caption(
            "Bu TEK trajectory için ortalama mutlak hata (MAE): " +
            " · ".join(f"{JEPA_SCALAR_LABELS[k][0]}={v:.3f}" for k, v in mae_summary.items()) +
            " -- tek bir hasta/ilaç kombinasyonu istatistiksel olarak anlamlı "
            "değildir, sadece bu spesifik çalıştırmanın sonucu. Test setindeki "
            "26 trajectory üzerinden ölçülen genel model performansı için "
            "proje_detayli_anlatim.html Bölüm 8'e bakın."
        )

        with st.expander("Ham veri tablosu (16 kare, her karede gerçek vs tahmin)"):
            st.dataframe(
                chart_df.style.format({c: "{:.4f}" for c in chart_df.columns if c != "frame_idx"}),
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
            sim["drug_names"], heart["hr_base"], heart["hr_drug_model"],
        )
        pdf_bytes = export_report(
            sim["patient"], sim["drugs"], sim["dose_rec"] or {}, sim["mc_stats"],
            sim["heart_result"], chart_figures=[fig_heart_report],
        )
        file_slug = "_".join(d.display_name.split()[0] for d in sim["drugs"])
        st.download_button(
            "PDF Raporu İndir",
            data=pdf_bytes,
            file_name=f"simulasyon_raporu_{file_slug}.pdf",
            mime="application/pdf",
            type="primary",
        )
