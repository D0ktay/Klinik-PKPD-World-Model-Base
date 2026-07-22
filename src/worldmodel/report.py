"""
Klinik Rapor Çıktısı

Simülasyon sonucunu, gerçek bir hastane raporu formatını taklit eden
(hasta bilgisi başlığı, grafik, özet, kaynakça) ama sayfanın en üstünde
BÜYÜK HARFLERLE, SİLİNEMEZ bir uyarıyla başlayan bir PDF'e aktarır.

ÖNEMLİ: Bu modül GERÇEK bir hastane/kurum kimliği TAŞIMAZ (başlıkta
sadece bu projenin adı geçer) ve DISCLAIMER metni hiçbir parametreyle
kaldırılamaz/gizlenemez -- her sayfanın hem üstünde hem altında görünür.
Bu, simülasyon çıktısının gerçek bir tıbbi belgeyle karıştırılmaması
için bilinçli bir tasarım kararı.
"""

import io
from datetime import datetime

from fpdf import FPDF

from .provenance import provenance_report
from .clinical_metrics import ejection_fraction, cardiac_output, classify_cardiac_function

DISCLAIMER = (
    "SİMÜLASYON SONUCU -- EĞİTİM / ARAŞTIRMA AMAÇLIDIR. "
    "KLİNİK KARAR ALMA İÇİN KULLANILAMAZ."
)

_ARIAL_REGULAR = r"C:\Windows\Fonts\arial.ttf"
_ARIAL_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


class _ClinicalReportPDF(FPDF):
    """FPDF alt sınıfı -- disclaimer'ı her sayfanın üstüne/altına
    OTOMATİK basar (header/footer hook'ları), çağıranın atlaması
    mümkün değildir."""

    def header(self):
        self.set_fill_color(200, 0, 0)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 10)
        self.cell(0, 9, DISCLAIMER, border=0, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "B", 8)
        self.set_text_color(200, 0, 0)
        self.cell(0, 5, DISCLAIMER, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Arial", "", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 4, f"Sayfa {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _add_unicode_fonts(pdf: _ClinicalReportPDF):
    """Türkçe karakterler (ı, ğ, ş, ç, ö, ü) için Unicode destekli bir TTF
    font yükler -- FPDF'in varsayılan çekirdek fontları (Helvetica/Arial
    core) Türkçe karakterleri desteklemiyor, bu yüzden gerçek bir Arial
    TTF'i (Windows sistem fontu) Unicode modunda ekliyoruz."""
    pdf.add_font("Arial", "", _ARIAL_REGULAR)
    pdf.add_font("Arial", "B", _ARIAL_BOLD)


def _matplotlib_fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def export_report(patient, drug, dose_rec: dict, mc_stats: dict,
                   heart_result: dict, chart_figures: list | None = None) -> bytes:
    """
    Simülasyon sonucunu PDF'e aktarır.

    patient, drug: bu simülasyonda kullanılan Patient/Drug nesneleri
    dose_rec: recommend_dose()'un döndürdüğü sözlük
    mc_stats: summarize()'ın döndürdüğü istatistiksel özet
    heart_result: run_comparison()'ın döndürdüğü CircAdapt sonucu
    chart_figures: PDF'e gömülecek matplotlib Figure nesneleri (opsiyonel)

    Dönüş: PDF dosyasının ham bytes'ı (Streamlit'te st.download_button'a
    doğrudan verilebilir, ya da open(path, "wb").write(...) ile diske
    yazılabilir).
    """
    pdf = _ClinicalReportPDF(format="A4")
    _add_unicode_fonts(pdf)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Mini Klinik Dünya Modeli — Simülasyon Raporu", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Oluşturulma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # --- Hasta Bilgileri ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Hasta Bilgileri", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 10)
    patient_rows = [
        ("Ad", patient.name),
        ("Kilo", f"{patient.weight_kg} kg"),
        ("Boy", f"{patient.height_cm} cm"),
        ("Yaş", str(patient.age)),
        ("Bazal nabız", f"{patient.baseline_hr} bpm"),
        ("Bazal sistolik tansiyon", f"{patient.baseline_sbp} mmHg"),
        ("Böbrek fonksiyonu", f"{patient.renal_function:.2f} (normal=1.0)"),
        ("Karaciğer fonksiyonu", f"{patient.hepatic_function:.2f} (normal=1.0)"),
        ("Potasyum", f"{patient.potassium_mEqL:.2f} mEq/L (normal 3.5-5.0)"),
        ("Kalsiyum", f"{patient.calcium_mgdL:.2f} mg/dL (normal 8.5-10.5)"),
        ("Komorbidite", patient.comorbidity or "Yok"),
    ]
    for label, value in patient_rows:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(55, 6, label, new_x="RIGHT")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- İlaç Bilgileri ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "İlaç Bilgileri", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 10)
    drug_rows = [
        ("İlaç", drug.display_name),
        ("Sınıf", drug.drug_class or "-"),
        ("Uygulanan doz", f"{drug.dose_mg:.2f} mg"),
    ]
    for label, value in drug_rows:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(55, 6, label, new_x="RIGHT")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Simülasyon Özeti ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Simülasyon Özeti", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 10)
    edv_drug = heart_result["v_drug"].max()
    esv_drug = heart_result["v_drug"].min()
    ef_drug = ejection_fraction(edv_drug, esv_drug)
    co_drug = cardiac_output(edv_drug, esv_drug, heart_result["hr_drug_model"])
    cls_drug = classify_cardiac_function(ef_drug, co_drug)

    summary_rows = [
        ("Önerilen doz", f"{dose_rec['dose_mg']:.1f} mg"),
        ("Bradikardi riski (anormal derecede yavaş nabız riski)", f"%{mc_stats['pct_bradycardia_risk']:.1f}"),
        ("Ortalama en düşük nabız", f"{mc_stats['mean_min_hr']:.1f} bpm"),
        ("Mekanik risk (CircAdapt)", "Var" if dose_rec.get("mechanical_risk") else "Yok"),
        ("CircAdapt nabzı (ilaçlı)", f"{heart_result['hr_drug_model']:.0f} bpm"),
        ("Maksimum LV basıncı (ilaçlı)", f"{heart_result['p_drug'].max():.1f} mmHg"),
        ("LVEDV / diyastol sonu hacim (ilaçlı)", f"{edv_drug:.1f} mL"),
        ("EF / ejeksiyon fraksiyonu (ilaçlı)", f"{cls_drug['ef_label']} ({cls_drug['ef_color']})"),
        ("CO / kardiyak output (ilaçlı)", f"{cls_drug['co_label']} ({cls_drug['co_color']})"),
    ]
    for label, value in summary_rows:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(65, 6, label, new_x="RIGHT")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, dose_rec.get("reasoning", ""))
    pdf.ln(4)

    # --- Grafikler ---
    if chart_figures:
        for fig in chart_figures:
            png_bytes = _matplotlib_fig_to_png_bytes(fig)
            pdf.image(io.BytesIO(png_bytes), w=pdf.epw)
            pdf.ln(4)

    # --- Kaynakça (sadece "literatür" olarak işaretlenen parametreler) ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Kaynakça (Bu İlaç İçin Literatür Kaynaklı Parametreler)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 9)
    report_rows = provenance_report(patient, drug)
    literature_rows = [r for r in report_rows if r["source_type"] == "literatür"]
    if literature_rows:
        for row in literature_rows:
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 5, f"{row['parameter']} = {row['value']}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5, row["detail"])
            pdf.ln(1)
    else:
        pdf.multi_cell(0, 5, "Bu ilaç için literatür kaynaklı olarak işaretlenmiş parametre bulunamadı.")

    pdf.ln(4)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 5, "Not:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(
        0, 5,
        "Bu raporda yer almayan tüm diğer parametreler (emax/ec50/keo gibi "
        "doz-yanıt büyüklükleri) TEMSİLİDİR, gerçek bir doz-yanıt "
        "çalışmasından kalibre edilmemiştir. Tam kaynak listesi için "
        "CALIBRATION_REPORT.md'ye bakın."
    )

    return bytes(pdf.output())
