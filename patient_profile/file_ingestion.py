# GÜVENLİK NOTU: Bu modül hassas hasta verisi işler.
# - Yüklenen PDF'ler ve çıkarılan veriler, harici LLM API'sine (Gemini)
#   gönderilir -- bu, verinin üçüncü taraf bir servise iletildiği anlamına gelir.
# - Prod ortamında: veri saklama süresi, şifreleme, erişim logları,
#   ve KVKK/GDPR uyumluluğu ayrıca değerlendirilmelidir (bu görevin kapsamı dışı).
# - Bu implementasyon bir demo/POC niteliğindedir, prodüksiyon güvenlik
#   gereksinimlerini karşılamak için ek çalışma gerekir.
"""
PDF dosya alım katmanı -- metin katmanı tespiti, metin çıkarma, taranmış
(scanned) PDF fallback stratejisi.

MİMARİ KARAR (ADIM 3'te istenen değerlendirme): taranmış PDF'ler için
YEREL OCR (pytesseract+pdf2image) yerine, dosyayı OLDUĞU GİBİ (ham PDF
byte'ları) Gemini'ye multimodal girdi olarak vermeyi TERCİH EDİYORUZ.
Gerekçe:
  1. Gemini (1.5+ / 2.x nesli) PDF'leri doğrudan (taranmış dahil) okuyabiliyor
     -- ayrı bir OCR adımına gerek kalmıyor.
  2. pytesseract, Python paketinin ötesinde sistem seviyesinde bir Tesseract
     OCR binary'si + pdf2image için Poppler gerektiriyor -- bu, kurulumu
     sessizce kırılgan hale getiren bir bağımlılık (bkz. README kurulum notu).
  3. Yerel OCR'ın metin çıktısı, source_quote'un ORİJİNAL cümleyle birebir
     eşleşmesini (llm_extraction.py'nin zorunlu kıldığı) OCR hatalarıyla
     bozma riski taşıyor; ham görüntüyü doğrudan LLM'e vermek bu riski
     azaltıyor (LLM görüntüyü kendi okuyor, aradaki OCR hata kaynağı ortadan
     kalkıyor).
`ocr_extract_text()` yine de burada duruyor -- Gemini API'sine erişimin
olmadığı ya da native PDF desteğinin başarısız olduğu durumlar için bir
YEDEK (fallback) yol, VARSAYILAN DEĞİL.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pdfplumber

# Sayfa başına bu karakter sayısının altında metin çıkarsa, PDF
# "muhtemelen taranmış" (metin katmanı yok/bozuk) kabul edilir.
SCANNED_TEXT_THRESHOLD_CHARS_PER_PAGE = 50

# Rapor metninde geçen tarihleri yakalamak için basit regex'ler --
# LLM'e BIRAKILMAZ, LLM extraction'ında ayrıca doğrulanır (bkz.
# llm_extraction.py prompt'u, madde: "RAPOR tarihini de doğrula").
_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"),  # 05.03.2026 / 05/03/2026
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),  # 2026-03-05 (ISO)
]


class FileIngestionError(Exception):
    """Bozuk PDF, desteklenmeyen format, boş dosya vb. için açık hata."""


@dataclass
class IngestedDocument:
    filename: str
    page_count: int
    extraction_method: str  # "text_layer" | "native_multimodal" | "ocr"
    text: str | None  # text_layer/ocr için dolu; native_multimodal için None
    raw_bytes: bytes | None  # native_multimodal/ocr-fallback için dolu
    detected_date: date | None
    warnings: list[str] = field(default_factory=list)


def _parse_date_match(match: re.Match, pattern_index: int) -> date | None:
    groups = match.groups()
    try:
        if pattern_index == 0:  # gün.ay.yıl
            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        else:  # ISO yıl-ay-gün
            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
        return date(year, month, day)
    except ValueError:
        return None


def extract_first_date(text: str) -> date | None:
    """
    Metindeki İLK geçen tarihi regex ile çıkarır (LLM'e bırakılmaz --
    bkz. modül docstring'i). Birden fazla format dener, ilk eşleşen ve
    geçerli (takvimde var olan) tarihi döndürür.
    """
    for pattern_index, pattern in enumerate(_DATE_PATTERNS):
        match = pattern.search(text)
        if match:
            parsed = _parse_date_match(match, pattern_index)
            if parsed is not None:
                return parsed
    return None


def extract_text_layer(pdf_path: str) -> tuple[str, int]:
    """
    pdfplumber ile PDF'in metin katmanını çıkarır.

    Dönüş: (metin, sayfa_sayısı). Bozuk/parse edilemeyen PDF için
    FileIngestionError fırlatır -- sessizce boş metin dönmez.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise FileIngestionError(f"'{pdf_path}': PDF sayfa içermiyor (boş dosya).")
            text_parts = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(text_parts), page_count
    except FileIngestionError:
        raise
    except Exception as exc:
        raise FileIngestionError(f"'{pdf_path}' açılamadı/parse edilemedi: {exc}") from exc


def is_scanned(text: str, page_count: int,
                threshold_chars_per_page: int = SCANNED_TEXT_THRESHOLD_CHARS_PER_PAGE) -> bool:
    """Sayfa başına düşen karakter sayısı eşiğin altındaysa 'taranmış' kabul eder."""
    if page_count == 0:
        return True
    return (len(text) / page_count) < threshold_chars_per_page


def ocr_extract_text(pdf_path: str) -> str:
    """
    YEDEK yol -- Gemini native PDF desteği kullanılamadığında devreye
    girer. pytesseract (Tesseract OCR binary gerektirir) + pdf2image
    (Poppler gerektirir) kullanır. Bu ikisi sistemde kurulu değilse,
    açık bir FileIngestionError fırlatır (sessizce boş metin dönmez).
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise FileIngestionError(
            "OCR fallback için 'pytesseract'/'pdf2image' paketleri kurulu değil "
            "(ayrıca sistem seviyesinde Tesseract OCR + Poppler gerektirir)."
        ) from exc

    try:
        images = convert_from_path(pdf_path)
    except Exception as exc:
        raise FileIngestionError(
            f"'{pdf_path}': PDF görüntüye çevrilemedi (Poppler kurulu mu?): {exc}"
        ) from exc

    try:
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as exc:
        raise FileIngestionError(f"'{pdf_path}': OCR başarısız (Tesseract kurulu mu?): {exc}") from exc


def ingest_pdf(pdf_path: str, use_native_multimodal_for_scanned: bool = True) -> IngestedDocument:
    """
    Tek bir PDF'i işler: metin katmanı var mı tespit eder, ona göre
    ("text_layer" | "native_multimodal" | "ocr") bir çıkarım yöntemi
    seçer, dosya metadata'sını (sayfa sayısı, tahmini tarih) doldurur.

    use_native_multimodal_for_scanned=True (varsayılan): taranmış PDF
    tespit edilirse, ham PDF byte'ları saklanır (llm_extraction.py bunu
    Gemini'ye multimodal girdi olarak verir) -- yerel OCR ÇALIŞTIRILMAZ.
    False verilirse (Gemini erişimi yoksa) yerel OCR fallback'i devreye
    girer.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileIngestionError(f"'{pdf_path}' bulunamadı.")
    if path.suffix.lower() != ".pdf":
        raise FileIngestionError(f"'{pdf_path}': desteklenmeyen dosya formatı (sadece PDF destekleniyor).")
    if path.stat().st_size == 0:
        raise FileIngestionError(f"'{pdf_path}': dosya boş.")

    text, page_count = extract_text_layer(pdf_path)
    warnings: list[str] = []

    if not is_scanned(text, page_count):
        return IngestedDocument(
            filename=path.name,
            page_count=page_count,
            extraction_method="text_layer",
            text=text,
            raw_bytes=None,
            detected_date=extract_first_date(text),
            warnings=warnings,
        )

    warnings.append(
        f"Sayfa başına ortalama {len(text) / max(page_count, 1):.0f} karakter -- "
        "muhtemelen taranmış (scanned) PDF, metin katmanı yetersiz."
    )

    if use_native_multimodal_for_scanned:
        raw_bytes = path.read_bytes()
        return IngestedDocument(
            filename=path.name,
            page_count=page_count,
            extraction_method="native_multimodal",
            text=None,
            raw_bytes=raw_bytes,
            detected_date=extract_first_date(text),  # zayıf metin katmanından best-effort
            warnings=warnings,
        )

    ocr_text = ocr_extract_text(pdf_path)
    return IngestedDocument(
        filename=path.name,
        page_count=page_count,
        extraction_method="ocr",
        text=ocr_text,
        raw_bytes=None,
        detected_date=extract_first_date(ocr_text),
        warnings=warnings,
    )


def ingest_files(pdf_paths: list[str], use_native_multimodal_for_scanned: bool = True) -> list[IngestedDocument]:
    """
    Çoklu dosya yüklemesini destekler -- her dosya AYRI AYRI işlenir
    (birleştirme temporal_merge.py'nin işi, burada değil). Bir dosya
    başarısız olursa, hangi dosyanın başarısız olduğunu belirten bir
    FileIngestionError fırlatılır -- diğer dosyalar sessizce atlanmaz.
    """
    if not pdf_paths:
        raise FileIngestionError("En az bir PDF dosyası verilmeli.")
    return [ingest_pdf(p, use_native_multimodal_for_scanned) for p in pdf_paths]
