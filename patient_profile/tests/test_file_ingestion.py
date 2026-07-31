"""
Dosya alım katmanı testleri -- tarih regex'i, taranmış-PDF tespiti,
ve desteklenmeyen/bozuk/boş dosya durumlarında SESSİZCE ATLAMA
yerine açık hata fırlatıldığını doğrular. Gerçek bir PDF motoru
gerektirmeyen kısımlar (regex, eşik mantığı, hata yolları) test edilir
-- pdfplumber'ın kendisi burada tekrar test edilmiyor (üçüncü parti
kütüphane).
"""

from datetime import date

import pytest

from patient_profile.file_ingestion import (
    FileIngestionError,
    extract_first_date,
    ingest_files,
    ingest_pdf,
    is_scanned,
)


def test_extract_first_date_dotted_format():
    assert extract_first_date("Rapor tarihi: 05.03.2026 hasta stabil") == date(2026, 3, 5)


def test_extract_first_date_iso_format():
    assert extract_first_date("2026-03-05 tarihli kontrol notu") == date(2026, 3, 5)


def test_extract_first_date_returns_none_when_absent():
    assert extract_first_date("bu metinde hiç tarih yok") is None


def test_extract_first_date_ignores_invalid_calendar_date():
    assert extract_first_date("32.13.2026 geçersiz tarih") is None


def test_is_scanned_true_for_empty_text():
    assert is_scanned("", page_count=1) is True


def test_is_scanned_false_for_dense_text():
    assert is_scanned("x" * 1000, page_count=1) is False


def test_is_scanned_true_for_sparse_text_relative_to_pages():
    assert is_scanned("kısa metin", page_count=5) is True


def test_ingest_pdf_raises_on_missing_file():
    with pytest.raises(FileIngestionError):
        ingest_pdf("bu_dosya_yok.pdf")


def test_ingest_pdf_raises_on_unsupported_extension(tmp_path):
    fake_file = tmp_path / "rapor.txt"
    fake_file.write_text("bu bir PDF değil")
    with pytest.raises(FileIngestionError):
        ingest_pdf(str(fake_file))


def test_ingest_pdf_raises_on_empty_file(tmp_path):
    empty_pdf = tmp_path / "bos.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(FileIngestionError):
        ingest_pdf(str(empty_pdf))


def test_ingest_files_raises_on_empty_list():
    with pytest.raises(FileIngestionError):
        ingest_files([])
