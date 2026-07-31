"""
patient_registry.py testleri -- gerçek dosya sistemine dokunmaz, her test
kendi izole tmp_path'inde çalışır.
"""

import json

import pytest

from patient_profile.patient_registry import (
    delete_patient_record,
    load_saved_patients,
    save_patient_record,
)


def test_load_saved_patients_empty_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert load_saved_patients(str(path)) == {}


def test_save_patient_record_creates_file_and_returns_all_records(tmp_path):
    path = tmp_path / "patients.json"
    fields = {"age": 70, "weight_kg": 82, "renal_function": 1.0}
    records = save_patient_record("Hasta A", fields, path=str(path))
    assert "Hasta A" in records
    assert records["Hasta A"]["age"] == 70
    assert records["Hasta A"]["weight_kg"] == 82
    assert path.exists()


def test_save_patient_record_persists_to_disk_across_loads(tmp_path):
    path = tmp_path / "patients.json"
    save_patient_record("Hasta A", {"age": 70}, path=str(path))
    reloaded = load_saved_patients(str(path))
    assert reloaded["Hasta A"]["age"] == 70


def test_save_patient_record_only_stores_known_fields(tmp_path):
    path = tmp_path / "patients.json"
    records = save_patient_record("Hasta A", {"age": 70, "unrelated_junk": "x"}, path=str(path))
    assert "unrelated_junk" not in records["Hasta A"]


def test_save_patient_record_missing_fields_become_none(tmp_path):
    path = tmp_path / "patients.json"
    records = save_patient_record("Hasta A", {"age": 70}, path=str(path))
    assert records["Hasta A"]["weight_kg"] is None
    assert records["Hasta A"]["comorbidity"] is None


def test_save_patient_record_upserts_same_name(tmp_path):
    path = tmp_path / "patients.json"
    save_patient_record("Hasta A", {"age": 70}, path=str(path))
    records = save_patient_record("Hasta A", {"age": 71}, path=str(path))
    assert records["Hasta A"]["age"] == 71
    assert len(records) == 1


def test_save_patient_record_rejects_empty_name(tmp_path):
    path = tmp_path / "patients.json"
    with pytest.raises(ValueError):
        save_patient_record("   ", {"age": 70}, path=str(path))


def test_save_patient_record_strips_name_whitespace(tmp_path):
    path = tmp_path / "patients.json"
    records = save_patient_record("  Hasta A  ", {"age": 70}, path=str(path))
    assert "Hasta A" in records
    assert "  Hasta A  " not in records


def test_save_patient_record_stores_source_and_timestamp(tmp_path):
    path = tmp_path / "patients.json"
    records = save_patient_record("Hasta A", {"age": 70}, path=str(path), source="pdf_extraction")
    assert records["Hasta A"]["source"] == "pdf_extraction"
    assert "saved_at_utc" in records["Hasta A"]


def test_save_patient_record_stores_extra_metadata_when_given(tmp_path):
    path = tmp_path / "patients.json"
    records = save_patient_record(
        "Hasta A", {"age": 70}, path=str(path), extra_metadata={"adjustment_log": ["x"]}
    )
    assert records["Hasta A"]["extra_metadata"] == {"adjustment_log": ["x"]}


def test_save_patient_record_omits_extra_metadata_when_not_given(tmp_path):
    path = tmp_path / "patients.json"
    records = save_patient_record("Hasta A", {"age": 70}, path=str(path))
    assert "extra_metadata" not in records["Hasta A"]


def test_delete_patient_record_removes_entry(tmp_path):
    path = tmp_path / "patients.json"
    save_patient_record("Hasta A", {"age": 70}, path=str(path))
    save_patient_record("Hasta B", {"age": 50}, path=str(path))
    records = delete_patient_record("Hasta A", path=str(path))
    assert "Hasta A" not in records
    assert "Hasta B" in records


def test_delete_patient_record_missing_name_is_noop(tmp_path):
    path = tmp_path / "patients.json"
    save_patient_record("Hasta A", {"age": 70}, path=str(path))
    records = delete_patient_record("Hasta Yok", path=str(path))
    assert "Hasta A" in records


def test_saved_file_is_valid_json_with_utf8_turkish_chars(tmp_path):
    path = tmp_path / "patients.json"
    save_patient_record("Öğüt Çelik", {"age": 70}, path=str(path))
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert "Öğüt Çelik" in raw
