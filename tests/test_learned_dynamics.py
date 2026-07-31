"""
learned_dynamics paketi için testler (Faz 6) -- mevcut tests/test_pk.py'nin
sys.path kalıbını takip ediyor.

İki grup test:
  1. Hızlı, torch-gerektirmeyen şekil/normalizasyon sanity testleri (her
     zaman çalışır).
  2. "Küçük bir batch'i ezberleyebiliyor mu" testi -- pipeline'ın uçtan uca
     bozuk olmadığını (encoder/predictor/target-encoder/loss doğru
     bağlanmış mı) ucuza kanıtlayan standart bir sağlık testi. torch
     kurulu değilse otomatik atlanır (`pytest.importorskip`).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from worldmodel.learned_dynamics.state_repr import (
    build_patient_covariate_vector, build_state_vector, NormStats,
    STATE_DIM, PATIENT_COVARIATE_DIM, TRAJECTORY_N_POINTS,
)


def _sample_row():
    return {
        "age": 50.0, "weight_kg": 76.0, "height_cm": 175.0,
        "baseline_hr": 78.0, "baseline_sbp": 125.0, "baseline_dbp": 80.0, "baseline_spo2": 97.0,
        "renal_function": 1.0, "hepatic_function": 1.0,
        "potassium_mEqL": 4.25, "calcium_mgdL": 9.5, "comorbidity": "none",
    }


def test_patient_covariate_vector_shape_and_one_hot():
    vec = build_patient_covariate_vector(_sample_row())
    assert vec.shape == (PATIENT_COVARIATE_DIM,)
    # Son 3 boyut komorbidite one-hot -- "none" -> [1, 0, 0]
    assert np.allclose(vec[-3:], [1.0, 0.0, 0.0])


def test_patient_covariate_vector_unknown_comorbidity_falls_back_to_none():
    row = _sample_row()
    row["comorbidity"] = "beklenmedik_deger"
    vec = build_patient_covariate_vector(row)
    assert np.allclose(vec[-3:], [1.0, 0.0, 0.0])


def test_build_state_vector_shape():
    p = np.zeros(TRAJECTORY_N_POINTS)
    v = np.ones(TRAJECTORY_N_POINTS)
    covariates = build_patient_covariate_vector(_sample_row())
    state = build_state_vector(p, v, covariates)
    assert state.shape == (STATE_DIM,)
    assert state[TRAJECTORY_N_POINTS:2 * TRAJECTORY_N_POINTS].sum() == TRAJECTORY_N_POINTS  # v=1 bloğu


def test_norm_stats_roundtrip_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    state_matrix = rng.normal(loc=5.0, scale=2.0, size=(200, STATE_DIM))
    action_matrix = rng.normal(loc=0.5, scale=0.2, size=(200, 1))
    scalar_targets = {"ef": rng.normal(55, 5, 200), "co": rng.normal(5, 1, 200),
                       "hr": rng.normal(75, 10, 200), "edv": rng.normal(120, 10, 200),
                       "esv": rng.normal(50, 8, 200)}
    stats = NormStats.compute(state_matrix, action_matrix, scalar_targets)

    normalized = stats.normalize_state(state_matrix)
    assert abs(normalized.mean()) < 0.1
    assert abs(normalized.std() - 1.0) < 0.2

    value = 60.0
    round_tripped = stats.denormalize_scalar("ef", stats.normalize_scalar("ef", value))
    assert abs(round_tripped - value) < 1e-6


def test_norm_stats_save_load(tmp_path):
    rng = np.random.default_rng(1)
    stats = NormStats.compute(
        rng.normal(size=(50, STATE_DIM)), rng.normal(size=(50, 1)),
        {"ef": rng.normal(55, 5, 50), "co": rng.normal(5, 1, 50),
         "hr": rng.normal(75, 10, 50), "edv": rng.normal(120, 10, 50), "esv": rng.normal(50, 8, 50)},
    )
    path = str(tmp_path / "norm_stats.json")
    stats.save(path)
    loaded = NormStats.load(path)
    assert loaded.state_mean == stats.state_mean
    assert loaded.scalar_target_mean == stats.scalar_target_mean


# --- torch-gerektiren testler -------------------------------------------

torch = pytest.importorskip("torch")

from worldmodel.learned_dynamics.model import (  # noqa: E402
    Encoder, Predictor, update_target_encoder, DecoderHead, predict_next_embedding,
)


def test_encoder_predictor_forward_shapes():
    encoder = Encoder(state_dim=STATE_DIM, hidden_dim=16, embedding_dim=8)
    predictor = Predictor(embedding_dim=8, action_dim=1, hidden_dim=16)
    state = torch.randn(4, STATE_DIM)
    action = torch.randn(4, 1)

    embedding = encoder(state)
    assert embedding.shape == (4, 8)
    predicted_next = predictor(embedding, action)
    assert predicted_next.shape == (4, 8)


def test_predict_next_embedding_is_residual_not_absolute():
    """
    predict_next_embedding()'in gerçekten `embedding + predictor(...)`
    hesapladığını, predictor'ın çıktısını DOĞRUDAN döndürmediğini kanıtlar.
    Predictor'ı sıfır ağırlıklı yaparsak (çıktısı HER ZAMAN 0), sonraki
    embedding'in GİRDİ embedding'le BİREBİR AYNI olması gerekir -- bu, delta-
    tabanlı tasarımın "zayıf/etkisiz aksiyonda durum değişmez" özelliğinin
    doğrudan kanıtı (bkz. model.py > predict_next_embedding docstring'i).
    """
    predictor = Predictor(embedding_dim=8, action_dim=1, hidden_dim=16)
    for p in predictor.parameters():
        torch.nn.init.zeros_(p)

    embedding = torch.randn(4, 8)
    action = torch.randn(4, 1)
    next_embedding = predict_next_embedding(predictor, embedding, action)

    torch.testing.assert_close(next_embedding, embedding)


def test_decoder_head_forward_shape():
    decoder = DecoderHead(embedding_dim=8, hidden_dim=16)
    embedding = torch.randn(4, 8)
    out = decoder(embedding)
    assert out.shape == (4, 5)  # len(SCALAR_TARGET_FIELDS)


def test_update_target_encoder_moves_toward_online():
    online = Encoder(state_dim=STATE_DIM, hidden_dim=16, embedding_dim=8)
    target = Encoder(state_dim=STATE_DIM, hidden_dim=16, embedding_dim=8)
    # target'ı sıfırla, online'dan kasıtlı olarak FARKLI başlat.
    for p in target.parameters():
        torch.nn.init.zeros_(p)

    before = [p.clone() for p in target.parameters()]
    update_target_encoder(target, online, momentum=0.5)
    after = list(target.parameters())

    # momentum=0.5 ile target, online'a doğru yarı yolda olmalı (sıfırdan uzaklaşmış).
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_jepa_can_overfit_tiny_batch():
    """
    Ezber testi: 10 sahte örneklik bir batch'e, birkaç yüz adım boyunca
    predictor+encoder eğitilirse kayıp anlamlı ölçüde düşmeli. Bu, uçtan uca
    boru hattının (forward, backward, optimizer.step) bozuk olmadığını
    ucuza doğrular -- gerçek CircAdapt verisi GEREKMEZ.
    """
    torch.manual_seed(0)
    n, embedding_dim, hidden_dim = 10, 8, 16
    state = torch.randn(n, STATE_DIM)
    action = torch.randn(n, 1)
    next_state = torch.randn(n, STATE_DIM)

    encoder = Encoder(state_dim=STATE_DIM, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    predictor = Predictor(embedding_dim=embedding_dim, action_dim=1, hidden_dim=hidden_dim)
    target_encoder = Encoder(state_dim=STATE_DIM, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    target_encoder.load_state_dict(encoder.state_dict())

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2)

    losses = []
    for _ in range(300):
        predicted = predict_next_embedding(predictor, encoder(state), action)
        with torch.no_grad():
            target = target_encoder(next_state)
        loss = torch.nn.functional.mse_loss(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_target_encoder(target_encoder, encoder, momentum=0.9)
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"Kayıp yeterince düşmedi (başlangıç={losses[0]:.4f}, son={losses[-1]:.4f}) "
        "-- JEPA eğitim döngüsünde bir bağlantı hatası olabilir."
    )


# --- Faz 7 -- Transient (zaman-içi, otoregresif) uzantı testleri ----------

from transient_integration import compute_absolute_targets  # noqa: E402
from worldmodel.learned_dynamics.state_repr import TRANSIENT_STATE_DIM  # noqa: E402


def test_compute_absolute_targets_does_not_compound_on_nonmonotonic_sequence():
    """
    'Mutlak hedef' mantığının EN kritik gereksinimi: iniş-çıkış yapan
    (monoton OLMAYAN) bir hr_target dizisinde bile, her adımın sonucu
    SADECE o adımın kendi fraksiyonuna ve SABİT baseline'a bağlı olmalı --
    bir önceki adımın sonucuna DEĞİL. Çarpımsal bir biriktirme hatası
    olsaydı (apply_drug_effect_to_circadapt()'in yanlış kullanımı gibi),
    iniş-çıkış yapan bir dizide 3. adımın sonucu, 2. adımdan ETKİLENİRDİ --
    burada öyle olmadığını, doğrudan matematiksel eşitlikle kanıtlıyoruz.
    """
    baseline_t_cycle = 0.8
    baseline_sf_act = np.array([100.0, 100.0, 100.0])
    baseline_c_tau_av1 = 0.05

    # Kasıtlı monoton OLMAYAN dizi (iner, çıkar, tekrar iner) -- biriktirme
    # hatasını yakalamak için tasarlandı.
    hr_fractions = [0.9, 0.85, 0.9, 0.95]

    for hr_fraction in hr_fractions:
        t_cycle, sf_act, c_tau_av1 = compute_absolute_targets(
            baseline_t_cycle, baseline_sf_act, baseline_c_tau_av1, hr_fraction, sbp_fraction=1.0,
        )
        # HER adımda DOĞRUDAN baseline'dan hesaplanmalı -- bir önceki
        # adımın t_cycle/c_tau_av1 DEĞERİ hiç kullanılmadığı için, aynı
        # hr_fraction HER ZAMAN aynı mutlak sonucu vermeli (dizideki
        # konumundan BAĞIMSIZ).
        assert t_cycle == pytest.approx(baseline_t_cycle / hr_fraction)
        assert c_tau_av1 == pytest.approx(baseline_c_tau_av1 / hr_fraction)
        np.testing.assert_allclose(sf_act, baseline_sf_act * 1.0)

    # Ek kanıt: hr_fractions[0] ve hr_fractions[2] AYNI değer (0.9) -- eğer
    # biriktirme olsaydı, aralarında 2 adım (0.85 ve 0.9) geçtiği için
    # SONUÇLARI FARKLI olurdu. Burada BİREBİR AYNI olmalı.
    result_step0 = compute_absolute_targets(baseline_t_cycle, baseline_sf_act, baseline_c_tau_av1, 0.9, 1.0)
    result_step2 = compute_absolute_targets(baseline_t_cycle, baseline_sf_act, baseline_c_tau_av1, 0.9, 1.0)
    assert result_step0[0] == result_step2[0]
    assert result_step0[2] == result_step2[2]


def test_build_state_vector_transient_dim_with_current_hr():
    """current_hr verildiğinde TRANSIENT_STATE_DIM (215) boyutlu vektör
    üretilmeli -- STATE_DIM (214) DEĞİL. state_dim testleriyle (Faz 3)
    birlikte, geriye-uyumluluğun İKİ yönünü de (None -> 214, sayı -> 215)
    kapatıyor."""
    p = np.zeros(TRAJECTORY_N_POINTS)
    v = np.ones(TRAJECTORY_N_POINTS)
    covariates = build_patient_covariate_vector(_sample_row())

    state_with_hr = build_state_vector(p, v, covariates, current_hr=75.0)
    assert state_with_hr.shape == (TRANSIENT_STATE_DIM,)
    assert state_with_hr[-1] == pytest.approx(75.0)

    state_without_hr = build_state_vector(p, v, covariates)
    assert state_without_hr.shape == (STATE_DIM,)


def test_transient_dataset_produces_correct_pair_counts(tmp_path):
    """TransientDynamicsDataset'in, tam-uzunluklu VE kesilmiş (truncated)
    trajectory'lerden DOĞRU sayıda ardışık çift ürettiğini, sahte küçük bir
    npz fixture ile doğrular -- gerçek CircAdapt/torch GEREKMEZ (torch
    Dataset arayüzünü kullandığı için importorskip sınırının altında,
    torch zaten yukarıda import edildi)."""
    from worldmodel.learned_dynamics.transient_dataset import TransientDynamicsDataset

    n_traj0_frames = 5   # tam trajectory -> 4 ardışık çift
    n_traj1_frames = 3   # "kesilmiş" trajectory -> 2 ardışık çift
    n_total = n_traj0_frames + n_traj1_frames

    rng = np.random.default_rng(0)
    arrays = {
        "trajectory_id": np.array([0] * n_traj0_frames + [1] * n_traj1_frames, dtype=np.int64),
        "frame_idx": np.array(list(range(n_traj0_frames)) + list(range(n_traj1_frames)), dtype=np.int64),
        "patient_id": np.array([0] * n_traj0_frames + [1] * n_traj1_frames, dtype=np.int64),
        "split": np.array(["train"] * n_total),
        "dose_mg_per_kg": rng.uniform(0.1, 1.0, n_total).astype(np.float32),
        "elapsed_min": np.zeros(n_total, dtype=np.float32),
        "conc_mg_L": rng.uniform(0, 1, n_total).astype(np.float32),
        "current_hr": rng.uniform(50, 100, n_total).astype(np.float32),
        "age": rng.uniform(20, 80, n_total).astype(np.float32),
        "weight_kg": rng.uniform(50, 100, n_total).astype(np.float32),
        "height_cm": rng.uniform(150, 190, n_total).astype(np.float32),
        "baseline_hr": rng.uniform(50, 100, n_total).astype(np.float32),
        "baseline_sbp": rng.uniform(90, 180, n_total).astype(np.float32),
        "baseline_dbp": rng.uniform(60, 100, n_total).astype(np.float32),
        "baseline_spo2": rng.uniform(90, 100, n_total).astype(np.float32),
        "renal_function": np.ones(n_total, dtype=np.float32),
        "hepatic_function": np.ones(n_total, dtype=np.float32),
        "potassium_mEqL": np.full(n_total, 4.25, dtype=np.float32),
        "calcium_mgdL": np.full(n_total, 9.5, dtype=np.float32),
        "comorbidity": np.array(["none"] * n_total),
        "traj_p": rng.uniform(0, 120, (n_total, TRAJECTORY_N_POINTS)).astype(np.float32),
        "traj_v": rng.uniform(40, 130, (n_total, TRAJECTORY_N_POINTS)).astype(np.float32),
        "ef": rng.uniform(40, 70, n_total).astype(np.float32),
        "co": rng.uniform(3, 7, n_total).astype(np.float32),
        "edv": rng.uniform(90, 140, n_total).astype(np.float32),
        "esv": rng.uniform(30, 60, n_total).astype(np.float32),
    }
    npz_path = str(tmp_path / "fake_transient_dataset.npz")
    np.savez(npz_path, **arrays)

    ds = TransientDynamicsDataset(npz_path, split="train")
    assert len(ds) == (n_traj0_frames - 1) + (n_traj1_frames - 1)  # 4 + 2 = 6

    sample = ds[0]
    assert sample["state"].shape == (TRANSIENT_STATE_DIM,)
    assert sample["action"].shape == (1,)
