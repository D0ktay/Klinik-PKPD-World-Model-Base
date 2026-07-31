"""
Zaman-İçi (Transient) CircAdapt Sürücüsü
===========================================

`integrate_drug_with_circadapt.py`'nin tek-seferlik (baseline→ilaçlı sıçrama)
entegrasyonundan FARKLI olarak, burada aynı CircAdapt model nesnesi CANLI
tutulup, ilaç konsantrasyonu zamanla değişirken atım-atım (frame frame)
ilerletiliyor -- kalbin an be an devam eden hareketini üreten gerçek
zaman-serisi verisi bu modülden geliyor.

KRİTİK TASARIM KARARI -- "mutlak hedef" (ASLA `apply_drug_effect_to_circadapt()`
KULLANILMAZ): O fonksiyon parametreyi ÇARPIMSAL/GÖRELİ günceller (örn.
`c_tau_av1[0] = c_tau_av1[0] / hr_fraction` -- o anki değeri böler), tek
seferlik bir sıçrama için doğru. Burada her 2.5 dakikada bir yeniden
çağrılsaydı, etkiler YANLIŞLIKLA katlanarak birikirdi (ikinci çağrı,
birincinin zaten küçülttüğü değeri TEKRAR küçültür). Bunun yerine, ilaç
uygulanmadan HEMEN ÖNCEKİ referans değerler (`baseline_t_cycle`,
`baseline_sf_act`, `baseline_c_tau_av1`) BİR KERE saklanıyor, ve HER karede
o anki etkinin MUTLAK hedef değeri bu SABİT referanslardan yeniden
hesaplanıp DOĞRUDAN atanıyor -- bir önceki karenin değeri hiç okunmuyor.

Bu, izole bir Python script'i ile doğrulanmış iki CircAdapt davranışına
dayanıyor: (1) aynı `model` nesnesine art arda `model.run(n_beats=k,
stable=False)` çağırmak GERÇEKTEN önceki fiziksel durumdan devam ediyor
(kararlı bir modelde tekrar tekrar çağrıldığında EDV -- end-diastolic
volume, kalbin en dolu anındaki hacmi -- sabit kalıyor), (2)
`model["Solver"]["t"]` her çağrıda YEREL bir pencereye [0, t_cycle]
sıfırlanıyor -- geçen TOPLAM süreyi (elapsed_min) biz kendimiz Python
tarafında saymak zorundayız.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from circadapt.error import CircAdaptException

from integrate_drug_with_circadapt import (
    run_baseline, lv_pressure_volume, VENTRICLE_WALL_INDICES,
)
from worldmodel.pk import plasma_concentration
from worldmodel.pd import emax_effect, apply_effect_to_vitals

DEFAULT_WINDOW_MIN = 75.0
DEFAULT_FRAME_INTERVAL_MIN = 2.5

# Bu modül şu an SADECE bu iki ilaç sınıfının mekanizmasını (kontraktilite +
# AV iletim gecikmesi, bkz. apply_drug_effect_to_circadapt() dokümantasyonu)
# destekliyor -- "vasodilator" (damar direnci üzerinden etki eden) sınıfı
# bilinçli olarak kapsam dışı (MVP esmolol'e özel). Desteklenmeyen bir sınıf
# SESSİZCE yanlış davranmak yerine NotImplementedError fırlatır.
SUPPORTED_DRUG_CLASSES = ("beta_blocker", "positive_inotrope")


def compute_absolute_targets(baseline_t_cycle: float, baseline_sf_act: np.ndarray,
                              baseline_c_tau_av1: float, hr_fraction: float,
                              sbp_fraction: float) -> tuple[float, np.ndarray, float]:
    """
    "Mutlak hedef" formülünün SAF (yan etkisiz, CircAdapt'e hiç dokunmayan)
    hâli -- test edilebilirlik için `run_transient_trajectory()`'nin ana
    döngüsünden ayrıştırıldı. HER ZAMAN `baseline_*` parametrelerinden
    hesaplar, önceki bir çağrının SONUCUNU asla girdi olarak almaz -- bu
    imza şekli (yalnızca `baseline_*` + o anki fraksiyonlar) biriktirmenin
    (compounding) YAPISAL OLARAK imkansız olduğunu garanti eder: fonksiyon
    kendi önceki çıktısını bir sonraki çağrıya besleyebilecek bir parametre
    dahi ALMIYOR.

    Dönüş: (yeni_t_cycle, yeni_sf_act, yeni_c_tau_av1)
    """
    new_t_cycle = baseline_t_cycle / hr_fraction
    new_sf_act = baseline_sf_act * sbp_fraction
    new_c_tau_av1 = baseline_c_tau_av1 / hr_fraction
    return new_t_cycle, new_sf_act, new_c_tau_av1


class TransientTrajectoryResult:
    """`run_transient_trajectory()`'nin dönüş değeri -- kare listesi + kesilme bayrağı."""

    def __init__(self, frames: list[dict], truncated: bool):
        self.frames = frames
        self.truncated = truncated


def run_transient_trajectory(patient, drug, window_min: float = DEFAULT_WINDOW_MIN,
                              frame_interval_min: float = DEFAULT_FRAME_INTERVAL_MIN
                              ) -> TransientTrajectoryResult:
    if drug.drug_class not in SUPPORTED_DRUG_CLASSES:
        raise NotImplementedError(
            f"run_transient_trajectory: drug_class={drug.drug_class!r} desteklenmiyor "
            f"(sadece {SUPPORTED_DRUG_CLASSES}) -- 'vasodilator' MVP kapsamı dışı."
        )

    n_frames = round(window_min / frame_interval_min)

    # --- Adım 0: BOŞ bir modelden DEĞİL, run_baseline()'ın ürettiği ZATEN
    # yakınsamış modelden başla -- bu, hastanın GERÇEK dinlenim durumu
    # (elektrolit/komorbidite ayarlanmış, stable=True ile yakınsamış). Aksi
    # halde "baseline" referansları CircAdapt'in henüz yakınsamamış, keyfi
    # bir başlangıç durumundan alınırdı.
    model = run_baseline(patient)

    # --- Adım 1: Referans (ilaç-öncesi) değerleri AÇIKÇA kopyala. `float()`/
    # `np.array()` sarmalaması kasıtlı -- CircAdapt'in Parameter nesnelerinin
    # view mi copy mi döndürdüğü belgelenmemiş, belirsizliğe güvenmiyoruz.
    baseline_t_cycle = float(model["General"]["t_cycle"])
    baseline_sf_act = np.array(model["Patch"]["Sf_act"][VENTRICLE_WALL_INDICES])
    baseline_c_tau_av1 = float(model["Timings"]["c_tau_av1"][0])

    frames = []
    t0, p0, v0 = lv_pressure_volume(model)
    frames.append({
        "frame_idx": 0,
        "elapsed_min": 0.0,
        "conc_mg_L": 0.0,
        "current_hr": 60.0 / baseline_t_cycle,
        "t": t0, "p": p0, "v": v0,
    })

    # --- Adım 2: esmolol konsantrasyon eğrisini önceden hesapla (vektörize,
    # tek çağrı) -- sensitivity=1.0 SABİT (simulation.py::run_monte_carlo'daki
    # rastgele örneklemeyle KARIŞTIRILMASIN, burada kasıtlı bir sadeleştirme).
    frame_times_min = np.arange(1, n_frames + 1) * frame_interval_min
    conc_curve = plasma_concentration(
        frame_times_min / 60.0, drug.dose_mg, drug.ka, drug.ke_mean,
        patient.weight_kg, drug.vd_per_kg, dose_mg_per_kg=drug.dose_mg_per_kg,
    )

    truncated = False
    for i in range(n_frames):
        frame_idx = i + 1
        conc_t = float(conc_curve[i])

        effect_fraction = emax_effect(np.array([conc_t]), drug.ec50, sensitivity=1.0)
        hr_target, sbp_target = apply_effect_to_vitals(
            patient.baseline_hr, patient.baseline_sbp, effect_fraction,
            drug.emax_hr, drug.emax_sbp,
        )
        hr_fraction = float(hr_target[0]) / patient.baseline_hr
        sbp_fraction = float(sbp_target[0]) / patient.baseline_sbp

        # --- MUTLAK hedef ata -- HER ZAMAN baseline_* referanslarından,
        # bir önceki karenin değerinden DEĞİL (biriktirme yok, bkz.
        # compute_absolute_targets() docstring'i).
        new_t_cycle, new_sf_act, new_c_tau_av1 = compute_absolute_targets(
            baseline_t_cycle, baseline_sf_act, baseline_c_tau_av1, hr_fraction, sbp_fraction,
        )
        model["General"]["t_cycle"] = new_t_cycle
        sf_act = model["Patch"]["Sf_act"]
        sf_act[VENTRICLE_WALL_INDICES] = new_sf_act
        c_tau_av1 = model["Timings"]["c_tau_av1"]
        c_tau_av1[0] = new_c_tau_av1

        n_beats = max(1, round((frame_interval_min * 60.0) / model["General"]["t_cycle"]))

        try:
            model.run(n_beats=n_beats, stable=False)
        except CircAdaptException:
            truncated = True
            break

        t, p, v = lv_pressure_volume(model)
        frames.append({
            "frame_idx": frame_idx,
            "elapsed_min": frame_idx * frame_interval_min,
            "conc_mg_L": conc_t,
            "current_hr": 60.0 / float(model["General"]["t_cycle"]),
            "t": t, "p": p, "v": v,
        })

    return TransientTrajectoryResult(frames=frames, truncated=truncated)
