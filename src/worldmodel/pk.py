"""
Farmakokinetik (PK / bkz. Sözlük) modülü — "İlaç vücutta nasıl dağılır
ve temizlenir?"

Bir-kompartmanlı model, kapalı-form çözüm kullanıyoruz:
İlaç emilir (ka -- emilim/dağılım hız sabiti, ilacın kana ne kadar
HIZLI karıştığının ölçüsü -- hızıyla) -> kana karışır -> eliminasyon
(ke -- eliminasyon hız sabiti, ilacın vücuttan ne kadar HIZLI
atıldığının ölçüsü -- hızıyla) ile atılır.

Hastanın KİLOSU burada devreye giriyor: dağılım hacmi (Vd -- ilacın
vücutta ne kadar YAYILDIĞININ ölçüsü) kiloyla ölçekleniyor, yani aynı
doz farklı kilodaki hastalarda farklı konsantrasyon üretiyor. Bu,
if-else yerine gerçek bir fiziksel bağıntı.
"""

import numpy as np


def plasma_concentration(t_hours: np.ndarray, dose_mg: float, ka: float,
                          ke: float, weight_kg: float, vd_per_kg: float,
                          dose_mg_per_kg: float | None = None) -> np.ndarray:
    """
    t_hours: zaman dizisi (saat)
    dose_mg: verilen doz (sabit mg) -- dose_mg_per_kg verilmezse kullanılır
    ka: emilim hız sabiti
    ke: eliminasyon hız sabiti (bireysel varyasyonla örneklenmiş)
    weight_kg: hastanın kilosu -> dağılım hacmini belirler
    vd_per_kg: ilaca özgü dağılım hacmi katsayısı
    dose_mg_per_kg: kilo bazlı doz (verilirse dose_mg yerine bunun
        weight_kg ile çarpımı kullanılır -- gerçek klinik dozlama
        genelde mg/kg'dır, sabit mg değil)

    Dönüş: zaman içindeki plazma konsantrasyonu (mg/L)
    """
    effective_dose_mg = dose_mg_per_kg * weight_kg if dose_mg_per_kg is not None else dose_mg
    vd = vd_per_kg * weight_kg
    conc = (effective_dose_mg / vd) * (ka / (ka - ke)) * (
        np.exp(-ke * t_hours) - np.exp(-ka * t_hours)
    )
    return np.clip(conc, 0, None)


def plasma_concentration_two_compartment(t_hours: np.ndarray, dose_mg: float,
                                          k10: float, k12: float, k21: float,
                                          vd_central: float,
                                          dose_mg_per_kg: float | None = None,
                                          weight_kg: float | None = None) -> np.ndarray:
    """
    Standart iki-kompartmanlı IV bolus PK çözümü (biexponential).

    Yukarıdaki plasma_concentration'dan farkı: ilaç doğrudan santral
    kompartmana (kana) enjekte edilir -- oral/absorpsiyon fazı (ka) yok,
    bunun yerine santral <-> periferik doku arasında geçiş var (k12, k21),
    santralden eliminasyon var (k10). Bu yüzden t=0'da konsantrasyon SIFIR
    DEĞİL, doğrudan dose/vd_central'dır -- gerçek bir IV bolusun fizyolojik
    davranışı budur (bkz. README.md > "Veri Kaynakları").

    t_hours: zaman dizisi (saat)
    dose_mg: verilen doz (sabit mg) -- dose_mg_per_kg verilmezse kullanılır
    k10: santral kompartmandan eliminasyon hız sabiti (1/saat)
    k12: santralden periferik kompartmana geçiş hız sabiti (1/saat)
    k21: periferikten santral kompartmana geri dönüş hız sabiti (1/saat)
    vd_central: santral kompartman dağılım hacmi (L) -- weight_kg verilirse
        bunun yerine vd_central bir "L/kg" katsayısı gibi ele alınıp
        weight_kg ile çarpılır (plasma_concentration'daki vd_per_kg ile
        tutarlılık için)
    dose_mg_per_kg: kilo bazlı doz (verilirse dose_mg yerine kullanılır)
    weight_kg: verilirse vd_central bir L/kg katsayısı olarak yorumlanır

    Dönüş: zaman içindeki santral (plazma) konsantrasyonu (mg/L)
    """
    effective_dose_mg = dose_mg_per_kg * weight_kg if dose_mg_per_kg is not None else dose_mg
    vc = vd_central * weight_kg if weight_kg is not None else vd_central

    sum_k = k10 + k12 + k21
    disc = sum_k ** 2 - 4 * k10 * k21
    alpha = (sum_k + np.sqrt(disc)) / 2
    beta = (sum_k - np.sqrt(disc)) / 2

    c0 = effective_dose_mg / vc
    a_coef = c0 * (alpha - k21) / (alpha - beta)
    b_coef = c0 * (k21 - beta) / (alpha - beta)

    conc = a_coef * np.exp(-alpha * t_hours) + b_coef * np.exp(-beta * t_hours)
    return np.clip(conc, 0, None)


def organ_function_adjusted_ke(ke_mean: float, renal_function: float = 1.0,
                                hepatic_function: float = 1.0,
                                renal_clearance_fraction: float = 0.0,
                                hepatic_clearance_fraction: float = 0.0) -> float:
    """
    Böbrek/karaciğer fonksiyon bozukluğunun eliminasyon hızına (ke) etkisini
    hesaplar. Mantık: toplam eliminasyonun sadece İLACA ÖZGÜ bir kısmı
    böbrek/karaciğer yoluyla oluyor (renal_clearance_fraction +
    hepatic_clearance_fraction, geri kalanı diğer yollar -- örn. esmolol
    kanda bulunan eritrosit esterazlarıyla [alyuvarların içindeki, ilacı
    parçalayan enzimlerle] metabolize olur, böbrek/karaciğerden bağımsız).
    Sadece o kısım, ilgili organın fonksiyon
    kaybıyla ORANTILI olarak azalır; geri kalan eliminasyon yolu etkilenmez.

    renal_function/hepatic_function: 1.0 = normal, 0.0 = tam yetmezlik.
    renal_clearance_fraction/hepatic_clearance_fraction: 0.0 = ilaç bu
        organdan hiç atılmıyor (etki YOK) -- örn. esmolol için ikisi de 0.0.

    Dönüş: ayarlanmış ke (1/saat). renal_clearance_fraction=
        hepatic_clearance_fraction=0.0 olan bir ilaçta, renal_function/
        hepatic_function ne olursa olsun ke DEĞİŞMEZ (esmololün gerçek
        davranışı budur -- bkz. README.md > Veri Kaynakları).
    """
    non_renal_hepatic_fraction = 1.0 - renal_clearance_fraction - hepatic_clearance_fraction
    remaining_fraction = (
        non_renal_hepatic_fraction
        + renal_clearance_fraction * renal_function
        + hepatic_clearance_fraction * hepatic_function
    )
    return ke_mean * remaining_fraction
