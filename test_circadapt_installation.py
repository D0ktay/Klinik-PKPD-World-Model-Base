"""
CircAdapt Kurulum Testi
=======================

Bu script, CircAdapt'in gerçekten çalıştığını doğrular. VanOsta2024
modelini yükler (hazır bir kalp-damar-valf-dolaşım sistemi), bir kalp
atışı simüle eder ve sol karıncık basınç/hacim eğrisini çizer.

Bu, senin PK/PD modelinden TAMAMEN farklı bir motor -- burada artık
gerçek bir kalbin mekanik olarak nasıl kasıldığı, kapakçıkların ne
zaman açılıp kapandığı, kan basıncının nasıl oluştuğu simüle ediliyor.
"""

import matplotlib.pyplot as plt
from circadapt import VanOsta2024

print("VanOsta2024 modeli yükleniyor...")
model = VanOsta2024()

print("Model çalıştırılıyor (stabil duruma ulaşana kadar birkaç kalp atışı)...")
model.run(stable=True)

print("Simülasyon bitti, sonuçlar çiziliyor...")

# Zaman ekseni (saniye -> milisaniyeye çeviriyoruz, okunması kolay olsun diye)
t = model['Solver']['t'] * 1e3

# Sol karıncık (LV) basıncı (Pa -> mmHg'ye çeviriyoruz: 1 mmHg = 133.322 Pa)
lv_pressure = model['Cavity']['p'][:, 'cLv'] / 133.322

# Sol karıncık hacmi (m^3 -> mL'ye çeviriyoruz)
lv_volume = model['Cavity']['V'][:, 'cLv'] * 1e6

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(t, lv_pressure, color="crimson")
axes[0].set_xlabel("Zaman (ms)")
axes[0].set_ylabel("Sol karıncık basıncı (mmHg)")
axes[0].set_title("Sol Karıncık Basıncı — Tek Kalp Atışı")

axes[1].plot(lv_volume, lv_pressure, color="navy")
axes[1].set_xlabel("Sol karıncık hacmi (mL)")
axes[1].set_ylabel("Sol karıncık basıncı (mmHg)")
axes[1].set_title("Basınç-Hacim İlişkisi (PV Loop)\nKardiyoloji ders kitaplarındaki ünlü grafik")

plt.tight_layout()
plt.savefig("circadapt_test_sonucu.png", dpi=150)
print("\nBAŞARILI! Grafik kaydedildi: circadapt_test_sonucu.png")
print(f"Maksimum LV basıncı: {lv_pressure.max():.1f} mmHg (normal aralık: 100-140)")
print(f"Maksimum LV hacmi: {lv_volume.max():.1f} mL (normal aralık: 100-150)")
