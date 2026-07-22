# CLAUDE.md

Bu dosya, bu proje üzerinde çalışan Claude Code için proje-özel kurallar içerir.

## Proje

Mini Klinik Dünya Modeli — Vivax'ın Acudx mimarisinden ilham alan bir
kavram-kanıtı (proof-of-concept). PK/PD (farmakokinetik/farmakodinamik)
Monte Carlo simülasyonunu, CircAdapt (gerçek kalp mekaniği motoru) ile
birleştiriyor. Detaylı mimari ve terim açıklamaları için `README.md` ve
`CALIBRATION_REPORT.md`'ye bakın.

## Kurallar

### Yeni bir tıbbi terim eklerken mutlaka açıklama ekle

Bu projeyi sunan kişinin tıp geçmişi yok. **Yeni bir tıbbi/klinik terim
eklerken (kod yorumunda, docstring'de, README'de, ya da Streamlit
arayüzünde), o terimin İLK KULLANILDIĞI yerde mutlaka parantez içinde
sade, günlük dilde bir açıklama ekle.** Örnek format:

- "sistolik (kalbin kasılıp kan pompaladığı an)"
- "LV / sol karıncık (kalbin vücuda kan pompalayan ana odacığı)"
- "EF / ejeksiyon fraksiyonu (kalbin her atışta kanın yüzde kaçını pompaladığı)"
- "bradikardi (anormal derecede yavaş kalp atışı)"

Bu kural her zaman geçerli -- yeni bir özellik eklenirken de, mevcut
kodu düzenlerken de uygulanmalı. Yeni bir kavram kategorisi (yeni bir
mekanizma, yeni bir formül) eklendiyse, `README.md`'nin ilgili
bölümüne de kısaca ekle.

Streamlit widget'larında (slider, metric, selectbox), açıklamayı
`help=` parametresiyle ver -- kullanıcı "?" ikonunun üzerine gelince
görsün.

### CircAdapt parametrelerini asla tahmin etme

CircAdapt (gerçek kalp motoru) parametre isimlerini VE davranışlarını
asla tahmin etme -- önce `print(model['Bileşen'])` ile gerçek isimleri
keşfet, sonra "atama gerçekten kalıcı oluyor mu" sorusunu küçük, izole
bir Python betiğiyle (taze bir okuma ile) test ederek doğrula. Bu
projede birden fazla kez (`cLv` isimlendirmesi, `Sf_act`/`ArtVen.p0`
mutasyon bug'ı, `PFC`'nin `ArtVen` direncini nötralize etmesi,
`Timings.tau_av`'ın bir "law"dan yeniden hesaplanması) gerçek, sessizce
yanlış sonuç üreten hatalara yol açtı -- detaylar için README.md >
"Bulunan mühendislik hataları" bölümüne bakın.

### Her değişiklikten sonra testleri çalıştır

`python -m pytest tests/ -v` her önemli değişiklikten sonra çalıştırılmalı.
Mevcut davranışı (özellikle `app.py --patient hasta_a --drug beta_bloker`
CLI çıktısı) bozmadığını doğrula.
