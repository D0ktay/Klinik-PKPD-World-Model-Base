# Geliştirme Günlüğü

Bu dosya, projenin fazlar/görevler boyunca geçirdiği gelişimi, bulunan
hataları ve alınan kararları kısa ve dürüst biçimde kaydeder. Kod detayı
için `README.md`, kalibrasyon detayı için `CALIBRATION_REPORT.md`, kural
seti için `CLAUDE.md`'ye bakın.

## Faz 1-8 — Temel PK/PD + CircAdapt entegrasyonu

- Tek-kompartmanlı PK modeli (ka/ke/Vd), Emax PD modeli, Monte Carlo
  simülasyonu (`simulation.py`) kuruldu.
- CircAdapt (VanOsta2024) entegre edildi — ilaç etkisini gerçek kalp
  mekaniği motoruna bağlayan `integrate_drug_with_circadapt.py` yazıldı.
- **Bulunan hata**: `model[bileşen][parametre] = dizi` ataması, bazı
  parametrelerde (`Patch.Sf_act`, `ArtVen.p0`, `Timings.c_tau_av1`)
  yeniden atama SONRASI sessizce eski değere dönüyordu. Sadece yerinde
  (in-place) mutasyon kalıcı oluyor. Bu, CircAdapt'in iç `Parameter`
  nesnesinin çalışma şeklinden kaynaklanıyor.
- **Bulunan hata**: `PressureFlowControl` (PFC), `model.run(stable=True)`
  sırasında `ArtVen` direncini kendi sabit hedef basıncına göre yeniden
  ayarlıyor, elle yapılan `ArtVen.p0` değişikliklerini nötralize ediyordu.
  Akut ilaç etkileri için `PFC.is_active = False`, kronik durumlar
  (örn. hipertansiyon) için `PFC.p0`'ı da orantılı değiştirme çözümü
  benimsendi.
- Streamlit arayüzü kuruldu, buton durumu / sonuç kalıcılığı sorunu
  (`st.session_state` ile) çözüldü.

## Faz 9-15 — Doğrulama, klinik gerçekçilik, izlenebilirlik

- RxNorm + openFDA entegrasyonu ile gerçek ilaç verisi doğrulaması
  (`drug_lookup.py`), izlenebilirlik (provenance) alt yapısı kuruldu.
- **Bulunan hata**: openFDA'da boşluklu ilaç adı aramaları ("sodium
  nitroprusside") tırnaksız arandığında yanlış eşleşme ("sodium
  fluoride") döndürüyordu — tam ifade araması (tırnak) ile düzeltildi.
- İki-kompartmanlı PK modeli, etki bölgesi (effect-compartment/Keo)
  gecikme modeli, elektrolit (K+/Ca2+) ve komorbidite (kalp yetmezliği,
  hipertansiyon) etkileri eklendi.
- **Bulunan hata**: `Timings.tau_av` doğrudan atanamıyor — her adımda
  `c_tau_av0 + c_tau_av1 * t_cycle` formülünden yeniden hesaplanıyor
  ("law"). Çözüm: katsayı `c_tau_av1`'i mutasyona uğratmak.
- Polifarmasi (çoklu ilaç) desteği, PDF rapor üretimi (Türkçe karakter
  desteğiyle, `report.py`) tamamlandı.
- Klinik doğrulama testi yazılırken ilk varsayım (bolus modelinin
  infüzyon çalışmasıyla karşılaştırıldığında "başarısız olması
  beklenir") YANLIŞ çıktı (test beklenmedik şekilde geçti) — testin
  amacı, konsantrasyonun 60 dakikada gerçekten <%5 kaldığını doğrulayan
  daha sağlam bir yapıya çevrilerek düzeltildi.

## Ek Görev (Faz numaralarından bağımsız, 2026-07-22)

Kullanıcı, CircAdapt sonuçlarının hastanın kendi bazal nabzını hiç
yansıtmadığını fark etti ve üç görev tanımladı:

### Görev A — Bazal nabız senkronizasyon hatası

**Bulunan hata**: CircAdapt'in `General.t_cycle`'ı hiçbir zaman
`patient.baseline_hr`'a göre kalibre edilmiyordu — her zaman CircAdapt'in
kendi jenerik varsayılanından (~70.6 bpm) başlıyor, ilacın sadece
FRAKSİYONEL etkisi bunun üzerine uygulanıyordu. Yani hasta_a (78bpm) ve
hasta_b (85bpm) CircAdapt'te aynı bazal değerden başlıyordu.

**Çözüm**: `calibrate_circadapt_to_patient(model, patient)` fonksiyonu
eklendi (`integrate_drug_with_circadapt.py`), tüm giriş noktalarında
(`run_baseline`, `run_with_drug`, `run_with_multiple_drugs`) ilaç/
elektrolit/komorbidite etkilerinden ÖNCE çağrılıyor.

**Doğrulama**: Hem pytest testleriyle (`test_circadapt_baseline_reflects_
patients_own_baseline_hr`, `test_calibrate_circadapt_stable_across_
realistic_hr_range`) hem de doğrudan Python betiği çalıştırılarak hasta_a
ve hasta_b'nin artık gerçekten farklı bazal CircAdapt nabızları verdiği
doğrulandı. Playwright tarayıcı testinde de arayüzde "78 bpm" (hasta_a'nın
kendi bazal nabzı) görüldü.

### Görev B — EF / CO klinik metrikleri

`src/worldmodel/clinical_metrics.py` eklendi:
`ejection_fraction`, `cardiac_output`, `classify_cardiac_function`
(yeşil/sarı/kırmızı sınıflandırma, normal/hafif/düşük EF eşikleri, normal
CO aralığı 4-8 L/dk).

Streamlit'in "CircAdapt Sonuçları" sekmesine "Kalp Fonksiyonu
Değerlendirmesi" bölümü eklendi (baseline + ilaçlı EF/CO, renkli uyarı
kutuları). PDF rapora da EF/CO satırları eklendi.

**Doğrulama**: hasta_a (sağlıklı) EF ≈57%  (yeşil) vs hasta_e_kalp_
yetmezligi (heart_failure komorbiditesi) EF ≈41-44% (sarı) — gerçekten
düşük kontraktiliteyi yansıtan farklı EF değerleri üretildiği hem pytest
hem Playwright tarayıcı testiyle doğrulandı. CO'nun iki senaryoda benzer
kalması (~5.1 L/dk) kompanzatuar dilatasyon (artan EDV'nin düşen EF'yi
telafi etmesi) fizyolojisiyle tutarlı, ayrıca not edildi.

### Görev C — Sözlük ve sade dil açıklamaları

- `README.md`'ye ~40 terimlik alfabetik "Sözlük" bölümü eklendi (başlığın
  hemen altında, içindekiler tablosundan önce).
- `pk.py`, `pd.py`, `patient.py` içindeki tüm docstring/yorumlar gözden
  geçirildi, tıbbi terimlerin ilk geçtiği yerde sade açıklama eklendi.
- `streamlit_app.py`'daki tüm slider/metric/selectbox'lara `help=`
  parametresi eklendi.
- `CLAUDE.md` oluşturuldu — projeye özel üç kural: (1) yeni tıbbi terim
  eklerken her zaman sade açıklama ekleme kuralı, (2) CircAdapt
  parametrelerini asla tahmin etmeme kuralı, (3) her değişiklikten sonra
  testleri çalıştırma kuralı.

**Doğrulama**: Playwright ile arayüzde EF/CO bölümünün gerçekten
göründüğü, konsol hatası olmadığı doğrulandı (`ekgorev_01_ef_co.png`).

### Sonuç

67/67 test geçiyor. Görev A, B, C tamamlandı ve hem otomatik testlerle
hem tarayıcı üzerinden manuel doğrulamayla teyit edildi.

## UI Yenileme + "Dünya Modelini Gözlemle" sayfası (2026-07-22)

### Görsel yenileme

Streamlit'in varsayılan (mor, emoji-yoğun) görünümü klinik bir panele
çevrildi: `.streamlit/config.toml` ile koyu teal ana renk, "MK"
monogramlı kurumsal başlık banner'ı, metrik gruplarının kenarlıklı
kart konteynerlerine alınması, bölüm başlıklarındaki dekoratif
emojilerin kaldırılması. Ayrıca Hasta Girişi sekmesine gerçek bir
Yaş slider'ı eklendi (önceden `age=45` sabit kodlanmıştı).

### "Dünya Modelini Gözlemle" sayfası

Simülasyonun "kara kutusunu" açan yeni bir sekme eklendi -- amaç,
"durum + aksiyon -> yeni durum" akışını soyut bir kavram olmaktan
çıkarıp ekranda somut, tıklanabilir bir şeye çevirmek.

- **Akış diyagramı**: GİRDİ -> İŞLEME -> ÇIKTI üç kutucuğu (Vivax'ın
  Acudx mimarisine bilinçli bir referans).
- **Durum Tablosu**: `run_reference_trace()` (yeni, `simulation.py`)
  ile üretilen, gürültüsüz/tek bir PK/PD izinin HER zaman noktasındaki
  TÜM ara değerlerini (konsantrasyon, etki oranı, nabız, tansiyon)
  gösteren sıralanabilir bir tablo.
- **Tek Adımı İncele**: kullanıcı bir zaman noktası seçtiğinde, o ana
  ait GİRDİ/AKSİYON/HESAPLAMA/ÇIKTI zincirini okunabilir metin olarak
  gösteren slider kontrolü.
- **Monte Carlo Denemesini İncele**: `SimulationResult`'a eklenen
  `ke_values`/`sensitivity_values` alanları sayesinde (önceden hiç
  dışa açılmıyordu), kullanıcı 300 denemeden birini seçip o denemede
  RASTGELE örneklenen ke/duyarlılık değerlerini ve sonucu popülasyon
  ortalamasıyla karşılaştırabiliyor.
- **Gerçek Kalp Modeliyle Göster**: CircAdapt'i varsayılan olarak
  ÇALIŞTIRMAZ (sayfa varsayılan olarak hızlı PK/PD motoruyla çalışır)
  -- opsiyonel bir butonla tetiklenir ve sadece İKİ referans an
  (ilaçsız / pik etki anı) için EDV/ESV/EF karşılaştırması gösterir;
  200 PK/PD noktasının her birinde CircAdapt çalıştırmak dakikalar
  sürerdi, bu yüzden bilinçli bir mühendislik kararı olarak
  sınırlandırıldı.

**Doğrulama**: 4 yeni pytest testi (`run_reference_trace`'in
deterministik olduğu, hastanın bazal değerleriyle başladığı, etki
oranının sınırlı kaldığı; Monte Carlo'nun ke/sensitivity dizilerini
gerçekten dışa açtığı) + Playwright ile sayfanın CircAdapt olmadan da
tam çalıştığı, CircAdapt butonunun EDV/ESV/EF tablosunu doğru
ürettiği (örn. esmolol ile baseline EF %57.1 -> pik etki anında
%62.9, Frank-Starling mekanizmasıyla tutarlı) doğrulandı. 71/71 test
geçiyor.

**Not (test sürecinde bulunan, gerçek olmayan "hata")**: Playwright
testinde slider'a 30 ok tuşu art arda, bekleme olmadan basıldığında,
grafik etiketi bir an için eski deneme numarasını gösterdi (slider
değeri ile grafik arasında geçici tutarsızlık). Bu bir uygulama
hatası DEĞİL -- her tuş basışı 300 denemelik bir Monte Carlo'yu
yeniden hesaplattığı için, arka arkaya gelen 30 rerun kuyruğa
girdi. Tek tek, gerçekçi bir hızda yapılan etkileşimde (ve normal
kullanıcı kullanımında) slider/grafik/metrik her zaman tutarlı.
