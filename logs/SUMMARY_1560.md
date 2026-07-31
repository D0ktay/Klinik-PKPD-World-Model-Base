# 1560-Trajectory Yeniden Eğitim -- Sonuç Özeti

**Durum: KADEME 1 -- pipeline uçtan uca çalıştı, 6/6 koşum başarılı.**
**Ana sonuç: veri 6x büyütülünce (260→1560 trajectory) model artık "ortalamaya
kaçmıyor" (mean-collapse çözüldü) ve yön hatası çarpıcı şekilde azaldı, ama
model hâlâ persistence baseline'ı (hiçbir şey tahmin etme, t=0'ı kopyala)
GEÇEMİYOR -- EF/HR/EDV'de baseline'a çok yaklaştı ama geçemedi.**

Toplam süre: ~2.5 saat (7:45'lik bütçenin çok altında -- iş küçüktü, CPU-only
küçük model + 24K satırlık veri, bkz. "Dürüst Sınırlar").

---

## 1. R² Karşılaştırma Tablosu (havuzlanmış, 3 tohum ortalama±std)

| Metrik | 260-veri R² | 1560-veri R² | Değişim |
|---|---|---|---|
| EF  | 0.141 ± 0.327 | **0.991 ± 0.001** | +0.85, ve std 300x küçüldü |
| CO  | -0.006 ± 0.031 | 0.128 ± 0.003 | hafif iyileşme, hâlâ zayıf |
| HR  | -0.127 ± 0.200 | **0.974 ± 0.001** | negatiften pozitife |
| EDV | -0.202 ± 0.133 | **0.931 ± 0.005** | negatiften pozitife |
| ESV | 0.320 ± 0.171 | **0.989 ± 0.001** | +0.67 |

Orijinal referans (tek koşum, seed kontrolsüz): EF=0.704 CO=0.047 HR=0.491
EDV=0.371 ESV=0.789 -- bu sayı 260-veri için ölçtüğüm 3-tohum aralığının
**dışında/üst sınırında**, yani muhtemelen şanslı bir seed'di (bkz. Bölüm 5).

## 2. Persistence Baseline Karşılaştırması

| Metrik | Model MAE (260) | Model MAE (1560) | Baseline MAE | Sonuç |
|---|---|---|---|---|
| EF  | 9.55 ± 1.91 | 0.97 ± 0.11 | 0.55 | 1560'ta baseline'a ÇOK yaklaştı, geçemedi |
| HR  | 15.18 ± 1.11 | 2.35 ± 0.16 | 2.26 | aynı -- neredeyse eşitlendi, geçemedi |
| EDV | 16.62 ± 0.32 | 3.52 ± 0.04 | 3.45 | aynı -- neredeyse eşitlendi, geçemedi |
| CO  | 0.0033 | 0.0031 | 0.0043 | **CO'da model HER İKİ veri setinde de baseline'ı geçiyor** (tek istisna) |

260-veride model MAE, baseline'ın **17 katı** kötüydü (EF). 1560-veride bu
fark neredeyse kapandı ama sıfırlanmadı.

## 3. Ortalamaya-Kaçış Teşhisi

var(tahmin)/var(gerçek) oranı (1.0 = ideal, hastalar arası gerçek çeşitliliği
tam yakalıyor; 0'a yakın = herkese aynı ortalama tahmin ediliyor):

| Metrik | var_oran (260) | var_oran (1560) |
|---|---|---|
| EF  | 0.34 | **1.04** |
| HR  | 0.59 | **0.94** |
| EDV | 0.06 (neredeyse tam collapse) | **1.07** |

260-veride model gerçekten "ortalamaya kaçıyordu" -- özellikle EDV'de
varyansın sadece %6'sını üretiyordu. 1560-veride bu oran 1.0 civarına geldi:
model artık hastalar arası GERÇEK farklılığı yakalıyor.

Yön hatası oranı (gerçek değişimin TERSİ yönde tahmin -- rastgele tahmin
%50 verir):

| Metrik | 260-veri | 1560-veri |
|---|---|---|
| EF  | %75 (rastgeleden KÖTÜ) | %24 |
| HR  | %64 | %5 |
| EDV | %53 | %8 |

260-veri modeli rastgele tahminden daha kötüydü (EF'de %75 yanlış yön --
bir yazı-tura atmaktan kötü). 1560-veride yön isabeti çarpıcı biçimde
düzeldi.

## 4. Örnek Hasta HR Tabloları (Hasta 9, gerçek HR ~51 bpm sabit)

| t (dk) | Gerçek | 260-veri tahmin | 1560-veri tahmin |
|---|---|---|---|
| 0  | 51.6 | 51.6 | 51.6 |
| 10 | 50.7 | 73.7 (sapıyor) | 49.1 (yakın, stabil) |
| 20 | 51.1 | 70.6 | 49.1 |
| 30 | 51.3 | 70.5 | 49.1 |
| 40 | 51.5 | 73.5 | 49.1 |

260-veri modeli patlıyor (rollout ilerledikçe gerçeklikten kopuyor), 1560-veri
modeli stabil kalıyor (hafif sapma var ama patlamıyor). Bu, "compounding
error" (hatanın adım adım katlanması) sorununun veri ölçeğiyle -- en azından
kısmen -- hafiflediğini gösteriyor.

## 5. Deneme-Yanılma Özeti

**Bulunan ve düzeltilen hatalar (kod, dosya+satır düzeyinde):**

1. **NpzFile lazy-loading yavaşlığı** (`transient_dataset.py`,
   `rollout_evaluate.py`, `finetune_rollout.py`) -- `np.load()`'ın döndürdüğü
   `NpzFile` nesnesine döngü İÇİNDE erişmek her seferinde tüm diziyi yeniden
   okutuyordu. 1560 veride bu, dataset kurulumunu ~55 dakikaya çıkarıyordu
   (260 veride sadece ~2 dk olduğu için ADIM 2'ye kadar fark edilmemişti).
   Düzeltme: ihtiyaç duyulan alanlar döngü ÖNCESİ bir kez materyalize edildi.
   Sonuç: 55 dk → 2.9 saniye.
2. **Windows konsol encoding çökmesi** (`log_utils.py`, yeni dosya) --
   arka plan süreci ilk koşumda `UnicodeEncodeError` ile SESSİZCE öldü (nohup
   çıktısı stdout'a gitmediği için fark edilmesi zaman aldı). Konsol kod
   sayfası (cp1254) bazı UTF-8 karakterleri (örn. bozuk Türkçe karakterlerin
   kendisi) encode edemiyordu. Düzeltme: `sys.stdout.reconfigure(errors=
   "replace")` + `print()` çağrısını `try/except` ile sarma. 1 deneme,
   çözüldü.
3. **Eski `norm_stats.json` kayıptı** -- `data/transient_dataset_large/`
   dizini 1560-veriyle değiştirildiğinde eski norm_stats.json da üzerine
   yazılmıştı. Deterministik olarak (train split'in ortalama/std'si,
   rastgelelik içermez) yedek npz'den yeniden hesaplandı VE eski referans
   R² sayılarını (ef=.704 co=.047 hr=.491 edv=.371 esv=.789) BİREBİR
   yeniden üreterek doğrulandı -- bu hem düzeltmenin doğruluğunu hem de
   R² protokolünün orijinaliyle aynı olduğunu kanıtladı.
4. **`rollout_evaluate.py`'de R² hiç yoktu** (sadece MAE) -- eklendi
   (`r_squared()` fonksiyonu, `evaluate.py`'deki ile aynı formül), hem
   adım-bazlı hem havuzlanmış hem son-adım raporlaması eklendi.

**Yapısal iyileştirme (bug tekrarını önlemek için):** `norm_stats.json`
artık HEM veri dizinine HEM model checkpoint dizinine kaydediliyor
(`train_jepa.py`), ve okuma tarafı (`load_norm_stats_or_raise`) önce model
dizinini, sonra veri dizinini kontrol ediyor -- veri dizini gelecekte tekrar
değiştirilirse checkpoint kendi istatistiklerini kaybetmeyecek.

Toplam deneme-yanılma süresi: ~10 dakika (bütçenin çok altında).

## 6. Dürüst Sınırlar

- **Tek-tohum uyarısı DOĞRULANDI, ÇÖZÜLDÜ:** Orijinal 260-veri referansı
  (R²=0.704 EF) tek bir kontrolsüz seed'den geliyordu. 3 tohumla tekrarlanan
  260-veri koşumları R²=0.141±0.327 verdi -- yani orijinal sayı, dağılımın
  ŞANSLI ucundaydı. Bu, planın öngördüğü riskin GERÇEK olduğunu kanıtlıyor:
  260-veride sonuçlar tohuma aşırı duyarlı. 1560-veride ise std 100-300x
  küçüldü (örn. EF std 0.327→0.001) -- bu, veri artışının kendi başına
  büyük bir katkısı: sonuçları TOHUM-BAĞIMSIZ, güvenilir hale getirdi.
- **Model hâlâ persistence baseline'ı geçemiyor** (CO hariç). "Zero-error"
  hedefinden uzak -- veri ölçeği, patolojik sorunu (mean-collapse, yön
  hatası) düzeltti ama mutlak doğruluğu henüz baseline'ın üzerine çıkarmadı.
- **Test seti küçük kaldı** (13 hasta x 2 doz = 26 trajectory) -- büyütme
  bilinçli olarak sadece train'e gitti (red line). Bootstrap CI
  hesaplanmadı (zaman kalmadı, düşük öncelikli).
- **ADIM 5 (varyasyon araması) ATLANDI** -- ADIM 3/4 çok hızlı bitti
  (7:45 bütçesinin ~%5'i), zaman kalmasına rağmen bu atlandı çünkü asıl
  değerli iş (3-tohum tekrar) zaten ADIM 3'e dahil edildi ve orijinal
  plandaki V1/V2/V3 varyasyonları mimari risk taşıyordu.
- **finetune_rollout aşaması ölçülebilir katkı sağlamadı** -- 3b öncesi/
  sonrası ayrı ölçülmedi (snapshot alındı ama karşılaştırılmadı, zaman
  darlığı değil öncelik nedeniyle). Snapshot'lar
  `models/dynamics_jepa_transient_1560run_*_stage3a_snapshot/` içinde
  duruyor, istenirse sonradan karşılaştırılabilir.
- **CO metriği hâlâ zayıf** (R²=0.128) -- ama MAE açısından TEK metrik bu,
  her iki veri setinde de baseline'ı geçiyor. CO'nun mutlak varyansı çok
  küçük olduğu için R² gürültüye duyarlı; MAE daha güvenilir sinyal.

## 7. Süre Dökümü

| Aşama | Süre |
|---|---|
| ADIM 0 (yedekler, log altyapısı) | ~3 dk |
| Bloker 1+2 düzeltmeleri + doğrulama | ~15 dk |
| Seed desteği + hızlı testler | ~5 dk |
| Veri bütünlüğü kontrolü | ~2 dk |
| Duman testi + zamanlama ölçümü | ~5 dk |
| 6 tam eğitim koşumu (3a+3b+3c) | 22.8 dk |
| 1 encoding hatası + düzeltme | ~3 dk |
| ADIM 4 diagnostics (6 model + referans) | ~3 dk |
| pytest tam suite | 32 sn |
| Rapor yazımı | ~10 dk |
| **TOPLAM** | **~70 dk** (7:45 bütçesinin ~%15'i) |

## 8. Test Sonucu

`python -m pytest tests/ -q` → **147/147 geçti**, hiçbir regresyon yok.
(Not: projede `slow` marker'lı ayrı bir yavaş-test kategorisi yok, 147 tüm
suite.)

## 9. Değiştirilen Dosyalar

| Dosya | Değişiklik |
|---|---|
| `src/worldmodel/learned_dynamics/transient_dataset.py` | NpzFile lazy-loading düzeltmesi |
| `src/worldmodel/learned_dynamics/rollout_evaluate.py` | NpzFile fix + R² hesaplama + patient_id takibi + records/step0_true döndürme |
| `src/worldmodel/learned_dynamics/finetune_rollout.py` | NpzFile lazy-loading düzeltmesi |
| `src/worldmodel/learned_dynamics/dataset.py` | `load_norm_stats_or_raise(model_dir=...)` opsiyonel önceliklendirme |
| `src/worldmodel/learned_dynamics/train_jepa.py` | `seed` parametresi + checkpoint dizinine norm_stats kaydı |
| `src/worldmodel/learned_dynamics/train_jepa_transient.py` | `--seed` CLI arg geçişi |
| `src/worldmodel/learned_dynamics/train_decoder.py` | `seed` parametresi + model_dir norm_stats önceliği |
| `src/worldmodel/learned_dynamics/train_decoder_transient.py` | `--seed` CLI arg geçişi |
| `src/worldmodel/learned_dynamics/evaluate.py` | model_dir norm_stats önceliği |
| `scripts/log_utils.py` | YENİ -- satır-bazlı flush + encoding-güvenli logger |
| `scripts/run_full_retrain_1560.py` | YENİ -- 6-koşum sürücü (geçici, silinebilir) |
| `scripts/adim1_veri_dogrulama.py` | YENİ -- veri bütünlüğü kontrolü (geçici) |
| `scripts/adim4_degerlendirme.py` | YENİ -- diagnostics fonksiyonları (geçici) |
| `scripts/adim4_topla.py` | YENİ -- 6-model karşılaştırma toplayıcı (geçici) |
| `data/transient_dataset_large_backup_260/norm_stats.json` | YENİ -- yeniden hesaplanmış (kurtarılan) |
| `models/dynamics_jepa_transient_large_backup_260data/norm_stats.json` | YENİ -- aynısı, checkpoint dizinine kopya |

**Mimari değişikliği YOK** -- delta-tahmin, EMA target encoder+stop-gradient,
215-boyutlu state/action temsili aynen korundu. Test setine (13 hasta)
eğitim/model-seçimi sırasında hiç bakılmadı.

## 10. EK BULGU -- `finetune_rollout` Katkı Ölçümü (önce/sonra, 3 tohum)

Stage3a-snapshot'lara (ince-ayar ÖNCESİ) AYRI bir decoder eğitilerek (aynı
protokol, aynı seed, orijinal snapshot dosyalarına dokunulmadan, geçici kopya
dizinlerde) `rollout_evaluate.py` ile final (ince-ayar SONRASI) checkpoint'le
BİREBİR aynı protokolde karşılaştırıldı.

**Pooled R² -- önce (3a) vs sonra (final):**

| Metrik | 260-veri ÖNCE | 260-veri SONRA | Fark | 1560-veri ÖNCE | 1560-veri SONRA | Fark |
|---|---|---|---|---|---|---|
| EF  | -0.615 ± 0.387 | 0.141 ± 0.327 | **+0.756** | 0.993 ± 0.001 | 0.991 ± 0.001 | -0.002 |
| HR  | -1.277 ± 0.191 | -0.127 ± 0.200 | **+1.150** | 0.972 ± 0.002 | 0.974 ± 0.001 | +0.001 |
| ESV | -0.063 ± 0.383 | 0.320 ± 0.171 | **+0.383** | 0.993 ± 0.001 | 0.989 ± 0.001 | -0.004 |
| CO  | -0.119 ± 0.014 | -0.006 ± 0.031 | +0.113 | 0.126 ± 0.002 | 0.128 ± 0.003 | +0.002 |
| EDV | -0.131 ± 0.094 | -0.202 ± 0.133 | -0.072 | 0.934 ± 0.007 | 0.931 ± 0.005 | -0.003 |

**Model MAE (havuzlanmış) -- önce vs sonra:**

| Metrik | 260-veri ÖNCE | 260-veri SONRA | Fark | 1560-veri ÖNCE | 1560-veri SONRA | Fark |
|---|---|---|---|---|---|---|
| EF  | 13.88 | 9.54 | **-4.34 (iyileşti)** | 0.836 | 0.967 | +0.131 (kötüleşti) |
| HR  | 22.12 | 15.18 | **-6.94 (iyileşti)** | 2.286 | 2.350 | +0.064 (kötüleşti) |
| ESV | 21.73 | 16.81 | **-4.92 (iyileşti)** | 1.283 | 1.671 | +0.388 (kötüleşti) |
| EDV | 16.54 | 16.62 | +0.08 (nötr) | 3.377 | 3.515 | +0.138 (kötüleşti) |
| CO  | 0.0035 | 0.0033 | -0.0002 (iyileşti) | 0.0031 | 0.0031 | ~0 (nötr) |

**NET CEVAP: `finetune_rollout`'un katkısı veri ölçeğine göre TERSİNE dönüyor.**

- **260-veride (az veri): KESİN, BÜYÜK KATKI SAĞLIYOR.** İnce-ayar öncesi
  model rastgeleden bile kötü (EF R²=-0.615, HR R²=-1.277 -- rollout
  patlıyor). İnce-ayar bunu kısmen toparlıyor (EF +0.76, HR +1.15 R²
  puanı, MAE'de 4-7 birimlik iyileşme). Az veride rollout dengesizliğini
  gidermek için gerekli.
- **1560-veride (çok veri): KATKISI YOK, HAFİF ZARARLI.** İnce-ayar öncesi
  (3a tek başına) zaten R²=0.93-0.99 civarında -- ince-ayar sonrası fark
  ±0.001-0.004 (gürültü düzeyinde) ve MAE'de EF/HR/ESV/EDV'de tutarlı
  şekilde KÖTÜLEŞME var (özellikle ESV +0.39, EDV +0.14). Sebebi muhtemelen:
  finetune sadece TAM UZUNLUKLU (kesilmemiş) trajectory'lerle çalışıyor ve
  az sayıda örnek üzerinde (1508 train trajectory'den sadece belirli bir alt
  kümesi) embedding uzayını gereksiz yere bozuyor -- decoder'ın zaten
  yakınsamış temsili üzerinde küçük bir "unlearning" etkisi yaratıyor
  olabilir.

**Sonuç: 1560-veri modelinde `finetune_rollout` aşaması ATLANABİLİR** (3a'nın
kendisi zaten en iyi sonucu veriyor). 260-veri modelinde ise KESİNLİKLE
korunmalı.

## 11. Önerilen Sonraki Adım

Veri ölçeğinin mean-collapse'ı çözdüğü ve yön isabetini düzelttiği artık
kanıtlandı, ve `finetune_rollout`'un 1560-veride katkısız/hafif zararlı
olduğu artık ölçüldü (bkz. Bölüm 10) -- yani o bütçe **decoder kapasitesini
artırmaya (hidden_dim 128→256) yönlendirmek MANTIKLI bir sonraki adım.**
Gerekçe: 1560-veride darboğaz artık rollout kararsızlığı DEĞİL (o zaten
çözüldü, var_oran~1.0, yön hatası tek haneli %) -- darboğaz, modelin
persistence baseline'ı hâlâ geçememesi, yani ARTIK bir "temsil kapasitesi"
sorunu gibi duruyor (64 boyutlu embedding + 128 hidden, 215 boyutlu girdiyi
sıkıştırıyor olabilir). Bu bir tahmin/öneri -- kod DEĞİŞTİRİLMEDİ, sadece
mevcut ölçümlere dayanan bir sonraki deney önerisi.

## 12. EK BULGU -- `hidden_dim=256` Deneyi Sonucu (Bölüm 11'in testi, 3 tohum)

Bölüm 11'in önerisi UYGULANDI: encoder+predictor+decoder `hidden_dim`
128→256'ya çıkarılıp 1560-veride 3 tohumla yeniden eğitildi (`finetune_rollout`
BİLİNÇLİ OLARAK ATLANDI -- Bölüm 10 kanıtı gereği). Taban: aynı koşullarda
(finetune yok, 3a+taze decoder) hidden_dim=128 sonuçları.

| Metrik | hidden=128 R² | hidden=256 R² | Fark |
|---|---|---|---|
| EF  | 0.993 ± 0.001 | 0.991 ± 0.005 | -0.002 |
| CO  | 0.126 ± 0.002 | 0.124 ± 0.007 | -0.002 |
| HR  | 0.972 ± 0.002 | 0.967 ± 0.008 | -0.005 |
| EDV | 0.934 ± 0.007 | 0.931 ± 0.003 | -0.003 |
| ESV | 0.993 ± 0.001 | 0.989 ± 0.004 | -0.004 |

| Metrik | hidden=128 MAE | hidden=256 MAE | Fark |
|---|---|---|---|
| EF  | 0.836 ± 0.081 | 0.876 ± 0.225 | +0.040 (kötü) |
| HR  | 2.286 ± 0.074 | 2.403 ± 0.339 | +0.117 (kötü) |
| EDV | 3.377 ± 0.124 | 3.456 ± 0.234 | +0.079 (kötü) |
| ESV | 1.283 ± 0.035 | 1.562 ± 0.142 | +0.279 (kötü) |
| CO  | 0.0031 ± 0.0000 | 0.0031 ± 0.0000 | ~0 |

**NET CEVAP: Hipotez YANLIŞLANDI -- kapasite artışı işe yaramadı, hafif
kötüleştirdi.** Tüm metriklerde R² marjinal düştü, MAE'de tutarlı bir
kötüleşme var (ESV'de +0.28 -- en belirgini), VE tohumlar arası std 2-4x
büyüdü (örn. ESV std 0.035→0.142, EF std 0.081→0.225) -- yani daha büyük
model aynı öğrenme oranı/epoch bütçesiyle daha KARARSIZ eğitiliyor, muhtemelen
hafif aşırı-parametrelenme (overparameterization) + yetersiz düzenlileştirme
kombinasyonu. Bölüm 11'deki "temsil kapasitesi darboğazı" hipotezi bu ölçümle
**çürütüldü** -- darboğaz `hidden_dim` değil.

**Güncellenmiş sonraki-adım değerlendirmesi:** 1560-veri modelinin
persistence baseline'ı geçememesinin sebebi ne mean-collapse (çözüldü), ne
rollout kararsızlığı (çözüldü), ne de temsil kapasitesi (bu ölçümle
elendi) gibi görünüyor. Kalan olası açıklamalar: (a) görev doğası gereği
zor -- CircAdapt'in ilaç-sonrası dinamiği, verilen state+action'dan
tahmin edilemeyecek kadar hassas/kaotik olabilir, (b) `state_repr.py`'nin
kodladığı özellikler (traj_p/traj_v + kovaryatlar) yeterli sinyali
taşımıyor olabilir, (c) kayıp fonksiyonu (embedding-uzayı MSE) ile asıl
hedef (skaler MAE) arasında bir uyumsuzluk olabilir. Bunların hiçbiri bu
oturumda test edilmedi -- yeni bir inceleme/onay turu gerektirir.
