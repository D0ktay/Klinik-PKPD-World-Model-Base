# Mini Klinik Dünya Modeli — Sanal Hasta Simülasyonu

Vivax'ın Acudx mimarisinden ilham alan, küçük ölçekli bir kavram-kanıtı
(proof-of-concept). Amaç: "durum + aksiyon → yeni durum dağılımı"
mantığını if-else olmadan, gerçek PK/PD (farmakokinetik/farmakodinamik)
denklemleri ve Monte Carlo simülasyonuyla göstermek.

## Sözlük

Bu projede geçen tüm tıbbi/teknik terimler, alfabetik sırayla, sade
açıklamalarıyla. Kodda/README'de/arayüzde bir terime rastlayıp
"bu ne demek" diye takılırsan önce buraya bak.

- **Afterload (ardyük):** Kalbin kanı dışarı pompalarken karşılaştığı
  direnç -- damarların ne kadar "sıkı" olduğu. Yüksek afterload (ör.
  hipertansiyon), kalbin daha çok zorlanmasına yol açar.
- **AV düğümü (atriyoventriküler düğüm):** Kalbin üst odacıklarından
  (atriyum) gelen elektrik sinyalini alt odacıklara (karıncık) ileten
  "trafik kontrolü" -- bu iletim ne kadar sürdüğü (AV gecikmesi),
  potasyum gibi elektrolitlerden etkilenir.
- **Bireysel duyarlılık (sensitivity):** Aynı ilaç konsantrasyonuna
  hastadan hastaya değişen tepki gücü -- 1.0 "ortalama" bir hastayı,
  >1.0 ilaca ortalamadan daha güçlü yanıt veren bir hastayı, <1.0 daha
  zayıf yanıt vereni temsil eder. Monte Carlo'nun HER denemesinde
  rastgele yeniden örneklenir (bkz. "Dünya Modelini Gözlemle" sayfası).
- **Bradikardi:** Anormal derecede YAVAŞ kalp atışı (bu projede <50
  bpm eşiği kullanılıyor).
- **CircAdapt:** Bu projede kullanılan, kalbin gerçek mekanik
  davranışını (kasılma, kapakçık açılıp kapanması, basınç-hacim
  ilişkisi) fizik denklemleriyle simüle eden bağımsız bir motor.
- **CO (kardiyak output / kalp debisi):** Kalbin DAKİKADA pompaladığı
  toplam kan miktarı (litre/dakika). Normal aralık: 4-8 L/dk.
- **Dağılım hacmi (Vd):** Bir ilacın vücutta ne kadar YAYILDIĞININ
  ölçüsü (litre ya da L/kg cinsinden) -- büyük Vd, ilacın kan
  dolaşımından çok dokulara dağıldığı anlamına gelir.
- **Diyastol:** Kalbin GEVŞEYİP kanla DOLDUĞU faz (sistolün tersi).
- **Durum (state) ve Aksiyon (action):** Bu projenin "dünya modeli"
  mantığının çekirdeği -- hastanın o anki durumu (nabız, tansiyon...)
  + verilen bir aksiyon (ilaç) -> YENİ bir durum üretir. "Dünya
  Modelini Gözlemle" sayfası bu zinciri adım adım gösterir.
- **EC50:** Bir ilacın YARI-MAKSİMUM etkiyi yapması için gereken
  konsantrasyon -- düşük EC50, ilacın az miktarda bile etkili olduğu
  (güçlü/duyarlı) anlamına gelir.
- **EDV (end-diastolic volume / diyastol sonu hacmi):** Kalbin
  diyastol (dolma) fazının SONUNDA ulaştığı EN YÜKSEK hacim.
- **EF (ejeksiyon fraksiyonu):** Kalbin HER ATIŞTA içindeki kanın
  YÜZDE KAÇINI pompaladığı -- kardiyolojinin "kalp ne kadar iyi
  pompalıyor" sorusuna verdiği standart cevap. Normal: %55-70,
  hafif azalmış: %40-54, düşük (kalp yetmezliği belirtisi): <%40.
- **Emax:** Bir ilacın yapabileceği MAKSİMUM etkinin büyüklüğü (ör.
  `emax_hr=25`, nabzı en fazla 25 bpm değiştirebileceği anlamına gelir).
- **ESV (end-systolic volume / sistol sonu hacmi):** Kalbin sistol
  (kasılma) fazının SONUNDA kalan EN DÜŞÜK hacim.
- **Farmakodinamik (PD):** İlacın vücuttaki KONSANTRASYONUNUN ne kadar
  ETKİ yarattığı -- "bu miktar ilaç, ne kadar fark yaratır?" sorusu.
- **Farmakokinetik (PK):** İlacın vücutta nasıl DAĞILIP ATILDIĞI --
  "ilaç vücuda girdikten sonra ne olur?" sorusu.
- **Frank-Starling mekanizması:** Kalbin, ne kadar GERİLİRSE (önyük
  arttıkça) o kadar GÜÇLÜ kasıldığı kuralı -- bu projede sıkça
  görülen "nabız yavaşladı ama basınç yine de arttı" gibi
  görünüşte-şaşırtıcı sonuçların fizyolojik açıklaması.
- **Hiperkalemi / Hipokalemi:** Kandaki potasyumun normalden YÜKSEK /
  DÜŞÜK olması (normal aralık: 3.5-5.0 mEq/L). Hiperkalemi kalbin
  elektrik iletimini yavaşlatır.
- **Hiperkalsemi / Hipokalsemi:** Kandaki kalsiyumun normalden YÜKSEK
  / DÜŞÜK olması (normal aralık: 8.5-10.5 mg/dL). Kalsiyum,
  kontraktiliteyle doğru orantılıdır.
- **Kalp yetmezliği (sistolik):** Kalbin yeterince GÜÇLÜ kasılamadığı,
  düşük EF ile karakterize bir durum.
- **Keo:** Bir ilacın etkisinin, kandaki konsantrasyonuna ne kadar
  HIZLI "yetiştiğinin" ölçüsü -- büyük keo, etkinin konsantrasyona
  hızlı yetiştiği (az gecikme) anlamına gelir.
- **Komorbidite:** Hastanın, tedavi edilen durumdan BAĞIMSIZ olarak
  ZATEN sahip olduğu kronik bir hastalık (ör. kalp yetmezliği,
  hipertansiyon).
- **Kontraktilite:** Kalp kasının NE KADAR GÜÇLÜ kasıldığı.
- **LV (sol karıncık / left ventricle):** Kalbin, oksijenli kanı
  VÜCUDA pompalayan ANA odacığı -- bu projedeki grafiklerin çoğu
  LV'yi izliyor çünkü kardiyolojide en çok izlenen odacık budur.
- **LVEDV:** LV'nin EDV'si (yani sol karıncığın diyastol sonundaki
  en yüksek hacmi).
- **Monte Carlo simülasyonu:** Aynı hasta+ilaç kombinasyonunu YÜZLERCE
  kez, her seferinde bireysel duyarlılık ve ke gibi değerleri rastgele
  örnekleyerek çalıştırma yöntemi -- tek bir "kesin" sonuç yerine,
  gerçekçi bir sonuç DAĞILIMI (ve bu dağılımdan çıkan risk yüzdeleri)
  üretir.
- **Nabız (HR / heart rate):** Kalbin DAKİKADA kaç kez attığı (bpm).
- **PFC (Pressure Flow Control):** CircAdapt'in, toplam kan hacmini/
  basıncını kendi hedefine göre otomatik ayarlayan iç kontrol
  mekanizması (kaba bir baroreseptör-refleks benzetmesi).
- **Polifarmasi:** Birden fazla ilacın AYNI ANDA verilmesi.
- **Preload (önyük):** Kalbin DOLMA anındaki yükü/gerilimi -- ne kadar
  kan geldiği. Frank-Starling mekanizmasının girdisi.
- **PV Loop (Basınç-Hacim İlişkisi):** Bir kalp atışı boyunca
  basıncın hacme karşı çizildiği, kardiyolojide klasik bir grafik --
  döngünün GENİŞLİĞİ atım hacmini gösterir.
- **RxCUI:** RxNorm (ABD Ulusal Tıp Kütüphanesi) veritabanının bir
  ilaca verdiği STANDART kimlik numarası.
- **SBP / DBP (sistolik / diyastolik tansiyon):** Tansiyonun sistol
  (kasılma) sırasındaki YÜKSEK değeri (SBP) ve diyastol (gevşeme)
  sırasındaki DÜŞÜK değeri (DBP).
- **Sistol:** Kalbin KASILIP kan pompaladığı faz (diyastolün tersi).
- **SVT (supraventriküler taşikardi):** Kalbin üst odacıklarından
  kaynaklanan, anormal derecede HIZLI bir kalp ritmi bozukluğu.
- **t½ (yarı ömür):** Bir ilacın vücuttaki miktarının YARISINA
  inmesi için geçen süre -- kısa yarı ömür, ilacın hızlı etkisinin
  geçtiği anlamına gelir.
- **Titrasyon:** Bir ilacın dozunun, istenen etkiye ulaşana kadar
  YAVAŞ YAVAŞ ayarlanması.
- **İki-kompartmanlı model:** İlacın vücutta TEK bir "havuzda" değil,
  biri hızlı (kan + yüksek kanlanan dokular) biri yavaş (diğer
  dokular) İKİ ayrı "havuzda" dağıldığını varsayan, tek-kompartmanlı
  modelden daha gerçekçi bir farmakokinetik model.
- **ka (emilim/dağılım hız sabiti):** İlacın kana ne kadar HIZLI
  karıştığının ölçüsü.
- **ke (eliminasyon hız sabiti):** İlacın vücuttan ne kadar HIZLI
  atıldığının ölçüsü.

---

## Proje Yapısı

```
vivax_world_model_demo/
├── app.py                  # CLI giriş noktası
├── streamlit_app.py        # İnteraktif tarayıcı demosu
├── test_circadapt_installation.py     # CircAdapt kurulum testi
├── integrate_drug_with_circadapt.py   # PK/PD -> CircAdapt entegrasyonu (ilaç sınıfına göre)
├── compare_drug_classes_circadapt.py  # 3 ilaç sınıfını CircAdapt'te karşılaştırır
├── compare_pk_models.py               # Tek- vs iki-kompartmanlı PK modeli karşılaştırması
├── plot_keo_effect_separation.py      # HR vs SBP etki zamanlaması (Keo) görselleştirmesi
├── drug_lookup.py                     # RxNorm/openFDA'dan gerçek ilaç verisi çeker (Faz 9)
├── compare_polypharmacy.py            # Tehlikeli ilaç kombinasyonu gösterimi (Faz 10)
├── compare_electrolyte_effects.py     # Elektrolit durumunun kalp üzerindeki etkisi (Faz 11)
├── compare_comorbidity_effects.py     # Aynı doz, farklı hasta/hastalık profili (Faz 12)
├── CALIBRATION_REPORT.md              # Hangi parametre gerçek, hangisi varsayım (Faz 13)
├── requirements.txt
├── configs/
│   ├── patients.yaml       # Hasta profilleri (kilo, boy, bazal vitaller, elektrolit, komorbidite...)
│   ├── drugs.yaml          # İlaç profilleri (PK/PD parametreleri, drug_class)
│   └── drugs_verified.yaml # Kaynak izlenebilirlikli (RxCUI/source_url) 5 gerçek ilaç (Faz 9)
├── src/worldmodel/
│   ├── patient.py          # Patient / Drug veri modelleri
│   ├── pk.py               # Farmakokinetik: konsantrasyon hesabı
│   ├── pd.py                # Farmakodinamik: konsantrasyon -> etki
│   ├── simulation.py       # Monte Carlo motoru (yüzlerce deneme) + doz önerisi + polifarmasi
│   ├── provenance.py       # Veri kaynağı izlenebilirliği / audit trail (Faz 14)
│   ├── report.py           # PDF klinik rapor çıktısı (Faz 15)
│   └── viz.py               # Grafik üretimi
├── tests/
│   ├── test_pk.py                    # Birim testler
│   └── test_clinical_validation.py   # Gerçek klinik veriyle karşılaştırma (Faz 13)
└── outputs/                 # Üretilen grafikler buraya kaydedilir
```

## Kurulum (VS Code'da)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
# CLI ile
python app.py --patient hasta_a --drug beta_bloker --n 300

# İnteraktif demo (mülakat için önerilir)
streamlit run streamlit_app.py

# Testler
python -m pytest tests/ -v
```

---

## Bilinen CircAdapt API Davranışları (Faz 3'te keşfedildi)

CircAdapt Python paketiyle çalışırken deneyerek bulduğumuz, dokümante
edilmemiş iki davranış:

1. **`model[bileşen][parametre]` bir dizi döndürdüğünde, öğesini
   değiştirip AYNI nesneyi geri atarsan değişiklik sessizce geri
   alınır.** Yani:
   ```python
   arr = model["Patch"]["Sf_act"]
   arr[[2, 3, 4]] = arr[[2, 3, 4]] * 0.8
   model["Patch"]["Sf_act"] = arr   # BUG: bu satır değişikliği İPTAL EDER
   ```
   Doğrusu, üçüncü satırı YAZMAMAK:
   ```python
   arr = model["Patch"]["Sf_act"]
   arr[[2, 3, 4]] = arr[[2, 3, 4]] * 0.8   # bu tek başına yeterli
   ```
   Skaler parametrelerde (örn. `General.t_cycle`, bir `float`) bu sorun
   yok, çünkü orada zaten normal Python skaler ataması yapılıyor.
   **Önemli:** Bu bug, projenin Faz 1-2'sindeki `Patch.Sf_act`
   (kontraktilite) değişikliklerinin FİİLEN HİÇ UYGULANMADIĞI, o
   fazlardaki CircAdapt sonuçlarının aslında SADECE `General.t_cycle`
   (nabız) değişikliğinden kaynaklandığı anlamına geliyordu. Faz 3'te
   `integrate_drug_with_circadapt.py` düzeltildi; artık kontraktilite
   değişikliği gerçekten uygulanıyor (uçtan uca simülasyonla doğrulandı:
   `Sf_act` yarıya indirildiğinde LV pik basıncı 118.0 -> 112.2 mmHg
   değişiyor, düzeltmeden önce hiç değişmiyordu).

2. **`PressureFlowControl` (PFC) bileşeni, `ArtVen` direncini kendi
   hedefine göre otomatik olarak yeniden ayarlıyor** ("Pressure Flow
   Control by fitting ArtVen resistance and total volume" --
   `circadapt/components/global_functions.py` docstring'i). Yani
   `ArtVen.p0` manuel değiştirilip `model.run(stable=True)` çağrılırsa,
   PFC bu değişikliği stabilizasyon sırasında kendi hedefine geri
   "fit" ederek sessizce nötralize eder. Vazodilatör etkisini (sistemik
   direnç düşüşü) gerçekten göstermek için, bu simülasyon süresince
   `model["PFC"]["is_active"] = False` ile geçici olarak devre dışı
   bırakılması gerekiyor (bkz. `apply_drug_effect_to_circadapt`).

**Ders:** CircAdapt parametre isimlerini VE davranışlarını asla tahmin
etmeyin -- hem isim hem de "atama gerçekten kalıcı oluyor mu" sorusunu
küçük, izole bir Python betiğiyle (`model[...] = ...` sonrası taze bir
`model[...]` okuması) test ederek doğrulayın.

---

## Veri Kaynakları

`configs/drugs.yaml` içindeki `beta_bloker` (esmolol) profili için:

- **PK parametreleri (`dose_mg`, `ka`, `ke_mean`, `vd_per_kg`) gerçek
  literatürden kalibre edilmiştir:**
  - Doz: 0.5 mg/kg bolus, referans ~76 kg hasta için (FDA prescribing
    information / Brevibloc etiketi)
  - `ka` (dağılım hız sabiti): dağılım yarı ömrü ~2 dk kabul edilerek
    `ln(2)/t½` ile hesaplanmıştır (Wiest DB et al., 1991)
  - `ke_mean` (eliminasyon hız sabiti): eliminasyon yarı ömrü ~9 dk
    kabul edilerek `ln(2)/t½` ile hesaplanmıştır (FDA label, Wiest 1991)
  - `vd_per_kg`: yayınlanmış yetişkin/pediatrik PK çalışmalarındaki
    dağılım hacmi aralığından alınmıştır
  - **Not:** Model şu an tek-kompartmanlı ve `ka` üzerinden "emilim"
    formülasyonu kullanıyor; esmolol klinikte IV bolus/infüzyon olarak
    verilir, oral emilmez. Buradaki `ka`, gerçek oral emilimi değil,
    esmolol'ün hızlı dağılım fazını temsili olarak yaklaştırmak için
    kullanılmıştır — daha doğru bir gösterim için Faz 4'teki
    iki-kompartmanlı IV bolus modeline bakın.
- **PD parametreleri (`emax_hr`, `emax_sbp`, `ec50`) hâlâ temsili
  (kalibrasyon gerektiren) değerlerdir** — herhangi bir yayınlanmış
  Emax çalışmasından alınmamıştır, sadece "makul görünen" sayılardır.
  Gerçek bir klinik/araştırma kullanımı için bu üç parametrenin de
  yayınlanmış doz-yanıt verisiyle kalibre edilmesi gerekir.

`vazodilator` profili hâlâ tamamen temsili/uydurma değerler içerir;
literatür kalibrasyonu henüz yapılmamıştır.

### İki-Kompartmanlı Model (Faz 4) ve k10/k12/k21'in Kaynağı

`pk.py > plasma_concentration_two_compartment` ve `drugs.yaml`'daki
`beta_bloker.k10/k12/k21/vd_central_per_kg` alanları:

- `alpha` (dağılım, t½≈2dk) ve `beta` (eliminasyon, t½≈9dk) makro hız
  sabitleri **gerçek literatürden** (Wiest 1991, FDA label) geliyor --
  bunlar `ka`/`ke_mean` ile aynı kaynak.
- Bu iki sayıdan üç mikro sabiti (k10, k12, k21) tek başına türetmek
  matematiksel olarak imkânsız (2 denklem, 3 bilinmeyen) -- üçüncü bir
  veri noktası gerekiyor. Klerens (`CL = ke_mean * vd_per_kg = 9.2
  L/saat/kg`) yine literatürden türetilebiliyor, ama esmolole özgü
  ayrı bir **santral kompartman hacmi (Vc)** yayını bulunamadı. Bu
  yüzden `vd_central_per_kg = 0.5 L/kg` (Vss'nin ~dörtte biri) bir
  **VARSAYIM** olarak kullanıldı; k10/k12/k21 bu varsayımla, gerçek
  alpha/beta değerlerini tam karşılayacak şekilde standart
  iki-kompartman denklemleriyle geriye doğru hesaplandı
  (`tests/test_pk.py::test_esmolol_two_compartment_config_matches_alpha_beta_half_lives`
  bunu doğruluyor).
- **Model farkı, sayı hatası değil:** tek-kompartmanlı model bir
  oral/absorpsiyon yaklaşıklığı (`ka` ile, t=0'da C=0) kullanırken,
  iki-kompartmanlı model gerçek bir IV bolus (t=0'da C=dose/Vc ile pik)
  varsayıyor. İkisi de aynı hastada çok farklı eğriler üretiyor (bkz.
  `compare_pk_models.py` çıktısı) -- bu beklenen bir davranış, esmolol
  klinikte IV bolus verildiği için ikinci model daha doğru bir temsil.
- **Roadmap'teki bir tutarsızlık düzeltildi:** orijinal Faz 4 talimatı
  iki-kompartmanlı modelin de "t=0'da konsantrasyon=0" vermesini test
  etmeyi istiyordu. Bu, IV bolus fiziği için YANLIŞ bir beklenti (IV
  bolus t=0'da tam tersine PİK yapar, sıfır değil) -- muhtemelen
  tek-kompartmanlı absorpsiyon modelinin testinden kopyalanırken
  gözden kaçmış. `tests/test_pk.py`'da bunun yerine modelin GERÇEKTE
  doğru olması gereken davranışı (t=0'da dose/Vc'ye eşit pozitif pik,
  sonrasında monoton azalma) test edildi.

### Etki Bölgesi (Keo) Gecikmesi (Faz 5)

`pd.py > effect_compartment_concentration` ve her ilaçtaki
`keo_hr`/`keo_sbp` alanları: nabız ve tansiyon etkisinin plazma
konsantrasyonuna FARKLI hızlarda "yetiştiğini" modelliyor (bkz.
`plot_keo_effect_separation.py` çıktısı -- esmololde nabız etkisi ~11
dk'da, tansiyon etkisi ~15 dk'da tepe yapıyor, plazma piki ise ~6 dk).

**Bu değerler TEMSİLİDİR, literatürden değil** -- her ilaç için
"hangi etki daha doğrudan/hızlı, hangisi dolaylı/downstream" mantığıyla
elle seçildi (örn. beta-blokerde kronotropi doğrudan reseptör etkisi
olduğu için `keo_hr > keo_sbp`; vazodilatörde ise SBP etkisi doğrudan,
HR etkisi refleks/baroreseptör aracılı olduğu için tam tersi). Gerçek
bir klinik kullanım için bu sabitlerin de yayınlanmış PK/PD
çalışmalarından kalibre edilmesi gerekir -- şu an sadece "nabız ve
tansiyon etkisi aynı anda tepe yapmaz" prensibinin gösterimi.

`keo_hr`/`keo_sbp` tanımlanmamış bir ilaçta (`None`), sistem otomatik
olarak eski (gecikmesiz, tek eğri) davranışa düşer -- geriye dönük
uyumluluk bozulmadı.

### Doz Önerisi: İstatistiksel + Mekanik Risk (Faz 6-7)

`simulation.py > recommend_dose`, artık iki farklı risk kaynağını
birleştiriyor:

1. **İstatistiksel risk** -- PK/PD Monte Carlo'dan gelen bradikardi
   riski (Faz 2'den beri var).
2. **Mekanik risk** -- CircAdapt'ten gelen LVEDV (diyastol sonu hacim)
   artışı, baseline'a göre %20'yi (varsayılan eşik) aşarsa "aşırı
   önyük (preload) riski" olarak işaretlenir.

İkisi çakıştığında (örn. esmololde istatistiksel model 16 mg'ı güvenli
buluyor, ama CircAdapt bu dozda LVEDV'nin %26 arttığını gösteriyor),
sistem istatistiksel öneri yerine daha temkinli bir doz (varsayılan
`conservative_dose_factor=0.7` ile ölçeklenmiş) önerir ve gerekçeyi
insan-okunur bir metin olarak döndürür (`rec["reasoning"]`). Streamlit
arayüzünde bu, kırmızı bir uyarı kutusu olarak gösteriliyor.

**Bilinen sınır:** `recommend_dose`'un istatistiksel tarafı hâlâ sadece
bradikardi riskine bakıyor -- bu, beta-blokerler için anlamlı ama
vazodilatör/pozitif inotrop gibi bradikardi yapmayan ilaçlarda "en
yüksek taranan doz her zaman güvenli" sonucuna varıyor (çünkü onların
gerçek riskleri, ör. aşırı hipotansiyon veya aritmi, farklı bir
metrik gerektirir -- şu an ölçülmüyor).

### Streamlit Arayüzü: Gerçek İlaç Seçimi (Faz 7)

Artık "Demo İlaç" adında sabit/hardcoded bir ilaç yok -- kullanıcı
`configs/drugs.yaml`'daki 4 gerçek ilaçtan birini (`display_name
[drug_class]` formatında) bir açılır menüden seçiyor. Doz ve EC50
slider'ları, seçili ilacın kendi referans değerlerine göre dinamik
aralık alıyor (esmolol ~38mg, nikardipin ~2.3mg gibi çok farklı
referans dozları tek sabit aralığa sığmadığı için). CircAdapt bölümü
otomatik olarak seçili ilacın `drug_class`'ına uygun mekanizmayı
(Faz 3) kullanıyor -- kullanıcı ayrıca bir şey seçmesine gerek yok.

**Test sırasında bulunan ayrı bir bug:** EC50 gibi çok küçük değerli
(nikardipin: 0.0015 mg/L) slider'lar, Streamlit'in varsayılan
`"%.2f"` formatıyla "0.00" gösteriyordu -- değer aslında doğruydu, sadece
görünmüyordu. `format="%.5f"` ile düzeltildi.

**Ortam notu (bu projeyle ilgisiz ama not etmeye değer):** Geliştirme
sırasında `lsof`'un bu Windows/git-bash ortamında bulunmadığı, bu
yüzden önceki "sunucuyu öldür" komutlarının hep sessizce başarısız
olduğu ve portta birden fazla eski `streamlit` sürecinin birikip
istekleri rastgele karşıladığı ortaya çıktı (bir süreç eski, düzeltilmemiş
kodu sunuyordu). Windows'ta süreç yönetimi için `Get-NetTCPConnection`
+ `Stop-Process` (PowerShell) kullanmak gerekiyor, `lsof` değil.

### Böbrek/Karaciğer Fonksiyonu (Faz 8, opsiyonel/stretch)

`Patient.renal_function`/`hepatic_function` (0-1, normal=1.0) ve
`Drug.renal_clearance_fraction`/`hepatic_clearance_fraction` (0-1,
varsayılan 0.0) eklendi. `pk.py > organ_function_adjusted_ke`, toplam
eliminasyonun sadece ilaca özgü böbrek/karaciğer payını, ilgili organ
fonksiyon kaybıyla orantılı düşürüyor.

- **Esmolol bilinçli olarak ETKİLENMEZ** (`renal_clearance_fraction=
  hepatic_clearance_fraction=0.0`) -- gerçekte eritrosit esterazlarıyla
  metabolize olur, böbrek/karaciğer yetmezliğinde bile öngörülebilir
  kısa etkili kalır. Testle doğrulandı: `renal_function`/`hepatic_function`
  ne olursa olsun esmololün konsantrasyon eğrisi birebir aynı kalıyor.
- **Yeni ilaç: `digoxin`** -- standart ders kitabı PK değerleriyle
  (t½≈36 saat, Vd≈7.3 L/kg, ~%65 değişmeden böbrekten atılım) eklendi.
  `renal_clearance_fraction=0.65` sayesinde, böbrek yetmezliği olan bir
  hastada (`renal_function=0.2`) 48 saat sonunda konsantrasyon normal
  böbrek fonksiyonuna göre >1.5x daha yüksek kalıyor -- dar terapötik
  indeksli bu ilacın böbrek yetmezliğinde neden toksik biriktiğinin
  doğrudan bir gösterimi (test edildi).
- Digoksin farmakolojik olarak hem pozitif inotrop hem negatif
  kronotroptur (AV düğümü yavaşlatır) -- dobutaminden farklı olarak
  `emax_hr` POZİTİF (nabzı düşürür), ama CircAdapt'te aynı
  `drug_class="positive_inotrope"` mekanizmasını (Sf_act artışı)
  kullanır, çünkü kontraktilite etkisi aynı yönde.

### Gerçek İlaç Veritabanı Entegrasyonu (Faz 9)

`drug_lookup.py`, iki ücretsiz halka açık API'ye bağlanıyor:

- **RxNorm** (`lookup_rxcui`) -- bir ilaç adının resmi RxCUI
  tanımlayıcısını çeker (ilacın gerçekten var olduğunu doğrular).
- **openFDA** (`fetch_fda_label_sections`) -- FDA prescribing
  information'ından ilgili metin bölümlerini getirir.

**Bilinçli tasarım kararı: TAM OTOMATİK sayı çıkarımı YAPILMIYOR.**
FDA etiketleri serbest metin; "half-life: 3.5 hours" gibi bir cümleden
regex'le sayı çıkarmaya çalışmak, ilaçtan ilaca format farklılığı
yüzünden sessizce yanlış bir değer üretebilir -- bu proje boyunca
defalarca gördüğümüz gibi (Faz 3'teki CircAdapt Parameter bug'ı, Faz
7'deki EC50 format bug'ı), "gerçek görünen ama yanlış" bir sayı, hiç
sayı olmamasından daha tehlikelidir. Bunun yerine `drug_lookup.py`
ilgili metni getirir, geliştirici bunu okuyup parametreleri elle
kalibre eder -- tam olarak `configs/drugs_verified.yaml`'ın nasıl
üretildiği budur.

`configs/drugs_verified.yaml`, `drugs.yaml`'dan farklı bir dosya --
her ilaç girdisine `rxcui`, `source_url`, `retrieved_date` ve
`calibration_notes` alanları ekliyor (Drug dataclass'ının parçası
değil, `load_verified_drugs()` bunları ayrıştırıp hem çalışan bir
`Drug` nesnesi hem de kaynak bilgisini ayrı döndürüyor). 5 ilaç:

- **esmolol** -- FDA etiketi Faz 1'in Wiest 1991 kaynaklı 9 dk
  eliminasyon yarı ömrünü ve 0.5 mg/kg dozunu BAĞIMSIZ olarak doğruladı.
- **metoprolol** (yeni) -- FDA etiketinden gerçek IV protokol:
  3×5mg bolus (~2dk arayla), maksimum beta-blokaj ~20 dk'da.
- **sodium_nitroprusside** (yeni) -- FDA etiketi doğrudan "circulatory
  half-life is about 2 minutes" ve dağılım hacminin ekstrasellüler
  sıvıya "coextensive" olduğunu belirtiyor -- `vd_per_kg=0.2` doğrudan
  bu ifadeden türetildi.
- **dobutamine**, **digoxin** -- `drugs.yaml`'daki değerlerle aynı,
  ayrıca openFDA'dan bağımsız doğrulandı (digoksin etiketi kelimesi
  kelimesine "renal function" u bir dozlama faktörü olarak sayıyor --
  Faz 8'in tüm önermesini FDA'nın kendi metninden doğruluyor).

**Bulunan bir bug:** `fetch_fda_label_sections` başta çok kelimeli ilaç
adlarını (örn. "sodium nitroprusside") tırnaksız aratıyordu, bu da
openFDA'nın kelimeleri ayrı ayrı eşleştirip **yanlış ilacı** ("sodium
fluoride" -- bir ağız gargarası!) döndürmesine yol açtı. Tam ifade
eşleşmesi için tırnak eklenerek düzeltildi (regresyon testiyle
korunuyor: `test_fetch_fda_label_sections_quotes_multiword_drug_names`).

### Çoklu İlaç Etkileşimi / Polifarmasi (Faz 10)

`simulation.py > run_polypharmacy_simulation`, birden fazla ilacın AYNI
ANDA verildiği bir senaryoyu simüle ediyor -- her ilacın nabız/tansiyon
üzerindeki katkısı (emax × etki oranı) toplanıyor. `integrate_drug_
with_circadapt.py > run_polypharmacy_with_circadapt` ise aynı mantığı
CircAdapt seviyesinde uyguluyor: her ilaç kendi `drug_class`'ının
mekanizmasını (t_cycle her zaman, + Sf_act veya ArtVen.p0) AYNI model
üzerinde sırayla hedefliyor -- farklı sınıftaki iki ilaç (örn.
beta-bloker + vazodilatör) gerçekten AYNI ANDA İKİ FARKLI parametreyi
değiştiriyor.

**Tehlikeli kombinasyon gösterimi:** esmolol (beta-bloker) + digoksin
(pozitif inotrop, ama AYNI ZAMANDA negatif kronotrop -- AV düğümünü
yavaşlatır) -- ikisi de farklı mekanizmalardan nabzı düşürüyor.
`compare_polypharmacy.py` hem istatistiksel PK/PD'de hem CircAdapt'te
bunu gösteriyor: birlikte verildiğinde nabız, İKİSİNİN TEK BAŞINA
ürettiğinden daha da düşük oluyor (CircAdapt: 53.8/61.6 bpm tek başına
-> 47.0 bpm birlikte). Bu, gerçek klinikte de bilinen bir uyarıdır
(beta-bloker + digoksin kombinasyonu AV blok riskini artırır).

`recommend_dose`, artık opsiyonel bir `polypharmacy_result` parametresi
kabul ediyor -- hastanın zaten kullandığı başka bir ilaçla birlikte
verildiğinde bradikardi riski tek başına risklerin
`polypharmacy_risk_multiplier`inden (varsayılan 1.5x) fazla artıyorsa,
gerekçeye "TEHLİKELİ KOMBİNASYON UYARISI" ekleniyor (dozu otomatik
değiştirmiyor -- hangi ilacın dozunun düşürüleceği klinik bir karar).

`interaction_matrix` parametresi, iki ilaç arasında EK bir sinerji
terimi eklemeyi sağlıyor (varsayılan: yok, saf toplamsal) -- test
edildi ama şu an hiçbir gerçek ilaç çiftinde varsayılan olarak
kullanılmıyor (temsili bir özellik, gerçek bir etkileşim çalışmasından
kalibre edilmedi).

### Elektrolit / Laboratuvar Verisinin Kalp Üzerindeki Etkisi (Faz 11)

`Patient.potassium_mEqL`/`calcium_mgdL` (normal aralık: K 3.5-5.0,
Ca 8.5-10.5) eklendi -- "sadece kilo" ötesinde gerçek hasta verisinin
simülasyona katıldığı yeni bir boyut, ve İLAÇTAN TAMAMEN BAĞIMSIZ.

- **Potasyum -> AV iletim gecikmesi** (`pd.py > potassium_av_conduction_
  factor`): hiperkalemi AV düğümü iletimini yavaşlatır (gerçek, iyi
  bilinen fizyoloji -- EKG'de PR uzaması). CircAdapt'te doğru parametre
  keşfedilerek bulundu: `Timings.tau_av`'ı DOĞRUDAN değiştirmek işe
  yaramıyor -- `tau_av`, her adımda `c_tau_av0 + c_tau_av1 * t_cycle`
  formülünden ("law") yeniden hesaplanıyor, manuel atama sessizce geri
  alınıyor (Faz 3'teki Sf_act/ArtVen bug'larıyla AYNI kalıp). Doğru
  kullanım `Timings.c_tau_av1` katsayısını değiştirmek. Test edildi:
  hiperkalemik hasta profilinde (K=6.5) AV gecikmesi 150ms'den 218ms'ye
  çıkıyor (`compare_electrolyte_effects.py`).
- **Kalsiyum -> kontraktilite** (`pd.py > calcium_contractility_factor`):
  `Patch.Sf_act`'e uygulanıyor. Hipokalsemi profilinde (Ca=6.8) LV pik
  basıncı hafifçe düşüyor (118.0->116.7 mmHg).
- `run_baseline(patient)` artık opsiyonel bir `patient` parametresi
  alıyor -- verilirse "ilaçsız baseline" artık jenerik bir sağlıklı kalp
  değil, HASTANIN KENDİ elektrolit durumunu yansıtan bir kalp. Normal
  orta noktadaki (K=4.25, Ca=9.5) hastalarda bu tam bir NO-OP, eski
  davranış bozulmadı (regresyon testiyle doğrulandı).
- `configs/patients.yaml`'a 2 yeni profil: `hasta_c_hiperkalemi` (böbrek
  yetmezliğine bağlı K=6.5) ve `hasta_d_hipokalsemi` (Ca=6.8).
- `recommend_dose`, hastanın `has_abnormal_electrolytes` bayrağı true
  ise otomatik bir "LAB UYARISI" ekliyor (ör. "hipokalemik hastada bu
  ilaç aritmi riskini artırabilir" tarzı bir bilgilendirme -- ilaç
  sınıfına özel bir risk hesaplamıyor, sadece klinisyeni uyarıyor).

### Komorbidite / Hastalık Durumları (Faz 12)

`Patient.comorbidity` (`None` / `"heart_failure"` / `"hypertension"`) ve
`integrate_drug_with_circadapt.py > apply_comorbidity_to_circadapt` --
artık herkes "sağlıklı bazal" bir kalple başlamıyor.

- **`heart_failure`** (sistolik kalp yetmezliği): `Patch.Sf_act`'i
  (kontraktilite) %40 düşürür -- sistolik KY'nin TANIMI zaten "azalmış
  kontraktilite/EF" olduğu için en doğrudan mekanizma. Sonuç: klasik
  "dilate + azalmış EF" imzası (LVEDV 120.3->155.3 mL, LV pik 118.0->
  114.8 mmHg, ilaçsız baseline'da bile).
- **`hypertension`** (kronik hipertansiyon): HEM `ArtVen.p0[0]` (sistemik
  direnç) HEM `PFC.p0` (PressureFlowControl'ün kendi hedef basıncı)
  BİRLİKTE artırılır. **Sadece `ArtVen.p0` artırmak yetmiyor** -- PFC
  normalde kendi sabit hedefine geri "fit" ediyor (Faz 3'te keşfedilen
  davranış), bu yüzden tek başına nötralize ediliyor (uçtan uca
  doğrulandı: 91.49->91.49 mmHg, değişim yok). `PFC.p0`'ı da artırmak,
  "vücudun yüksek basıncı normal kabul ettiği" KRONİK bir adaptasyonu
  temsil ediyor -- akut vazodilatör senaryosundaki (PFC devre dışı
  bırakılan) YAKLAŞIMDAN kasıtlı olarak farklı bir modelleme kararı.
  İkisi birlikte kalıcı bir basınç yükselmesi üretiyor (91.49->118.94
  mmHg SyArt, 118.0->145.4 mmHg LV pik).
- `configs/patients.yaml`'a 2 yeni profil: `hasta_e_kalp_yetmezligi`,
  `hasta_f_hipertansif`.

**En güçlü gösterim (`compare_comorbidity_effects.py`):** AYNI esmolol
dozu üç farklı kalpte çalıştırıldığında -- sağlıklı hastada LVEDV
+31.0 mL artarken, kalp yetmezliği hastasında (zaten dilate olan bir
ventrikülde) +45.0 mL artıyor (200.3 mL'ye ulaşıyor) -- akut
dekompansasyon riskini somut olarak gösteren bir sayı. Bu, gerçek
klinikte dozların hastaya göre uyarlanmasının TAM OLARAK nedenidir.

### Doğrulama Katmanı (Faz 13)

`tests/test_clinical_validation.py`, esmolol için modelin ÇIKTISINI
(girdisini değil) yayınlanmış klinik çalışmalarla kıyaslıyor -- tam
liste ve dürüst değerlendirme için **[CALIBRATION_REPORT.md](CALIBRATION_REPORT.md)**'ye
bakın. Özet: onset/duration testleri gerçek çalışmalarla iyi uyuşuyor;
bir test (pik etki büyüklüğü) beklenmedik şekilde bir SVT infüzyon
çalışmasına yakın çıktı ama bu GÜVENİLİR bir doğrulama sayılmamalı
(rastlantısal olabilir, farklı doz rejimi/hasta profili); en sağlam
bulgu, modelin TEK-BOLUS PK'sının (60 dk'da konsantrasyon ~%2'ye iniyor)
published SÜREKLİ İNFÜZYON çalışmasıyla yapısal olarak kıyaslanamaz
olduğu -- yani proje idame infüzyon tedavisini modellemiyor, bu bilinen
bir kapsam sınırı.

### Veri Kaynağı İzlenebilirliği / Audit Trail (Faz 14)

`src/worldmodel/provenance.py > provenance_report(patient, drug)`, bir
simülasyonda kullanılan HER parametreyi üç kategoriye ayırıyor:
📚 **literatür** (yayınlanmış kaynak), ⚠️ **varsayım** (yönü gerçek
fizyolojiye dayanan ama kalibrasyon gerektiren temsili değer), 👤
**hasta verisi**. Sınıflandırma, `CALIBRATION_REPORT.md`'deki tabloyla
BİREBİR TUTARLI tutuluyor (biri insan-okunur rapor, biri programatik
sorgu -- aynı gerçeğin iki görünümü).

Streamlit arayüzünde, her simülasyon sonucunun altında "🔍 Bu sonuç
neye dayanıyor?" adında katlanabilir bir bölüm var -- kullanılan her
parametreyi bu üç etiketle birlikte bir tabloda gösteriyor (Vivax'ın
"Güven ve Performans Skoru" ürününe doğrudan referans veren bir özellik).

**Bilinen sınır:** `provenance_report`, ilacı `display_name` ile
eşleştiriyor (Drug nesnesi hangi yaml girdisinden geldiğini ayrıca
taşımıyor) -- katalogda olmayan (ör. elle oluşturulmuş deneysel) bir
ilaç `"sınıflandırılmamış"` (❔) olarak işaretlenir, çökme olmaz (test
edildi).

### Klinik Rapor Çıktısı ve Arayüz Cilası (Faz 15)

`src/worldmodel/report.py > export_report(...)`, bir simülasyon
sonucunu (hasta bilgisi, ilaç bilgisi, özet, CircAdapt grafiği,
sadece "literatür" olarak işaretlenen parametrelerden oluşan kaynakça)
`fpdf2` ile PDF'e aktarıyor. Rapor gerçek bir hastane raporu formatını
taklit ediyor AMA her sayfanın hem üstünde hem altında, **hiçbir
parametreyle kaldırılamayan** kırmızı bir uyarı bandı var: *"SİMÜLASYON
SONUCU -- EĞİTİM / ARAŞTIRMA AMAÇLIDIR. KLİNİK KARAR ALMA İÇİN
KULLANILAMAZ."* (`_ClinicalReportPDF.header()`/`.footer()` hook'ları
her sayfada otomatik basıyor, çağıranın atlaması mümkün değil).

**Türkçe karakter notu:** FPDF'in varsayılan çekirdek fontları (Helvetica/
Arial core) Türkçe karakterleri (ı, ğ, ş, ç, ö, ü, İ) desteklemiyor --
Windows sistem fontundan gerçek bir Arial TTF Unicode modunda yüklenerek
düzeltildi (`_add_unicode_fonts`). İlk taslakta disclaimer metni bu
yüzden aksansız yazılmıştı ("AMAClIDIR" gibi) -- Unicode font eklendikten
sonra düzeltildi ("AMAÇLIDIR").

Streamlit arayüzü artık tek uzun akış değil, **5 sekmeye** ayrıldı:
🧑‍⚕️ Hasta Girişi (temel bilgiler + gelişmiş: böbrek/karaciğer/elektrolit/
komorbidite -- Faz 8/11/12'nin arayüzde daha önce hiç açığa çıkmayan
alanları artık burada canlı ayarlanabiliyor), 💊 İlaç Seçimi, 📊
Simülasyon, 🫀 CircAdapt Sonuçları, 📄 Rapor İndir. Sekmeler arasında
veri akışı `st.session_state["sim"]` üzerinden değişmeden devam ediyor.

---

## ROADMAP — Adım Adım Genişletme Planı

Bu proje şu an **çalışan bir iskelet** halinde. Aşağıdaki fazları
sırayla, VS Code içinde Claude Code eklentisini kullanarak
genişleteceğiz. Her faz için önerilen Claude Code promptunu da yazdım
— editördeki Claude Code sohbet paneline bunu yapıştırıp
başlayabilirsin.

### Faz 0 — Ortam Kurulumu (bugün, 15 dk)
- [ ] VS Code'da bu klasörü aç (`code vivax_world_model_demo`)
- [ ] VS Code'a şu eklentileri kur: **Python** (Microsoft), **Claude Code**
- [ ] Sanal ortamı kur ve `requirements.txt`'i yükle (yukarıdaki komutlar)
- [ ] `python app.py` çalıştığını doğrula
- [ ] `git init` yap, ilk commit'i at

### Faz 1 — Mevcut Mimariyi Anla (bugün, 20 dk)
Kodu satır satır oku: `patient.py` → `pk.py` → `pd.py` → `simulation.py` → `viz.py`.
Mülakatta bu akışı anlatabilmen lazım.

**Claude Code promptu:**
> "src/worldmodel klasöründeki her dosyayı oku ve bana PK'dan PD'ye,
> oradan Monte Carlo'ya veri akışını adım adım, basit bir dille anlat."

### Faz 2 — İkinci İlaç Etkileşimi Ekle (interaktif — stretch goal)
Şu an tek ilaç var. Gerçekçilik için: aynı anda iki ilaç verildiğinde
etkileşim (biri diğerinin etkisini güçlendirsin/zayıflatsın) ekle.

**Claude Code promptu:**
> "simulation.py içine run_monte_carlo_multi_drug adında yeni bir
> fonksiyon ekle: iki ilacın aynı anda verildiği, etkilerinin additive
> (toplamsal) birleştiği bir versiyon. Mevcut testleri bozma, yeni
> testler de ekle."

### Faz 3 — Zamana Yayılı Olaylar (stretch goal)
Şu an tek doz, tek zaman noktası var. Bunun yerine 24 saatlik bir
"hasta günü" simüle et: sabah ilaç, öğlen başka bir olay (örn. egzersiz
-> nabız artışı), gece uyku (bazale dönüş).

**Claude Code promptu:**
> "simulation.py'a bir 'event' sistemi ekle: [(saat, event_tipi,
> parametre), ...] şeklinde bir olay listesi alıp, her olayın vitalleri
> nasıl etkilediğini zaman çizelgesinde gösteren bir fonksiyon yaz."

### Faz 4 — Belirsizlik/Güven Skoru (Vivax'ın vurguladığı nokta)
Sitede "her klinik karar için kalibre edilmiş güven skoru" vurgusu var.
Simülasyon sonuçlarından bir "güven/risk skoru" hesapla ve bunu
Streamlit arayüzünde göster (zaten kısmen `summarize()` fonksiyonunda var).

**Claude Code promptu:**
> "simulation.py'daki summarize fonksiyonunu genişlet: 0-100 arası bir
> 'klinik güven skoru' hesapla (düşük varyans = yüksek güven, mantığını
> sen belirle ve docstring'de açıkla). streamlit_app.py'a bunu bir
> gösterge (gauge) olarak ekle."

### Faz 5 — Gerçek Veri Bağlantısı (opsiyonel, ileri seviye)
Sentetik değil, halka açık bir farmakokinetik veri setiyle (örn.
PubChem, DrugBank açık verileri) parametreleri kalibre et.

### Faz 6 — Sunum Hazırlığı (mülakat/demo günü)
- [ ] `streamlit run streamlit_app.py` ile canlı demo prova et
- [ ] README'yi 1 dakikalık bir "elevator pitch" haline getir
- [ ] GitHub'a push'la, repo linkini hazır tut

**Claude Code promptu:**
> "Bu projeyi bir mülakatta 2 dakikada nasıl anlatacağımı, hangi
> sırayla hangi dosyayı göstereceğimi maddeler halinde yaz."

---

## Neden Bu Mimari? (Mülakatta Anlatım İçin)

1. **Veri ile mantık ayrık** (`patient.py` sadece veri, `pk.py`/`pd.py` sadece denklem) — gerçek yazılım mühendisliğinde "separation of concerns" denir.
2. **Config-driven** (`patients.yaml`, `drugs.yaml`) — kod değiştirmeden yeni hasta/ilaç eklenebilir, yani genişletilebilir.
3. **If-else yok** — PK/PD gerçek fiziksel/biyolojik denklemler, hastanın kilosu gerçekten hesaba giriyor.
4. **Monte Carlo** — tek bir kesin cevap yerine, gerçekçi bir olasılık dağılımı üretiyor (Vivax'ın "kalibre edilmiş güven skoru" felsefesiyle örtüşüyor).
5. **Test edilebilir** (`tests/`) — mühendislik disiplinini gösteriyor.
