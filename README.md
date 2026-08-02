# Mini Klinik Dünya Modeli — Sanal Hasta Simülasyonu

Vivax'ın Acudx mimarisinden ilham alan, gerçekten çalışan bir kavram-kanıtı
(proof-of-concept). PK/PD (farmakokinetik/farmakodinamik) Monte Carlo
simülasyonunu, CircAdapt (gerçek kalp-damar mekaniği motoru) ile birleştirir.

"Dünya modeli" burada teknik bir terim: bir hastanın **durumunu** (nabız,
tansiyon, kalp mekaniği) alıp, bir **aksiyon** (bir ilaç verilmesi)
uyguladığında ortaya çıkan **yeni durumu** hesaplayan bir sistem. If-else
zincirleriyle değil, gerçek farmakoloji denklemleri ve gerçek bir kalp
mekaniği motoruyla.

## Canlı demo

Uygulamayı doğrudan tarayıcıdan açmak için: https://klinik-pkpd-world-model-base-ahrm47cwvjar9cgogclrz9.streamlit.app/

## İçindekiler

1. [Genel bakış](#1-genel-bakış)
2. [Mimari haritası](#2-mimari-haritası)
3. [Motor 1 — PK/PD](#3-motor-1--pkpd)
4. [Monte Carlo + doz önerisi](#4-monte-carlo--doz-önerisi)
5. [Motor 2 — CircAdapt](#5-motor-2--circadapt)
6. [Bulunan mühendislik hataları](#6-bulunan-mühendislik-hataları)
7. [Hasta & ilaç verisi](#7-hasta--ilaç-verisi)
8. [Arayüz turu](#8-arayüz-turu)
9. [Dünya Modelini Gözlemle](#9-dünya-modelini-gözlemle)
10. [Dürüstlük & izlenebilirlik](#10-dürüstlük--izlenebilirlik)
11. [Testler](#11-testler)
12. [Sınırlar — ne YAPMIYOR](#12-sınırlar--ne-yapmıyor)
13. [Tüm formüller — referans](#13-tüm-formüller--referans)
14. [Kurulum & çalıştırma](#14-kurulum--çalıştırma)
15. [Proje yapısı](#15-proje-yapısı)
16. [N-İlaç (Polifarmasi) Genellemesi](#16-n-ilaç-polifarmasi-genellemesi)
17. [JEPA — Öğrenilmiş Dünya Modeli](#17-jepa--öğrenilmiş-dünya-modeli)

---

## 1. Genel bakış

İki farklı simülasyon motorunu birbirine bağlar:

- **PK/PD motoru** (kendi yazdığımız): bir ilacın vücutta nasıl dağılıp
  atıldığını (farmakokinetik) ve bunun nabız/tansiyona ne kadar etki
  ettiğini (farmakodinamik) matematiksel olarak hesaplar. Hızlıdır, saf
  matematik.
- **CircAdapt** (bağımsız, akademik bir motor): kalbin gerçek mekanik
  davranışını — kasılma, kapakçıklar, basınç-hacim ilişkisi — fizik
  denklemleriyle simüle eder. Yavaştır ama gerçek bir kalp-damar sistemi
  modelidir.

Bu ikisi, PK/PD'nin ürettiği "ilaç etkisi" sayısını CircAdapt'in ilgili
mekanik parametresine (kontraktilite veya damar direnci) aktararak
birbirine bağlanır. Sonuç: "esmolol verilirse nabız kaç olur" sorusunun
hem istatistiksel (yüzlerce sanal hasta denemesi) hem mekanik (gerçek
kalp fiziği) cevabı.

## 2. Mimari haritası

```
Patient / Drug (configs/patients.yaml, drugs.yaml)
        │
        ├── pk.py            konsantrasyon(t) hesabı — tek/iki kompartman
        ├── pd.py            Emax etki hesabı, Keo gecikmesi, elektrolit çarpanları
        │
        ├── simulation.py                    Monte Carlo (300×), recommend_dose, run_reference_trace
        └── integrate_drug_with_circadapt.py etkiyi CircAdapt (VanOsta2024) parametrelerine bağlar
                │
                ├── clinical_metrics.py   EF / CO hesabı + yeşil/sarı/kırmızı sınıflama
                ├── report.py             PDF rapor (Türkçe karakter destekli)
                └── streamlit_app.py      6 sekmeli tarayıcı arayüzü
```

Her iki motor da **aynı** girdilerden (patient, drug) besleniyor ama
birbirinden bağımsız çalışıyor — PK/PD saniyenin altında, CircAdapt birkaç
saniyede sonuç üretiyor. Bu hız farkı, arayüzdeki tasarım kararlarının
çoğunu belirliyor (CircAdapt hep bir butonun arkasında, otomatik
tetiklenmiyor).

## 3. Motor 1 — PK/PD

### Farmakokinetik (PK) — "ilaç vücutta nasıl dağılır/atılır"

Vücut, ilaç için tek bir karıştırılmış "kova" (kompartman) gibi düşünülür.
İki süreç var, ikisi de **birinci derece kinetik** kuralına uyar — birim
zamanda ne kadarının değiştiği, o an ne kadar mevcut olduğuyla orantılıdır:

```
dA_depot/dt = -ka · A_depot          (emilim bölgesindeki miktar azalıyor)
dA_body/dt  =  ka·A_depot - ke·A_body (kandaki miktar: girer - çıkar)
```

Bu iki diferansiyel denklemin kapalı-form çözümü:

```
C(t) = (Doz / Vd) × [ka / (ka − ke)] × (e^(−ke·t) − e^(−ka·t))
```

Yani formül rastgele seçilmiş bir eğri değil — **iki üstel sürecin farkı**:
biri "içeri giriyor" (emilim, `ka` hızıyla), biri "dışarı çıkıyor"
(eliminasyon, `ke` hızıyla). Neden üstel? Böbrekten filtrasyon veya
karaciğerdeki enzimlerle parçalanma gibi süreçler klinik dozlarda genelde
doymaz — atılan miktar, o an kanda ne kadar ilaç varsa onunla orantılıdır,
bu da doğrudan üstel azalmaya götürür.

Bazı ilaçlar (esmolol gibi) IV bolus olarak verildiği için
**iki-kompartmanlı** model de desteklenir — emilim fazı yok, santral
kompartmandan (kan) periferik dokuya geçiş (`k12`/`k21`) ve santralden
eliminasyon (`k10`) var. Bu, 2×2'lik bağlı bir denklem sistemi — çözümü
tek üstel değil, **iki** üstelin toplamı (biexponential: hızlı dağılım
fazı + yavaş eliminasyon fazı).

Böbrek/karaciğer yetmezliği, sadece o organdan atılan ilaçların
eliminasyonunu yavaşlatır — `organ_function_adjusted_ke()`, ilacın
`renal_clearance_fraction`/`hepatic_clearance_fraction`'ına göre `ke`'yi
orantılı düşürür. Esmolol bu iki değeri de 0.0 alır çünkü kanda bulunan
eritrosit esterazlarıyla parçalanır — böbrek/karaciğer fonksiyonundan
tamamen bağımsızdır (bilinçli bir modelleme kararı, unutulmuş bir alan
değil).

### Farmakodinamik (PD) — "bu konsantrasyon ne kadar etki yapar"

Emax formülü keyfi değil — **ilaç-reseptör bağlanma denge kinetiğinden**
gelir (Michaelis-Menten enzim kinetiğiyle matematiksel olarak birebir
aynı denklem). İlaç, hücre yüzeyindeki reseptörlere bağlanır; etki, o an
dolu olan reseptör yüzdesiyle orantılıdır:

```
etki_oranı = sensitivity × C / (EC50 + C)      [0 – 1.3 aralığına sabitlenir]
nabız      = bazal_nabız     − Emax_hr  × etki_oranı_nabız
tansiyon   = bazal_tansiyon  − Emax_sbp × etki_oranı_tansiyon
```

Neden bir tavana (`Emax`) doğru düzleşiyor? Reseptör sayısı sınırlı —
hepsi dolduğunda daha fazla ilaç eklemek etkiyi artıramaz (doygunluk).
`EC50`, "reseptörlerin yarısının dolu olduğu" konsantrasyon — düşük EC50,
reseptörlerin ilaca az miktarda bile güçlü tutunduğu anlamına gelir.

Nabız ve tansiyon etkisi **farklı zamanlarda** tepe yapabilir — bunun için
"etki bölgesi" (effect-compartment / `Keo`) gecikme modeli var: kandaki
ilaç, etki yerine (örn. kalp dokusu) anında değil, difüzyonla ulaşır.

```
Ce[i+1] = Cp[i] + (Ce[i] − Cp[i]) × e^(−keo·Δt)
```

Elektrolitler de PD'ye girer: hiperkalemi (yüksek potasyum) AV iletimini
yavaşlatır, anormal kalsiyum kontraktiliteyi değiştirir — bu iki çarpan
CircAdapt tarafına da aktarılır.

## 4. Monte Carlo + doz önerisi

Gerçek hastalar aynı ilaca aynı hızda yanıt vermez. Tek bir "kesin" sayı
yerine, aynı hasta+ilaç kombinasyonunu **300 kez** — her seferinde iki
değeri rastgele yeniden örnekleyerek — çalıştırıp gerçekçi bir sonuç
dağılımı üretiyoruz:

- **ke** (eliminasyon hızı) — log-normal dağılımdan, σ=0.25 ile
- **sensitivity** (bireysel duyarlılık) — log-normal dağılımdan, σ=0.30
  ile; 1.0 = ortalama hasta

Buna ölçüm gürültüsü de eklenir. Sonuçtan çıkan klinik özet: ortalama en
düşük nabız, %5–%95 persentil aralığı, ve **bradikardi riski** (denemelerin
kaçında nabız <50 bpm'e indiği, yüzde olarak).

`recommend_dose()`, 1–20mg aralığında 20 aday doz tarayarak, bradikardi
riskini %5 eşiğinin altında tutan **en yüksek** dozu önerir. Buna iki
katman daha eklenir:

- **Mekanik risk**: CircAdapt sonucu varsa, LVEDV baseline'a göre %20'den
  fazla artıyorsa "aşırı önyük" riski var demektir — istatistiksel öneri
  ×0.7 ile aşağı çekilir.
- **Polifarmasi & elektrolit uyarısı**: hasta başka bir ilaç kullanıyorsa
  ya da laboratuvar değerleri normal aralık dışındaysa, ayrıca metin
  uyarısı eklenir (dozu otomatik değiştirmez).

## 5. Motor 2 — CircAdapt

CircAdapt (VanOsta2024 modeli), bu projede kullanılan **bağımsız,
akademik** bir simülasyon motoru — kasılma, kapakçık açılıp kapanması,
basınç-hacim ilişkisini gerçek fizik denklemleriyle çözer. **Sadece kalbi
değil, tüm dolaşım döngüsünü** modeller: 4 kalp odacığı (Cavity), sistemik
ve pulmoner damar yatakları (ArtVen), kapakçıklar (Valve), kalp siklusu
zamanlaması (Timings) ve otomatik basınç/hacim dengeleme mekanizması
(PressureFlowControl / PFC — kaba bir baroreseptör-refleks benzetmesi).
Bu projede odak, kardiyolojide en çok izlenen odacık olan **LV / sol
karıncığa** (vücuda kan pompalayan ana odacık) verilse de, motorun kendisi
tüm sistemi hesaplıyor.

### İlaç sınıfına göre farklı mekanizma

| İlaç sınıfı | Hedeflenen CircAdapt parametresi | Fizyolojik anlamı |
|---|---|---|
| `beta_blocker` | `General.t_cycle` + `Patch.Sf_act` + `Timings.c_tau_av1` | nabız yavaşlar + kontraktilite azalır + AV iletim gecikmesi artar |
| `positive_inotrope` | `General.t_cycle` + `Patch.Sf_act` + `Timings.c_tau_av1` | nabız değişir + kontraktilite artar + AV iletimi de aynı yönde değişir |
| `vasodilator` | `General.t_cycle` + `ArtVen.p0[0]` | nabız değişir + sistemik direnç azalır (kontraktiliteye/AV iletimine dokunmaz) |

**Faz 5 notu (`Timings.c_tau_av1`):** Bu parametre, hastanın KENDİ elektrolit
durumunun (potasyum -- bkz. `apply_patient_electrolytes_to_circadapt`)
AV düğümü iletimini etkilediği AYNI kanal. Beta-bloker/pozitif inotrop
sınıfındaki bir ilaç ile hiperkalemi artık gerçekten aynı fiziksel yoldan
birikiyor -- önceden (Faz 1-4) ilaç etkisi SADECE `t_cycle` üzerinden
uygulanıyordu, hastanın elektrolit durumundan tamamen bağımsız bir
kanaldan. **Dürüst kısıt (ölçüldü, tahmin değil):** CircAdapt bir 0D/lumped
dolaşım modeli, gerçek AV BLOĞU (atlanan atımlar) fizyolojisini
modellemiyor -- izole testte `c_tau_av1`'i 2x büyütmek EDV'de görünür fark
YARATMADI, 5x'te GERÇEK bir fark ölçüldü (EDV 120.26→135.59 mL), 10x'te
model sayısal olarak çöktü. Mekanizma çalışıyor ama varsayılan hasta/ilaç
büyüklüklerinde ürettiği değişim bu görünürlük eşiğinin altında kalıyor
(bkz. `CALIBRATION_REPORT.md` §5).

### Bazal nabız kalibrasyonu

Önemli bir gerçek bulgu: CircAdapt'in bazal kalp hızı, uzun süre hastanın
**kendi** `baseline_hr`'ını hiç yansıtmıyordu — her zaman CircAdapt'in
kendi jenerik varsayılanından (~70.6 bpm) başlıyordu; ilacın sadece
*fraksiyonel* etkisi bunun üzerine uygulanıyordu. Yani farklı bazal nabza
sahip iki hasta, CircAdapt'te aynı bazal değerden başlıyordu — sessizce
yanlış bir sonuçtu. Çözüm: `calibrate_circadapt_to_patient(model, patient)`
fonksiyonu, `General.t_cycle = 60/baseline_hr` atamasını, ilaç/elektrolit/
komorbidite etkilerinden **önce**, her CircAdapt çalıştırmasının başında
yapıyor.

### Klinik metrikler: EF ve CO

Ham basınç/hacim sayıları yerine, kardiyolojinin gerçekte kullandığı iki
standart metrik hesaplanıyor ve yeşil/sarı/kırmızı olarak
sınıflandırılıyor:

```
EF (%)    = (EDV − ESV) / EDV × 100     normal ≥55, hafif azalmış 40–54, düşük <40
CO (L/dk) = (EDV − ESV) × HR / 1000     normal aralık 4–8 L/dk
```

EDV/ESV = diyastol/sistol sonu hacmi (kalbin dolduğu/boşaldığı andaki en
yüksek/düşük hacim). Kalp yetmezliği komorbiditesiyle test edildiğinde EF
gerçekten düşük çıkıyor (~%41–44, sarı) sağlıklı hastaya (~%57, yeşil)
kıyasla.

## 6. Bulunan mühendislik hataları

CircAdapt'in dokümante edilmemiş davranışları, izole test betikleriyle
keşfedildi. Hepsi *sessizce* yanlış sonuç üretiyordu (hata fırlatmadan).

**Parametre ataması sessizce geri dönüyor.** `model[bileşen][parametre] =
dizi` ataması, bazı parametrelerde (`Patch.Sf_act`, `ArtVen.p0`,
`Timings.c_tau_av1`) yeniden atama SONRASI eski değere geri dönüyordu —
hata vermeden. Çözüm: sadece yerinde (in-place) mutasyon kalıcı oluyor,
yeniden atama yapılmamalı.

**PFC, manuel damar direnci değişikliklerini nötralize ediyor.**
`PressureFlowControl`, `model.run(stable=True)` sırasında `ArtVen`
direncini kendi sabit hedef basıncına göre otomatik yeniden ayarlıyor —
elle yapılan `ArtVen.p0` değişikliğini görünmez şekilde iptal ediyordu.
Çözüm: akut etkiler için `PFC.is_active = False`; kronik durumlar
(hipertansiyon) için `PFC.p0`'ı da orantılı değiştirmek (yeni bir "adapte
olmuş denge" temsil ediyor).

**AV gecikmesi doğrudan atanamıyor.** `Timings.tau_av`, her adımda
`c_tau_av0 + c_tau_av1 × t_cycle` formülünden (bir "law"dan) yeniden
hesaplanıyor — direkt atama sessizce yok sayılıyordu. Çözüm: doğrudan
`tau_av` yerine katsayı `c_tau_av1`'i mutasyona uğratmak.

**Ders:** CircAdapt parametrelerini asla tahmin etmeyin — her zaman izole
bir betikle doğrulayın (bkz. `CLAUDE.md`).

## 7. Hasta & ilaç verisi

### Hasta profilleri (`configs/patients.yaml`)

| Profil | Bazal nabız | Öne çıkan özellik |
|---|---|---|
| Hasta A | 78 bpm | sağlıklı referans hasta |
| Hasta B | 85 bpm | yaşlı, düşük kilo |
| Hasta C | 78 bpm | hiperkalemik (böbrek yetmezliği) |
| Hasta D | 78 bpm | hipokalsemik |
| Hasta E | 88 bpm | sistolik kalp yetmezliği, EF azalmış — yüksek bazal nabız kompanse taşikardiyi temsil ediyor |
| Hasta F | 76 bpm | kronik hipertansif kalp |

### İlaç profilleri (`configs/drugs.yaml`)

| İlaç | Sınıf | Referans doz | Kaynak |
|---|---|---|---|
| Esmolol | beta_blocker | 0.5 mg/kg bolus | 📚 literatür — FDA prescribing info |
| Nikardipin | vasodilator | 0.03 mg/kg (30 mcg/kg IV) | 📚 literatür — Drugs.com / DailyMed |
| Dobutamin | positive_inotrope | 0.037 mg/kg | ⚠️ yaklaşıklık — ~100 ng/mL pik konsantrasyona kalibre |
| Digoksin | positive_inotrope | 0.012 mg/kg | 📚 literatür — böbrek fonksiyonuna duyarlı |
| Örnek Vazodilatör | vasodilator | 10 mg sabit | ⚠️ jenerik örnek |

## 8. Arayüz turu

Streamlit arayüzü 6 sekmeden oluşuyor:

1. **Hasta Girişi** — yaş, kilo, boy, bazal nabız/tansiyon; gelişmiş
   panelde böbrek/karaciğer fonksiyonu, elektrolitler, komorbidite.
2. **İlaç Seçimi** — ilaç, doz (kilo bazlı referansa göre otomatik
   ayarlanan slider aralığı), Monte Carlo deneme sayısı, EC50.
3. **Simülasyon** — PK/PD Monte Carlo sonucu (nabız/tansiyon dağılım
   grafiği), doz önerisi, klinik özet metrikleri.
4. **CircAdapt Sonuçları** — gerçek kalp simülasyonu: LV basıncı, PV loop,
   EF/CO değerlendirmesi, veri kaynağı izlenebilirliği.
5. **Dünya Modelini Gözlemle** — kara kutuyu açan sayfa, aşağıda detaylı.
6. **Rapor İndir** — Türkçe karakter destekli PDF; hasta/ilaç/simülasyon
   özeti + sabit klinik kullanım uyarısı.

## 9. Dünya Modelini Gözlemle

Diğer sekmeler sadece SONUCU gösterir; bu sayfa "durum + aksiyon → yeni
durum" zincirinin HER halkasını, gizlemeden gösterir.

```
[GİRDİ: Hasta Durumu] → [İŞLEME: PK/PD (+CircAdapt)] → [ÇIKTI: Yeni Durum]
   nabız, tansiyon         konsantrasyon + etki oranı      yeni nabız/tansiyon
   kilo, boy                   hesabı                       opsiyonel EF/CO
```

4 bileşen:

- **Durum Tablosu** — `run_reference_trace()`'ın ürettiği, gürültüsüz/tek
  bir PK/PD izinin HER zaman noktasındaki konsantrasyon, etki oranı,
  nabız, tansiyon değerlerini gösteren sıralanabilir tablo.
- **Tek Adımı İncele** — bir zaman noktası seçildiğinde, o ana ait
  GİRDİ/AKSİYON/HESAPLAMA/ÇIKTI zincirini düz metin olarak gösteren
  slider.
- **Monte Carlo Denemesini İncele** — 300 denemeden biri seçildiğinde, o
  denemede rastgele örneklenen `ke`/`sensitivity` değerlerini ve
  popülasyon ortalamasına göre sapmasını gösterir.
- **Gerçek Kalp Modeliyle Göster** — opsiyonel bir buton; CircAdapt'i
  sadece İKİ referans an için (ilaçsız / pik etki anı) çalıştırıp
  EDV/ESV/EF karşılaştırması gösterir. 200 PK/PD noktasının HER birinde
  CircAdapt çalıştırmak dakikalar sürerdi — bu yüzden sayfa **varsayılan
  olarak** hızlı PK/PD motoruyla çalışır, CircAdapt isteğe bağlıdır.

## 10. Dürüstlük & izlenebilirlik

Her parametrenin kaynağı üç kategoriden birine ayrılıyor ve arayüzde/PDF
raporda görünüyor:

- 📚 **Literatür** — yayınlanmış bir kaynağa dayanıyor (FDA prescribing
  info, DailyMed, RxNorm RxCUI).
- ⚠️ **Varsayım** — yönü gerçek fizyolojiye dayanıyor ama kesin sayı
  kalibrasyon gerektiriyor (örn. elektrolit eğimleri).
- 👤 **Hasta verisi** — bu hasta için arayüzde girilen değer.

Bu sınıflandırma `CALIBRATION_REPORT.md`'de detaylandırılıyor. Her PDF
raporun her sayfasında silinemez bir uyarı var: *"SİMÜLASYON SONUCU —
EĞİTİM/ARAŞTIRMA AMAÇLIDIR. KLİNİK KARAR ALMA İÇİN KULLANILAMAZ."*

## 11. Testler

```bash
python -m pytest tests/ -v
```

Kapsam: PK/PD formüllerinin doğruluğu, organ fonksiyonu ayarlaması,
iki-kompartmanlı model, Keo zamanlama ayrımı, doz önerisi mantığı
(istatistiksel + mekanik + polifarmasi + elektrolit), RxNorm/openFDA
doğrulanmış ilaçlar, CircAdapt bazal kalibrasyonu, EF/CO klinik
metrikleri, ve `run_reference_trace`'in deterministik olduğu.

## 12. Sınırlar — ne YAPMIYOR

- Klinik karar almak için kullanılamaz — bir eğitim/araştırma
  kavram-kanıtıdır, gerçek hasta verisiyle/EHR ile entegre değildir.
- Elektrolit ve komorbidite eğim büyüklükleri (örn. hiperkaleminin AV
  iletimi ne kadar yavaşlattığı) **temsilidir** — yönü gerçek fizyoloji,
  kesin sayı bir doz-yanıt çalışmasından kalibre edilmedi.
- CircAdapt, 200 PK/PD zaman noktasının her birinde değil, sadece İKİ
  referans an için çalıştırılıyor — sürekli zaman-çözünürlüklü bir kalp
  simülasyonu değil.
- Polifarmasi modeli toplamsal (additive) bir varsayıma dayanıyor; gerçek
  ilaç-ilaç etkileşimleri çok daha karmaşık olabilir.
- "Örnek Vazodilatör" gibi bazı ilaçlar jenerik/öğretici amaçlı, gerçek
  bir ilacın kalibrasyonunu temsil etmiyor.
- Model, tek-doz/bolus senaryolarını temsil ediyor, sürekli infüzyon
  tedavisini değil. İdame infüzyonu olan ilaçlar (dobutamin, nitroprussid)
  için kullanılan "bolus-eşdeğeri" dozlar birer yaklaşıklıktır.

## 13. Tüm formüller — referans

**PK — konsantrasyon hesabı** (`pk.py > plasma_concentration`)

```
C(t) = (Doz / Vd) × [ka / (ka − ke)] × (e^(−ke·t) − e^(−ka·t))
```

**PK — iki-kompartmanlı, IV bolus** (`pk.py > plasma_concentration_two_compartment`)

```
sum_k = k10 + k12 + k21
disc  = sum_k² − 4·k10·k21
α = (sum_k + √disc) / 2          β = (sum_k − √disc) / 2
C0 = Doz / Vc
A = C0·(α − k21)/(α − β)         B = C0·(k21 − β)/(α − β)

C(t) = A·e^(−α·t) + B·e^(−β·t)
```

**PK — organ fonksiyonu ayarlaması** (`pk.py > organ_function_adjusted_ke`)

```
diğer_yol_payı = 1 − renal_pay − hepatik_pay
ke_ayarlı = ke × (diğer_yol_payı + renal_pay·böbrek_fonk + hepatik_pay·karaciğer_fonk)
```

**PD — Emax/Hill modeli** (`pd.py > emax_effect`)

```
etki_oranı = sensitivity × C / (EC50 + C)         [0 – 1.3 aralığına sabitlenir]
```

**PD — etkiden vitallere** (`pd.py > apply_effect_to_vitals`)

```
nabız     = bazal_nabız     − Emax_hr  × etki_oranı_nabız
tansiyon  = bazal_tansiyon  − Emax_sbp × etki_oranı_tansiyon
```

**PD — Keo gecikme filtresi** (`pd.py > effect_compartment_concentration`)

```
Ce[i+1] = Cp[i] + (Ce[i] − Cp[i]) × e^(−keo·Δt)
```

**PD — elektrolit çarpanları** (`pd.py`)

```
AV_iletim_çarpanı     = 1.0 + 0.3 × max(0, K⁺ − 5.0)
kontraktilite_çarpanı = 1.0 + 0.08 × (Ca²⁺ − 9.5)
```

**Monte Carlo — rastgele örnekleme** (`simulation.py > run_monte_carlo`)

```
ke          = ke_ayarlı  × lognormal(μ=0, σ=0.25)
sensitivity =            lognormal(μ=0, σ=0.30)
bradikardi_riski (%) = ortalama[ min(nabız_izi) < 50 bpm ] × 100
```

**Doz önerisi** (`simulation.py > recommend_dose`)

```
seç: risk(doz) ≤ %5 olan dozlar içinde EN YÜKSEK doz
eğer LVEDV_artışı > %20 (mekanik risk):
    final_doz = seçilen_doz × 0.7
```

**CircAdapt kalibrasyonu** (`integrate_drug_with_circadapt.py > calibrate_circadapt_to_patient`)

```
General.t_cycle = 60 / hasta_bazal_nabız
```

**CircAdapt komorbidite** (`apply_comorbidity_to_circadapt`)

```
kalp_yetmezliği:   Patch.Sf_act ×= 0.6
hipertansiyon:     ArtVen.p0[0] ×= 1.3   VE   PFC.p0 ×= 1.3
```

**Klinik metrikler** (`clinical_metrics.py`)

```
EF (%)    = (EDV − ESV) / EDV × 100
CO (L/dk) = (EDV − ESV) × nabız / 1000
```

## 14. Kurulum & çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# CLI ile
python app.py --patient hasta_a --drug beta_bloker --n 300

# İnteraktif arayüz
streamlit run streamlit_app.py

# Testler
python -m pytest tests/ -v
python -m pytest patient_profile/tests/ -v
```

**`patient_profile/` modülü için ek kurulum notu:** `pytesseract`/`pdf2image`
Python paketleri `pip install -r requirements.txt` ile kurulur, ama bunlar
sadece OCR **fallback** yolu için gerekli (varsayılan yol değil -- bkz.
`patient_profile/file_ingestion.py`). Bu fallback yolunu kullanacaksan,
Python paketlerinin YANINDA sistem seviyesinde **Tesseract OCR** ve
**Poppler** binary'lerinin de ayrıca kurulu olması ve PATH'e eklenmesi
gerekir -- bunlar pip ile gelmez, sessizce eksik kalırlarsa OCR çağrısı
çalışma anında hata verir (bkz. Windows için:
[Tesseract kurulumu](https://github.com/UB-Mannheim/tesseract/wiki),
[Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)).
Ayrıca Gemini API çağrıları için `GEMINI_API_KEY` ortam değişkeni
ayarlanmalı.

## 15. Proje yapısı

```
vivax_world_model_demo/
├── app.py                             # CLI giriş noktası
├── streamlit_app.py                   # İnteraktif tarayıcı arayüzü
├── test_circadapt_installation.py     # CircAdapt kurulum testi
├── integrate_drug_with_circadapt.py   # PK/PD -> CircAdapt entegrasyonu
├── compare_drug_classes_circadapt.py  # 3 ilaç sınıfını CircAdapt'te karşılaştırır
├── compare_pk_models.py               # Tek- vs iki-kompartmanlı PK modeli karşılaştırması
├── plot_keo_effect_separation.py      # HR vs SBP etki zamanlaması (Keo) görselleştirmesi
├── drug_lookup.py                     # RxNorm/openFDA'dan gerçek ilaç verisi çeker
├── compare_polypharmacy.py            # Tehlikeli ilaç kombinasyonu gösterimi
├── compare_electrolyte_effects.py     # Elektrolit durumunun kalp üzerindeki etkisi
├── compare_comorbidity_effects.py     # Aynı doz, farklı hasta/hastalık profili
├── CALIBRATION_REPORT.md              # Hangi parametre gerçek, hangisi varsayım
├── CLAUDE.md                          # Proje kuralları
├── requirements.txt
├── configs/
│   ├── patients.yaml       # Hasta profilleri
│   ├── drugs.yaml          # İlaç profilleri (PK/PD parametreleri, drug_class)
│   └── drugs_verified.yaml # Kaynak izlenebilirlikli (RxCUI/source_url) gerçek ilaçlar
├── src/worldmodel/
│   ├── patient.py           # Patient / Drug veri modelleri
│   ├── pk.py                # Farmakokinetik: konsantrasyon hesabı
│   ├── pd.py                # Farmakodinamik: konsantrasyon -> etki
│   ├── simulation.py        # Monte Carlo motoru + doz önerisi + polifarmasi
│   ├── clinical_metrics.py  # EF / CO hesabı + sınıflama
│   ├── provenance.py        # Veri kaynağı izlenebilirliği / audit trail
│   ├── report.py            # PDF klinik rapor çıktısı
│   └── viz.py                # Grafik üretimi
├── tests/
│   ├── test_pk.py                    # Birim testler
│   └── test_clinical_validation.py   # Gerçek klinik veriyle karşılaştırma
└── outputs/                 # Üretilen grafikler buraya kaydedilir
```

---

## 16. N-İlaç (Polifarmasi) Genellemesi

Proje başlangıçta yalnızca N=1/N=2 ilaç için güvenilir çalışıyordu.
Aşağıdaki, N=1..8 ilaç için genelleme çalışmasının özeti -- tam denetim
(`N_DRUG_AUDIT.md`), literatür araştırması + mimari karar
(`RESEARCH_N_DRUG.md`, ADR-1..6) ve nihai rapor (`N_DRUG_REPORT.md`)
ayrı dosyalarda.

### Seçilen kombinasyon yöntemi

- **İstatistiksel motor (HR/SBP büyüklüğü):** Loewe additivity (doz-
  eşdeğerliği, `pd.py > loewe_combined_effect`) -- literatürde N ilaca
  en iyi genellenmiş, en düşük hesap maliyetli yöntem. `min(Emax)` tavanı
  (en düşük tavanlı TEK ilacın kombinasyonun üst sınırını belirlemesi)
  **kaldırılmadı** -- hiçbir literatür yöntemi (MuSyC dahil) bunu bu
  proje için uygulanabilir şekilde kaldırmıyor (kalibrasyon verisi
  eksikliği) -- bunun yerine kullanıcıya (Streamlit) hangi ilacın tavanı
  belirlediği açıkça gösteriliyor.
- **CircAdapt'e uygulanan parametre çarpanları (t_cycle/Sf_act/ArtVen.p0/
  c_tau_av1):** ilaç-başına, sırayla çarpımsal birikim (Bliss
  independence'ın "kalan fraksiyonların çarpımı" formülasyonuyla örtüşen,
  QSP kardiyovasküler pratiğiyle uyumlu bir yaklaşım) -- N=4 ilaçla 24
  permütasyonun tamamen sıra-bağımsız olduğu çalışma-zamanında doğrulandı.
- **Zıt yönlü ilaçlar** (biri nabzı düşürür, biri artırır): literatürde
  hiçbir standart yöntem yok -- proje kendi mühendislik kararını
  uyguluyor (`pd.py > grouped_loewe_combined_effect`): ilaçlar Emax
  işaretine göre gruplanır, her grup kendi içinde Loewe ile birleştirilir,
  net sonuç gruplar arası (işaretli) toplamdır. Bu **literatürden gelen
  bir yöntem değildir**, kod ve arayüzde açıkça böyle etiketlenir.

### Düzeltilen kritik hatalar

- **AV-blok formül sapması** (N=5'te ölçülen %52): istatistiksel motorun
  AV-iletim çarpanı, CircAdapt'in gerçek ilaç-başına çarpımsal formülüyle
  hizalandı (bkz. §5, `av_conduction_cumulative_multiplier`).
- **PK-DDI kaybı N≥3'te**: `run_polypharmacy_simulation_loewe()`'ye
  eksik olan `drug_keys`/`pk_interaction_matrix` parametreleri eklendi.
- **t_cycle'ın hiç kontrol edilmemesi**: CircAdapt'in en kırılgan
  parametresi (3.0x'te çöküyor, `Sf_act`'in 100x'i ve `ArtVen.p0`'ın
  500x+'inden çok daha düşük) için genelleştirilmiş bir ön-kontrol
  eklendi (`cumulative_parameter_multipliers`/`circadapt_instability_risk`).
- **PD interaction teriminin asimetrisi**: N≥3'te simetrikleştirildi
  (N=1/2 davranışı `symmetric_interaction_terms=False` varsayılanıyla
  korunarak).

### Bilinen sınırlar (dürüstçe)

- `min(Emax)` tavanı N büyüdükçe daha sık bağlayıcı hale gelir -- bu
  kaldırılmadı, sadece görünür kılındı.
- Zıt yönlü kombinasyon birleştirme kuralı literatür kaynaklı değil.
- İstatistiksel motorun Monte Carlo (rastgele örneklenmiş) yolu,
  paylaşılan bir RNG akışı kullandığı için ilaç SEÇİM SIRASINDAN
  bit-düzeyinde bağımsız DEĞİL (istatistiksel olarak aynı dağılım, ama
  aynı dizi değil) -- CircAdapt tarafı (RNG'siz) tam sıra-bağımsız.
- `ArtVen.p0[0]` için test edilen aralıkta (0.01x-500x) hiç çöküş
  gözlenmedi -- bu "sonsuz güvenli" anlamına gelmez, sadece ölçülen
  aralıkta güvenli demektir.

Detaylar için: `N_DRUG_AUDIT.md` (denetim), `RESEARCH_N_DRUG.md`
(literatür + ADR), `CALIBRATION_REPORT.md` §10 (CircAdapt eşikleri),
`N_DRUG_REPORT.md` (nihai özet).

## 17. JEPA — Öğrenilmiş Dünya Modeli

CircAdapt güvenilir ama yavaş. **JEPA** (Joint Embedding Predictive
Architecture), CircAdapt'in ürettiği 1560 simülasyondan (farklı hasta
profili × ilaç dozu, her biri 40 dakika/16 kare) öğrenerek, fiziği hiç
bilmeden kalbin bir sonraki anını **embedding uzayında** (piksel/ham
veri uzayında değil) tahmin eden bir sinir ağı -- Streamlit'te "JEPA
Dünya Modeli (Deneysel)" sekmesi. 26 trajectory hiç eğitime sokulmadı,
sadece test için ayrıldı.

Mimari: **Encoder** (state → 64 boyutlu embedding), **Target Encoder**
(EMA, gradyan almayan yavaş kopya -- collapse'e karşı ikinci savunma
hattı, birincisi varyans cezası), **Predictor** (embedding uzayında
tahmin), ayrı ve denetimli eğitilen **Decoder** (embedding → EF, CO,
Nabız, LVEDV, LVESV). Değerlendirme, eğitimde hiç görülmemiş 26
trajectory'de **otoregresif rollout** ile yapılır: model kendi
tahminini bir sonraki adımın girdisi yapar, gerçek veri hiç araya
girmez, 16 adım (40 dakika) sonunda MAE/R² ölçülür.

### Delta-tahminden hedef-durum (goal-conditioned) mimariye geçiş

İlk (delta-tabanlı) `Predictor`, `embeddingₜ₊₁ = embeddingₜ +
Predictor(embeddingₜ, actionₜ)` formülüyle sınırsız bir "fark"
ekliyordu -- zayıf/yavaş aksiyonlarda en güvenli çözüm delta≈0'a
yakınsamaktı, bu da otoregresif rollout'ta hata birikimine
(compounding error) yol açıyordu: ilk hareket doğru tahmin ediliyor,
ufuk uzadıkça model gerçek eğriden sürükleniyordu.

Kullanıcının önerisiyle (`worldmodel/learned_dynamics/model.py >
GoalConditionedPredictor`), `transient_integration.py`'nin CircAdapt
tarafında zaten kullandığı "her adımda mutlak hedefi taze hesapla,
önceki adımdan hafıza yok" mantığının embedding-uzayındaki karşılığı
uygulandı:

```
goalₜ = embedding_baseline + goal_net(actionₜ)
embeddingₜ₊₁ = (1-α)·embeddingₜ + α·goalₜ         (α = 0.7, sabit)
```

`embedding_baseline`, trajectory'nin ilaç-öncesi (t=0) karesinin
embedding'i -- rollout boyunca sabit. Kayıp fonksiyonu:
`MSE(tahmini, TargetEnc(stateₜ₊₁)) + 0.05·mean(ReLU(1.0 - std(e)))`
(`train_jepa.py > variance_regularization`).

### Deney günlüğü (`scripts/train_goal_jepa_experiment.py`)

1. **α + gradient clipping taraması** (α ∈ {0.3, 0.5, 0.7}) -- α=0.7
   son-adım isabetinde en iyisi.
2. **Weight decay + embedding L2 cezası** -- varyans cezası sadece ALT
   sınır koyuyordu (std≥1), üst sınır yoktu; L2=1e-3 (weight decay'siz)
   en iyi kombinasyon.
3. **Düşük öğrenme oranı + güçlü L2** -- eğitim ıraksamasını (epoch
   9→17) sadece geciktirdi, önlemedi.
4. **Stabil (EMA) hedef referansı hipotezi** -- `embedding_baseline`'ı
   hızlı değişen online Encoder yerine Target Encoder'dan hesaplama --
   ÇÜRÜTÜLDÜ, daha erken ıraksadı.
5. **Şampiyon ayar (α=0.7, L2=1e-3) × 3 seed** -- sonucun tesadüf
   olmadığını doğrulamak için.

Eğitim, denenen tüm varyantlarda ~epoch 9-17'de "ıraksıyor" (loss
patlıyor) -- kök sebep henüz tam çözülmedi, en iyi val_loss'taki
checkpoint kullanılıyor (200 epoch'un ~%10'u, undertrained).

### 3-seed sonucu (40. dakika R², eski canlı model vs yeni)

| Metrik | Eski | Yeni (3-seed ort.) | Not |
|---|---|---|---|
| EF | 0.990 | 0.995 | gerçek senaryoda erken-zirve gecikmesi bulundu |
| Nabız | 0.977 | 0.991 | tutarlı, 3/3 seed'de kazanç |
| LVEDV | 0.943 | 0.986 | tutarlı, 3/3 seed'de kazanç |
| LVESV | 0.985 | 0.996 | tutarlı, MAE de küçük |
| CO | 0.209 | -0.48 … +0.21 | seed'e göre değişken, güvenilmez |

### Canlıda çalışan karar: hibrit model

Gerçek bir senaryoda (Esmolol + varsayılan hasta) test edilince,
pooled/havuzlanmış R²'nin gizlediği iki sorun bulundu: **EF**, ilacın
erken zirvesinde (0-5 dk) gerçek eğrinin TERSİ yönde hareket ediyordu
(α-karışımının yarattığı gecikme/aşırı-tepki etkisi); **CO** seed'ler
arası tutarsızdı (1/3 seed'de R²=-0.475'e çöktü). Bu yüzden
`streamlit_app.py` **iki modeli paralel çalıştırıp** alan bazında
birleştiriyor (`JEPA_FIELD_SOURCE`): **Nabız + LVEDV + LVESV** → yeni
(goal-conditioned) model, **EF + CO** → eski (delta) model
(`models/dynamics_jepa_transient_1560run_1560data_seed0`, hâlâ diskte,
kaldırılmadı).

### Bilinen sınırlar (dürüstçe)

- Goal-conditioned modelin eğitimi hiçbir varyantta 200 epoch'u
  tamamlayamadı (~epoch 9-17'de ıraksıyor) -- undertrained, potansiyel
  olarak daha da iyileştirilebilir.
- CO tahmini, mimari ne olursa olsun (eski ya da yeni) güvenilir
  değil -- eski modelde de pooled R²=0.126 gibi düşük.
- Tek-trajectory testlerde R², gerçek değerlerin varyansı düşükse
  (örn. hızlı sönümlenen bir ilaç etkisi) yanıltıcı olabilir -- MAE'ye
  de bakılmalı (bkz. Streamlit JEPA sekmesindeki uyarı kutusu).
- Sadece **beta-bloker** ve **pozitif inotrop** sınıfı ilaçlarla
  eğitildi.

Detaylı deney sonuçları: `logs/SUMMARY_1560.md`,
`proje_detayli_anlatim.html` Bölüm 8.
