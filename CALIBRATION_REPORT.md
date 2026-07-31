# Kalibrasyon Raporu

Bu rapor, projedeki HER parametrenin nereden geldiğini (gerçek literatür /
temsili varsayım / kullanıcı girdisi) ve modelin çıktılarının yayınlanmış
klinik veriyle nerede uyuştuğunu, nerede uyuşmadığını **dürüstçe**
belgeliyor. Amaç: "kör kör her şey doğru" iddiası yerine, "şuraya kadar
güvenebilirsin, burada dikkatli ol" diyen bir mühendislik duruşu.

Canlı, otomatik bir versiyonu için: `provenance_report()` ve Streamlit
arayüzündeki "Bu sonuç neye dayanıyor?" bölümü.

---

## Durum Özeti (navigasyon -- detaylar aşağıdaki bölümlerde, bu sadece bir harita)

| # | Boşluk | Durum | Detay |
|---|---|---|---|
| 1 | PD kalibrasyonu (nicardipine/dobutamine/nitroprusside emax/ec50) | 🔶 Kısmen | Nitroprussid EC50'si pediatrik literatürden türetildi (§1c); dobutamin/nikardipin literatür arandı, konsantrasyon-bazlı tek nokta çıkarılamadı, sonuç belgelendi (§1d, §1e). Esmolol ec50 daha önce §1a'da. |
| 2 | PK-seviyeli ilaç-ilaç etkileşimi (klerens/AUC) | ✅ Tamamlandı | `pk_interaction_adjusted_ke()`, esmolol→digoksin (Kessler 1987, §5e). **Bu turda düzeltildi (Bulgu 1):** `app.py` (CLI) ve `streamlit_app.py`, `drug_keys`/`pk_interaction_matrix` parametrelerini artık `build_pk_interaction_matrix()` ile otomatik tespit edip `run_polypharmacy_simulation()`'a geçiriyor -- hem CLI çıktısına hem Streamlit arayüzüne (İlaç Seçimi sekmesi, `st.info`) etkileşimin aktif olduğunu, oranını ve kaynağını gösteren bir bilgi satırı eklendi. AppTest ile esmolol+digoksin seçilip canlı doğrulandı (bkz. §5e). |
| 3 | Discrete (all-or-nothing) AV blok | ✅ Tamamlandı | İstatistiksel motor + CircAdapt ön-kontrolü (§8). **Bu turda doğrulanan bulgu:** istatistiksel motorun (Monte Carlo) kaçış-ritmi tetiklenmesi mevcut Streamlit slider aralıklarıyla (K≤8.0) canlı doğrulandı; CircAdapt sekmesindeki "AV blok tetiklendi" mesajı ise mevcut ilaç seçenekleri + slider üst sınırlarıyla (K≤8.0, doz≤3x) ULAŞILAMIYOR (gereken çarpan ~3.0'ı bu aralıkta hiçbir kombinasyon aşmıyor, K≈10+ gerekiyor) -- kod doğru çalışıyor (birim testleriyle kanıtlı), ama UI'dan tetiklenemiyor. |
| 4 | Sürekli infüzyon PK modeli (dobutamin, nitroprussid) | ✅ Tamamlandı | `plasma_concentration_infusion()` (Rowland & Tozer), §7. `run_reference_trace()` bilinçli olarak dışarıda bırakıldı (aynı bölümde not var). |
| 5 | Hücresel elektrofizyoloji/aritmi riski katmanı (ORd/CiPA tarzı) | ❌ Bilinçli kapsam dışı | Sohbette tartışıldı (organ-mekanik model ile hücresel elektrofizyolojinin farklı katmanlar olduğu netleştirildi), hiç implementasyon denemesi yapılmadı. |
| 6 | 3+ ilaç için otomatik doz önerisi | ✅ Tamamlandı | `recommend_polypharmacy_dose_scale()` -- ortak ölçek katsayısı, grid-scan (bisection değil). |
| — | `patient_profile/` modülü (LLM ile hasta dosyası çıkarımı) | ✅ Tamamlandı | Ayrı bir özellik (gap listesinde değildi) -- şema, extraction, temporal merge, validasyon, kovaryat eşleme (Cockcroft-Gault/Child-Pugh), onay ekranı. `Patient.known_av_block_degree` üzerinden #3'e kısmen bağlandı (sadece "third"). |

---

## 1. İlaç Parametreleri — Kaynak Durumu

| İlaç | PK parametreleri | PD parametreleri (emax/ec50) | Doğrulama |
|---|---|---|---|
| **Esmolol** (`beta_bloker`) | ✅ Gerçek (Wiest 1991, FDA label; openFDA ile bağımsız doğrulandı) | 📚/⚠️ **Faz 3'te kısmen kalibre edildi** (ec50 -- bkz. §1a); emax_hr/emax_sbp hâlâ temsili | ✅ Onset/duration testleri geçti |
| **Nikardipin** | ✅ Gerçek (Clinical Pharmacokinetics 2006, DailyMed) | ⚠️ Temsili -- **literatür arandı, konsantrasyon-bazlı EC50 çıkarılamadı** (bkz. §1e) | ❌ Doğrulanmadı (kapsam sadece esmolol) |
| **Dobutamin** | ✅ Gerçek (Kates & Leier 1978), ama **dose_mg_per_kg bolus-eşdeğeri bir yaklaşıklık** (gerçek klinik kullanım sadece infüzyon) | ⚠️ Temsili -- **literatür arandı, incelenen aralıkta EC50/Emax tanımsız bulundu** (bkz. §1d) | ❌ Doğrulanmadı |
| **Digoksin** | ✅ Gerçek (standart ders kitabı: t½, Vd, renal atılım fraksiyonu) | ⚠️ Temsili -- **Faz 3'te aranıp bulunamadı** (bkz. §1b) | ❌ Doğrulanmadı |
| **Metoprolol** (`drugs_verified.yaml`) | ✅ Gerçek (FDA etiketi: doz, onset zamanlaması); t½/Vd ders kitabı | ⚠️ Temsili | ❌ Doğrulanmadı |
| **Sodyum Nitroprussid** | ✅ Gerçek (FDA etiketi: t½=2dk, Vd=ekstrasellüler sıvı) ama **dose_mg_per_kg bolus-eşdeğeri bir yaklaşıklık** | 📚/⚠️ **ec50 pediatrik literatürden kısmen kalibre edildi** (bkz. §1c); emax_hr/emax_sbp hâlâ temsili | ❌ Doğrulanmadı |
| **"Örnek Vazodilatör"** (`vazodilator`) | ❌ Tamamen temsili/uydurma | ⚠️ Temsili | ❌ |

**Okuma:** "✅ Gerçek" işaretli parametreler (`ka`/`ke_mean`/`vd_per_kg`/doz),
gerçekten yayınlanmış bir kaynaktan (FDA etiketi, hakemli çalışma, ya da
standart ders kitabı) geliyor -- `configs/drugs.yaml` ve
`configs/drugs_verified.yaml`'daki yorumlarda tek tek kaynak gösterildi.
**`emax_hr`/`emax_sbp`/`ec50` (doz-yanıt büyüklüğü) neredeyse hiçbir
ilaçta gerçek bir doz-yanıt çalışmasından kalibre edilmedi** -- bunlar
"makul görünen, yönü doğru" temsili sayılar. Bu, projenin en büyük tek
kalibrasyon boşluğu. **Faz 3'te esmolol için kısmi bir istisna yapıldı**
(§1a), digoksin için araştırma sonuçsuz kaldı (§1b), **nitroprussid için
pediatrik literatürden kısmi bir EC50 türetildi** (§1c, esmolol'den daha
zayıf bir güven kategorisinde), dobutamin (§1d) ve nikardipin (§1e) için
literatür arandı ama konsantrasyon-bazlı bir EC50 noktası çıkarılamadı --
hepsi dürüstçe belgelendi, aşağıda.

### 1a. Faz 3 — Esmolol EC50 kalibrasyonu (kısmi başarı)

Reilly ve ark. (esmolol farmakodinamiği, *Eur J Clin Pharmacol*), egzersiz
sırasında nabız artışının baskılanması için EC50'yi **infüzyon hızı**
cinsinden veriyor: 113 mcg/kg/dk. Projenin PK modeli EC50'yi **plazma
konsantrasyonu** (mg/L) cinsinden istediği için, kararlı-durum ilişkisiyle
(`Css = infüzyon_hızı / klerens`) dönüştürüldü -- klerens de zaten
literatürden (Wiest 1991) gelen `ke_mean * vd_per_kg = 9.2 L/saat/kg`:

```
113 mcg/kg/dk × 60 = 6.78 mg/kg/saat
EC50 = 6.78 / 9.2 ≈ 0.737 mg/L   (önceki temsili değer: 0.03 mg/L -- ~24 kat fark)
```

**Bu bir TÜRETME, doğrudan ölçülmüş bir plazma-EC50 DEĞİL** -- iki dürüst
kısıt var: (1) infüzyon-hızı EC50'sini konsantrasyona çevirmek, klerensin
kararlı-durumda sabit kaldığını varsayıyor; (2) kaynak çalışma EGZERSİZ
sırasındaki nabız artışının baskılanmasını ölçüyor, bizim senaryomuz
(dinlenim bazalinin düşürülmesi) birebir aynı fizyolojik durum değil.
`emax_hr`/`emax_sbp` (etkinin BÜYÜKLÜĞÜ) bu çalışmadan gelmedi, hâlâ ⚠️
temsili -- sadece EC50 (etkinin NE ZAMAN/HANGİ KONSANTRASYONDA belirgin
hâle geldiği) güncellendi.

### 1b. Faz 3 — Digoksin araştırması (sonuçsuz, dürüstçe belgelendi)

Digoksinin AV-düğümü/nabız-yavaşlatma etkisine özel bir EC50/Emax arandı.
Bulunan tek sayısal PD çalışması (PubMed 15032303, "Inotropic effect of
digoxin in humans") **inotropik** (kasılma gücü) etkiyi ölçüyor -- farklı
bir mekanizma/uç nokta, `emax_hr`'nin temsil ettiği kronotropik etki
değil. Tam metin (Springer, oturum açma gerektiriyor) bu oturumda
erişilemedi; özette sadece "denge yarı-ömrü 13 saat" veriliyor, sayısal
EC50/Emax yok. **Karar: yanlış mekanizmadan ödünç alınan bir sayı, hiç
sayı olmamasından daha yanıltıcı olur** -- `emax_hr`/`emax_sbp`/`ec50`
⚠️ temsili bırakıldı. Gelecekte tam-metin erişimi olan biri için bu bir
başlangıç noktası (bkz. `configs/drugs.yaml > digoxin` yorumu).

### 1c. Nitroprussid EC50 kalibrasyonu -- pediatrik literatür, yetişkine ekstrapolasyon

Gregoire ve ark. (PMC4516882, "A hemodynamic model to guide blood pressure
control during deliberate hypotension with sodium nitroprusside in
children") -- MAP (ortalama arter basıncı) için inhibitory sigmoidal Emax
K-PD modeli. ER50'yi (Emax'ın yarısını üreten kararlı-durum infüzyon hızı)
**iki alt-grup** için AYRI ayrı veriyor: yüksek-EC50 grubu 0.34 mcg/kg/dk,
düşük-EC50 grubu 0.103 mcg/kg/dk.

**Varsayım:** iki alt-grubun ARİTMETİK ORTALAMASI ((0.34+0.103)/2=0.2215
mcg/kg/dk) tek bir temsili ER50 olarak kullanıldı -- popülasyon gerçekte
iki alt-gruba ayrılmışken tek nokta seçmek bilgi kaybı yaratıyor, bu
dürüstçe kabul edilen bir basitleştirme.

Esmolol Faz 3'teki (§1a) AYNI `Css=R/Cl` dönüşümü uygulandı -- klerens
zaten FDA etiketinden gerçek: `ke_mean * vd_per_kg = 20.79 * 0.2 = 4.158
L/saat/kg`:

```
0.2215 mcg/kg/dk × 60/1000 = 0.01329 mg/kg/saat
EC50 = 0.01329 / 4.158 ≈ 0.0032 mg/L   (önceki temsili değer: 0.011 mg/L -- ~3.4 kat fark)
```

**Bu, esmolol'den (§1a) DAHA ZAYIF bir güven kategorisinde** -- üç ek
kısıt var (esmolol'de bunlardan hiçbiri yoktu): (1) kaynak popülasyon
**PEDİATRİK**, bu proje YETİŞKİN hastaları simüle ediyor -- doğrudan bir
yaş-grubu ekstrapolasyonu; (2) iki alt-grubun (gerçek bir bimodal
dağılımın) ortalaması alınarak tek noktaya indirgendi; (3) kaynak **MAP**
ölçtü, `emax_sbp`'miz sistolik basıncı temsil ediyor -- birebir aynı
ölçüt değil. `emax_hr`/`emax_sbp` (etkinin BÜYÜKLÜĞÜ) bu çalışmadan
gelmedi, hâlâ ⚠️ temsili -- sadece EC50 güncellendi (bkz. `provenance.py`,
bu kayıt "📚 LİTERATÜR AMA ZAYIF KATEGORİ" notuyla ayrıca işaretlendi,
esmolol/Kessler'in doğrudan-yetişkin kategorisiyle karıştırılmaması için).

### 1d. Dobutamin -- literatür arandı, EC50/Emax tanımsız bulundu

Ahonen ve ark. 2008 (*Clin Drug Investig*, sağlıklı yetişkin gönüllüler,
2.5-10 mcg/kg/dk infüzyon aralığı) plazma konsantrasyonu-nabız
ilişkisinin **incelenen aralıkta DOĞRUSAL** olduğunu, bir doygunluk/Emax
platosunun **GÖZLENMEDİĞİNİ** bulmuş -- yani standart Emax/Hill modeli bu
ilaç için, en azından bu doz aralığında, iyi tanımlı bir EC50 üretmeyebilir
(eğri hiç düzleşmiyorsa, "yarı-maksimum etkiyi üreten konsantrasyon"
kavramı belirsizleşir). Makalenin tam metni ücretli erişim gerektirdiği
için kesin slope/regresyon katsayısı bu oturumda çıkarılamadı.

**Sonuç:** `emax_hr`/`emax_sbp`/`ec50` ⚠️ temsili KALIYOR -- ama artık
"neden hâlâ temsili" sorusunun literatür-destekli bir cevabı var: bu,
sadece "henüz aranmadı" değil, "arandı ve model varsayımı bu ilaç için
sorgulanabilir çıktı" durumu. Tam metne erişimi olan biri için bu, hem
EC50 kalibrasyonu hem de modelin kendisinin (Emax/Hill) bu ilaca uygunluğu
için bir başlangıç noktası.

### 1e. Nikardipin -- dağınık literatür, tek nokta çıkarılamadı

Doz-yanıt çalışmaları ve ayrı bir endikasyonda doz-bazlı ED50 verileri
bulundu, ayrıca tam metnine erişilemeyen bir PK/PD sistem-analizi çalışması
tespit edildi -- ama bunlardan hiçbiri, projenin modelinin ihtiyaç duyduğu
biçimde (konsantrasyon → etki) tek bir EC50 noktasına dönüştürülebilecek
kadar doğrudan/eksiksiz değildi. `emax_hr`/`emax_sbp`/`ec50` ⚠️ temsili
KALIYOR -- bu, ayrı bir takip aramasını (muhtemelen tam-metin erişimi
gerektiren) hak ediyor.

---

## 2. Diğer Fizyolojik Parametreler

| Mekanizma | Durum |
|---|---|
| Böbrek/karaciğer fonksiyonunun ke üzerindeki etkisi | ✅ Yön/varlık gerçek (esmolol=etkilenmez, digoksin=%65 renal), kesin oran (`renal_clearance_fraction`) ders kitabı |
| Keo (etki bölgesi gecikmesi) | ⚠️ Tamamen temsili -- hiçbir ilaçta yayınlanmış bir Keo çalışmasından gelmedi |
| Potasyum -> AV iletim gecikmesi | ✅ Yön gerçek (hiperkalemi AV iletimini yavaşlatır, iyi bilinen fizyoloji), eğim (0.3) temsili |
| Kalsiyum -> kontraktilite | ✅ Yön gerçek (Ca-kontraktilite ilişkisi, iyi bilinen fizyoloji), eğim (0.08) temsili |
| Kalp yetmezliği / hipertansiyon profilleri | ✅ Mekanizma gerçek (KY=azalmış kontraktilite, HT=artmış direnç+basınç setpoint'i), büyüklük (%40, %30) temsili |
| Polifarmasi interaction_matrix | ⚠️ Varsayılan olarak KULLANILMIYOR (saf toplamsal); örnek çarpan (0.5) tamamen temsili |
| İki-kompartmanlı esmolol k10/k12/k21 | ✅ alpha/beta gerçek, ama santral hacim (Vc=0.5 L/kg) VARSAYIM (esmolole özgü ayrı bir Vc yayını bulunamadı) |

---

## 3. Klinik Doğrulama Sonuçları — `tests/test_clinical_validation.py`

Esmolol için, gerçek yayınlanmış klinik çalışmalarla 5 karşılaştırma
yapıldı:

| # | Karşılaştırma | Kaynak | Sonuç |
|---|---|---|---|
| 1 | Onset ≤2 dk | Geriatrik kataraki cerrahisi çalışması: "onset within 2 minutes" | ✅ GEÇTİ |
| 2 | %90 etki ≤5 dk | Aynı çalışma: "90% of steady-state beta-blockade within 5 minutes" | ✅ GEÇTİ |
| 3 | Yüksek doz -> daha uzun etki süresi (göreli ilişki) | Yaşlı hasta bolus çalışması: 50mg~5dk, 100mg~9.5dk | ✅ GEÇTİ |
| 4 | Pik etkide nabız azalma oranı, SVT infüzyon çalışmasına ~%20 toleransla yakın | Esmolol Research Group 1986: bazal HR 139->106 (~%24 azalma) | ✅ GEÇTİ (ama bkz. aşağıdaki uyarı) |
| 5 | 60 dk'da plazma konsantrasyonu <%5 (pik'e göre) | Aynı çalışma -- SÜREKLİ infüzyonla kıyasla | ✅ GEÇTİ (konsantrasyon %2'ye iniyor) |

### ÖNEMLİ UYARI — Test #4'ün Yanıltıcı Olabilecek "Başarısı"

Test #4 başlangıçta **başarısız olması beklenerek** yazıldı: Kaynak
[Esmolol Research Group 1986], SVT hastalarında (bazal HR 139±12),
SAATLERCE süren, titre edilmiş bir SÜREKLİ İNFÜZYONLA (ortalama 97.2
mcg/kg/dk) elde edilen bir sonuç. Bizim modelimiz ise TEK BİR 0.5 mg/kg
BOLUS'u, NORMAL bazal nabızlı (78 bpm, SVT değil) bir hastada simüle
ediyor -- yapısal olarak çok farklı iki senaryo.

Test çalıştırıldığında, pik etkideki nabız azalma ORANI (%27 model vs
%24 çalışma) beklenmedik şekilde ±%20 tolerans içinde çıktı. **Bu bir
doğrulama olarak GÜVENİLİR SAYILMAMALI** -- `emax_hr=25` parametresi bu
çalışmadan türetilmedi (temsili olarak seçildi), yakınlık büyük
olasılıkla TESADÜFİ. Gerçek, sağlam kanıt Test #5'te: modelin
konsantrasyonu 60 dakikada pik değerin ~%2'sine iniyor (tek-bolus PK
kinetiği), oysa published çalışmadaki etki İNFÜZYON SÜRESİNCE
SÜRDÜRÜLDÜ -- bu, modelin **idame infüzyon tedavisini modelleyemediği**
anlamına gelen, dürüstçe kabul edilmesi gereken bir kapsam sınırı.

**Test yazılırken keşfedilen ayrı bir nüans (Faz 3 sonrası GÜNCEL DURUM
değişti, bkz. §1a):** Eski (temsili, 0.03 mg/L) ec50 değerinde, pik
konsantrasyon (0.163 mg/L) EC50'ye çok yakındı -- Emax eğrisi neredeyse
DOYGUNDU (%84.5 pik etki), bu yüzden etki konsantrasyondan çok daha
yavaş sönüyordu (60 dk'da ~%11 vs ~%2). Faz 3'te ec50, literatürden
türetilen 0.737 mg/L'ye güncellendikten SONRA bu davranış DEĞİŞTİ: pik
konsantrasyon artık EC50'nin çok altında kalıyor, Emax eğrisi doygun
olmayan (yaklaşık doğrusal) bölgede çalışıyor -- etki artık konsantrasyonla
NEREDEYSE AYNI HIZDA sönüyor (60 dk'da ikisi de ~%2). Bu, kalibrasyon
değişince modelin davranış REJİMİNİN de değişebileceğinin somut bir
örneği -- bug değil, ama izlenmesi gereken bir etki.

---

## 4. Genel Sonuç ve Öneriler

1. **PK'nin "iskelet" kısmı (ka/ke/Vd, dozlar) genel olarak güvenilir** --
   çoğu gerçek literatürden, birden fazla bağımsız kaynaktan (FDA etiketi
   + RxNorm + ders kitabı) çapraz doğrulandı.
2. **PD'nin "büyüklük" kısmı (emax/ec50) neredeyse hiçbir ilaçta gerçek
   bir doz-yanıt çalışmasından kalibre edilmedi -- Faz 3'te SADECE
   esmolol'ün ec50'si için kısmi bir istisna yapıldı** (§1a; `emax_hr`/
   `emax_sbp` esmolol'de bile hâlâ temsili, digoksin'de hiçbir ilerleme
   olmadı, §1b). Bu projenin gerçek bir klinik/araştırma aracına
   dönüşmesi için yapılması gereken EN ÖNEMLİ iş hâlâ budur.
3. ~~Model, tek-doz/bolus senaryolarını temsil ediyor, sürekli infüzyon
   tedavisini DEĞİL.~~ -- **ÇÖZÜLDÜ** (bkz. §7). Dobutamin ve nitroprussid
   artık gerçek infüzyon+kesme PK modeliyle (Rowland & Tozer standart
   formülü) temsil ediliyor. Nikardipin KAPSAM DIŞI bırakıldı: klinikte
   sıkça infüzyonla verilse de, kod tabanında hâlen doğrulanmış bir bolus
   dozu (DailyMed) olarak işaretli, ayrı bir değerlendirme gerektirir.
   `sodium_nitroprusside` artık `drugs.yaml`'da seçilebilir durumda
   (önceden sadece provenance-only dosyadaydı).
4. Elektrolit/komorbidite çarpanlarının YÖNÜ gerçek fizyolojiye dayanıyor,
   ama KESİN büyüklükleri (eğimler, yüzdeler) hiçbir hasta kohortundan
   kalibre edilmedi -- bunlar "doğru yönde, kalibrasyon gerektiren"
   demonstratif parametreler.

Bu proje bir **kavram-kanıtı (proof-of-concept)** olarak tasarlandı; yukarıdaki
sınırlamalar bilinçli mühendislik kararlarıdır, gözden kaçan hatalar değil.

---

## 5. Faz 4 — Polifarmasi (Esmolol + Digoksin) Klinik Doğrulaması

Projenin "tehlikeli kombinasyon" demo senaryosunu (esmolol+digoksin --
`compare_polypharmacy.py`, `configs/drug_interactions.yaml`), yayınlanmış
klinik kaynaklarla karşılaştırdık. Sonuç **karışık** -- bazı kaynaklar
senaryoyu destekliyor, bir tanesi ise DOĞRUDAN ÇELİŞİYOR. İkisi de
dürüstçe aşağıda.

### 5a. Kaynak [F] -- Sağlıklı gönüllülerde ÇELİŞEN kanıt

Kessler ve ark. (esmolol-digoksin etkileşim çalışması, PubMed 2888792):
11 sağlıklı erkek gönüllüde, esmolol infüzyonu digoksin ile birlikte
verildiğinde:
- Digoksin AUC'si **~%10.8 arttı** (2.60 → 2.88 ng·saat/mL, p<0.05) --
  esmololün digoksinin böbrek/bağırsak P-glikoprotein atılımını
  inhibe etmesinden kaynaklanan bir **farmakokinetik** (PK) etkileşim.
- Ama: **"nabız ve tansiyonda klinik olarak anlamlı değişiklik YOKTU"**,
  PR aralıkları digoksin-tek-başına ile digoksin+esmolol arasında
  **benzerdi**.

**Bu, projenin "kombinasyon additive/sinerjistik olarak daha kötü"
varsayımıyla DOĞRUDAN ÇELİŞİYOR** -- en azından sağlıklı, normal
elektrolitli bir popülasyonda, bu spesifik dozlarda. Model bu PK
etkileşimini (esmololün digoksin AUC'sini artırması) HİÇ modellemiyor --
bilinen, düzeltilmemiş bir kapsam boşluğu.

### 5b. Kaynak [G] -- Vulnerable hastada DESTEKLEYEN kanıt

Bir vaka raporu (PMC11856498): 82 yaşında, 77 kg kadın hasta, digoksin
0.25 mg (haftada 5 gün) + metoprolol 100 mg/gün (esmolol değil ama aynı
mekanizma sınıfı -- beta-bloker) kullanırken **tam AV bloğuna** girdi
(ventriküler hız 35/dk). Hasta hafif hiperkalemikti (K=4.90 mmol/L) ve
böbrek fonksiyonu bozulmuştu (BUN 40 mg/dL). İlaçlar kesildikten 3 gün
sonra AV blok düzeldi.

**Bu, [F]'nin aksine senaryoyu DESTEKLİYOR** -- ama KRİTİK bir farkla:
[F]'deki sağlıklı gönüllülerin aksine, bu hasta **yaşlı + hafif
hiperkalemik + böbrek fonksiyonu bozuk**.

### 5c. Sonuç -- Faz 4'ün asıl bulgusu: risk POPÜLASYONA bağlı, tek başına ilaç çiftine değil

[F] ve [G] birlikte okunduğunda ortaya çıkan gerçek ders: bu kombinasyonun
tehlikesi **sabit bir "ilaç A + ilaç B = kötü" kuralı değil, hastanın
temel durumuna (yaş, böbrek fonksiyonu, elektrolitler) bağlı koşullu bir
risk**. Bu, doğrudan modelin mimarisini test etmemize yol açtı --

**Bulunan ve düzeltilen gerçek bir tutarsızlık:** `pd.py >
potassium_av_conduction_factor()`, önceden SADECE CircAdapt tarafında
kullanılıyordu (`apply_patient_electrolytes_to_circadapt`) -- istatistiksel
Monte Carlo motoru (`run_monte_carlo`/`run_polypharmacy_simulation*`)
hastanın potasyum/kalsiyum düzeyinden **tamamen bağımsız** çalışıyordu.
Doğrulama: `hasta_a` (sağlıklı) ve `hasta_c_hiperkalemi` (K=6.5,
renal_function=0.3) ile aynı esmolol+digoksin kombinasyonu koşulduğunda,
düzeltmeden ÖNCE ikisi de ~65 bpm veriyordu (istatistiksel olarak farksız)
-- [G]'nin gösterdiği "vulnerable hastada daha kötü" ilişkisiyle TUTARSIZ.
Düzeltmeden SONRA (`electrolyte_adjusted_emax_hr`/`_sbp`, bkz. `pd.py`):

| Hasta | K (mEq/L) | Kombinasyon ort. en düşük nabız | Bradikardi riski |
|---|---|---|---|
| `hasta_a` (sağlıklı) | 4.25 | 65.1 bpm | %0 |
| `hasta_c_hiperkalemi` | 6.5 | 60.5 bpm | %3 |

Artık [G]'nin niteliksel (yön) bulgusuyla tutarlı -- ama **büyüklük hâlâ
temsili** (0.3 eğim kalibre edilmedi) ve vaka raporundaki DRAMATİK sonuç
(ventriküler hız 35 bpm, tam blok) modelin ürettiği ölçülü farktan (60.5
vs 65.1 bpm) çok daha şiddetli -- model, gerçek AV BLOĞU riskini (dropped
beats, iletim kesintisi) hiç temsil etmiyor, sadece ortalama nabız
düşüşünü. Bu, Faz 5'in (CircAdapt'te gerçek AV iletim mekanizması)
motivasyonu.

### 5d. Dürüstçe kabul edilmesi gereken kapsam sınırları

1. ~~Esmololün digoksin AUC'sini artırma etkisi (Kaynak [F], ~%11) modele
   HİÇ eklenmedi~~ -- **KISMEN ÇÖZÜLDÜ** (bkz. §5e). Sadece esmolol->digoksin
   çifti (Kessler ve ark. 1987, `auc_ratio=1.11`) modellendi. Diğer TÜM ilaç
   çiftleri için PK-seviyeli etkileşim HÂLÂ modellenmiyor, sadece PD-seviyesinde
   (`interaction_matrix`, temsili sinerji çarpanı) temsil ediliyor. Yeni
   doğrulanmış bir vaka bulunursa `configs/drug_pk_interactions.yaml`'a TEK
   SATIR eklemek yeterli -- kod değişikliği gerekmez.
2. ~~Model, gerçek AV BLOĞU (iletim kesintisi, atlanan atımlar) fizyolojisini
   temsil etmiyor -- sadece ortalama nabız düşüşünü. Vaka raporundaki gibi
   dramatik, kesikli (all-or-nothing) bir olayı yumuşak/sürekli bir nabız
   eğrisiyle yakalayamaz.~~ -- **KISMEN ÇÖZÜLDÜ** (bkz. §8). 3. derece
   (tam) AV blok artık ayrık bir eşik-aşımı olayı olarak modelleniyor;
   Wenckebach tipi periyodiklik (2. derece) VE 1. derece AV blok bilinçli
   olarak dışarıda bırakıldı, gerekçesi §8'de.
3. `interaction_matrix`'teki 0.5 katsayısı (esmolol+digoksin) hâlâ tamamen
   temsili -- [F]/[G]'den TÜRETİLMEDİ, sadece yönü destekleniyor.

### 5e. PK-seviyeli ilaç-ilaç etkileşimi (esmolol -> digoksin, Kessler 1987)

`configs/drug_pk_interactions.yaml` -- PD-seviyesindeki `interaction_matrix`'ten
BİLİNÇLİ OLARAK AYRI bir dosya/veri yapısı (`src/worldmodel/pk.py >
pk_interaction_adjusted_ke()`, `simulation.py > build_pk_interaction_matrix()`).
Ayrı tutulma gerekçesi: (1) PD etkileşimi effect-çarpanına gider, PK etkileşimi
ke'ye -- aynı alanı ikisi için kullanmak ileride "factor her zaman PD'dir"
varsayımıyla sessiz bir hata riski taşırdı; (2) PK etkileşimi YÖNLÜ (esmolol
digoksini etkiler, tersi yok) -- PD'nin index-simetrik `(i,j)` yapısından
farklı, isim-tabanlı `(perpetrator, victim)` çifti kullanır.

**Mekanizma:** AUC (eğrinin altındaki alan -- toplam ilaç maruziyeti) ∝ 1/ke
(Vd sabitken). Kessler ve ark. 1987, esmolol varlığında digoksin AUC'sinin
~%10.8 arttığını (2.60 -> 2.88 ng*saat/mL, p<0.05, 11 sağlıklı gönüllü)
ölçtü -- `auc_ratio=1.11` olarak yuvarlanıp uygulandı:

```
ke_final = organ_function_adjusted_ke(...) / auc_ratio
```

yani hastanın böbrek/karaciğer kapasitesi ÖNCE hesaplanır, PK etkileşiminin
o kapasiteyi ne kadar "bloklandığı" SONRA uygulanır.

**Doğrulama (izole script, tahmin değil ölçüm):** `ke_organ_adjusted /
ke_final` tam olarak `1.11` (elle hesap ve `pk_interaction_adjusted_ke()`
çıktısı birebir eşleşiyor). Bu tek-kompartman kapalı-form modelde AUC_0-∞
analitik olarak `dose/(Vd*ke)` olduğundan, `ke` oranı 1.11 ise AUC_0-∞ oranı
da MATEMATİKSEL OLARAK tam 1.11 -- analitik hesapla doğrulandı (0.01925 ->
0.017342, AUC oranı = 1.11 tam). Sonlu bir zaman penceresinde (`np.trapezoid`
ile sayısal integrasyon) sonuç, pencere kısa olduğunda (örn. 96 saat, ~2.7
yarı-ömür) hedefin altında kalıyor (~1.068) çünkü kuyruk (tail) kesiliyor --
pencere genişledikçe (240s, 500s, 1000s) 1.11'e yakınsıyor (sırasıyla 1.10,
1.11, 1.11). Bu, bir hata değil, sonlu-pencere integrasyonunun beklenen bir
artefaktı -- analitik (sonsuz-pencere) sonuç dizayn hedefiyle tam eşleşiyor.

**Dürüstçe kabul edilmesi gereken kapsam sınırları:**
1. Sadece esmolol->digoksin çifti modellendi -- `configs/drug_pk_interactions.yaml`
   şu an TEK kayıt içeriyor, başka hiçbir çift için temsili bir `auc_ratio`
   EKLENMEDİ (bkz. §5d madde 1).
2. `auc_ratio`, Kessler 1987'nin ölçtüğü BÜTÜN (whole-body/net) AUC değişimini
   doğrudan `ke`'ye uygular -- digoksinin `renal_clearance_fraction=0.65`
   parametresiyle çarpıp "sadece renal atılım kısmına" bölmek gibi daha
   "mekanistik" görünen bir alt-bölümleme YAPILMADI, çünkü kaynak çalışma
   böyle bir ayrımı ölçmedi -- kaynakta olmayan bir hassasiyet uydurmak
   olurdu.
3. Çoklu-perpetrator senaryosu (aynı victim ilacı birden fazla ilaç aynı anda
   etkiliyorsa) kod İSKELETİ olarak desteklenir (`pk_interaction_adjusted_ke()`
   içinde sırayla/çarpımsal uygulama), ama şu an tabloda TEK kayıt olduğu
   için bu yol HİÇ egzersiz edilmiyor/doğrulanmıyor -- gerçek çoklu-ilaç PK
   etkileşimlerinde doygunluk/rekabetçi inhibisyon gibi doğrusal-olmayan
   etkiler olabilir, çarpımsal varsayım şimdilik test edilmemiş bir
   basitleştirme.
4. Bu, digoksinin kendi PD etkisini (emax_hr/emax_sbp, hâlâ ⚠️ temsili,
   bkz. §1b) DEĞİŞTİRMEZ -- sadece digoksinin plazma konsantrasyonunu
   (dolayısıyla PD hesabına giren `ce_hr`/`ce_sbp`'yi) etkiler. PD-seviyesindeki
   `interaction_matrix`'teki 0.5 katsayısı (§5d madde 3) hâlâ ayrı ve temsili.

---

## 6. Faz 5 — CircAdapt'te AV Düğümü Mekanizması

Faz 4'te bulunan tutarsızlığın (ilaç etkisi ve elektrolit etkisi
istatistiksel motorda tamamen ayrı kanallardan geçiyordu) CircAdapt
karşılığı da vardı: `apply_drug_effect_to_circadapt()` ilaç etkisini
SADECE `General.t_cycle`+`Patch.Sf_act` üzerinden uyguluyordu,
`apply_patient_electrolytes_to_circadapt()`'in kullandığı
`Timings.c_tau_av1`'e hiç dokunmuyordu -- yani CircAdapt'te de bir
AV-düğümü-yavaşlatıcı ilaç ile hiperkalemi, birbirinden habersiz iki ayrı
mekanizmaydı.

**Düzeltme:** `beta_blocker`/`positive_inotrope` sınıfındaki ilaçlar artık
`Timings.c_tau_av1`'i de hedefliyor (bkz.
`integrate_drug_with_circadapt.py > apply_drug_effect_to_circadapt`).

**İzole doğrulama (CLAUDE.md kuralı gereği, tahmin değil ölçüm):**
- `c_tau_av1` mutasyonu hem BİRİKMELİ (art arda çağrılar çarpılıyor) hem
  `model.run(stable=True)` sonrası KALICI (ArtVen.p0'ın aksine PFC benzeri
  bir "geri alma" mekanizması yok) -- doğrulandı.
- İlk kontrolde YANLIŞ metriğe bakıldı: `c_tau_av1`'i izole olarak
  büyütmek EDV/LV-pik-basıncı gibi DOLAYLI hemodinamik metriklerde
  ancak büyük çarpanlarda (5x) görünür fark yaratıyordu (2x'te fark yok).
  Ama **DOĞRUDAN** metriğe (`Timings.tau_av`, ms cinsinden -- gerçek
  klinikte EKG'de PR aralığı olarak görülen büyüklük) bakılınca sonuç
  ÇOK daha net: esmolol, sağlıklı hastada AV gecikmesini 135.8→153.0ms
  (+17.3ms) uzatıyor; AYNI doz, hiperkalemik hastada (`hasta_c_hiperkalemi`,
  zaten yükselmiş bir bazalden, 196.9ms) 196.9→221.9ms (+25.0ms) --
  **mutlak ilaç etkisi bile hiperkalemik hastada daha büyük**, tam
  vaka raporu [G]'nin işaret ettiği yönde. Bu artık `streamlit_app.py`'de
  CircAdapt sekmesinde "AV gecikmesi (PR aralığı benzeri)" metriği olarak
  kullanıcıya gösteriliyor.

**Düzeltilmiş sonuç:** Mekanizma GERÇEK, doğru bağlı, VE kendi doğal
biriminde (ms, PR-aralığı benzeri) açıkça görünür -- önceki "sub-visible"
değerlendirmesi yanlış metriğe (EDV/basınç) bakmaktan kaynaklanıyordu.
~~Hâlâ dürüstçe kalan kısıt: CircAdapt'in kendisi gerçek AV BLOĞUNU (atlanan
atımlar, ani/kesikli olay) modellemiyor -- `tau_av` sürekli bir eksen,
vaka raporundaki gibi "35 bpm'e ani düşüş" olayını üretemez, sadece
iletimin ne kadar YAVAŞLADIĞINI gösterir.~~ -- **KISMEN ÇÖZÜLDÜ** (bkz. §8):
`tau_av`'ın kendisi hâlâ sürekli bir eksen, ama artık bunun ÜZERİNE, `c_tau_av1`
çarpanı CircAdapt'i çökertecek bir eşiği aşarsa (ADIM 0 izole deneyiyle
doğrulanan 5x-7x çöküş sınırının altında, 3.0x güvenlik payıyla) CircAdapt'e
hiç dokunulmadan doğrudan bir "kaçış ritmi" (35 bpm) durumuna geçen ayrı bir
katman var. `tau_av`'ın hangi mutlak değerden itibaren (örn. >200ms, klinikte
1. derece AV blok sınırı) klinik olarak anlamlı sayılması gerektiği hâlâ
kalibre edilmedi -- bu, §8'de bilinçli olarak dışarıda bırakılan 1./2. derece
AV blok için hâlâ geçerli bir boşluk.

---

## 7. Sürekli İnfüzyon PK Modeli (dobutamin, nitroprussid)

Önceki modelde dobutamin ve nitroprussid (klinikte SADECE sürekli infüzyonla
verilen, gerçek bir bolus dozu olmayan iki ilaç) `plasma_concentration()`
(Bateman/bolus denklemi) ile "bolus-eşdeğeri" bir yaklaşıklıkla temsil
ediliyordu -- bu, konsantrasyon-zaman eğrisinin ŞEKLİNİ (zirve yapıp düşme)
yanlış veriyordu; gerçekte infüzyon sırasında konsantrasyon zirve yapmadan
kararlı-duruma (plato) doğru yükselir.

**Yeni fonksiyon:** `pk.py > plasma_concentration_infusion()` -- standart
klinik farmakokinetik ders kitabı formülü (Rowland & Tozer, *Clinical
Pharmacokinetics and Pharmacodynamics*; Winter, *Basic Clinical
Pharmacokinetics*), icat edilmiş bir denklem DEĞİL:

```
Cl = ke * Vd
t <= T_inf:  C(t) = (R/Cl) * (1 - e^(-ke*t))
t >  T_inf:  C(t) = C(T_inf) * e^(-ke*(t-T_inf))
```

`simulation.py > get_plasma_concentration()` merkezi bir yardımcı fonksiyon
olarak, `drug.infusion_rate_mcg_per_kg_min` doluysa bu formüle, boşsa
(varsayılan -- tüm diğer ilaçlar) mevcut bolus formülüne yönlendiriyor --
mevcut davranış DEĞİŞMEDİ (regresyon testleriyle doğrulandı).

**Hangi ilaçlar geçirildi, hangileri KAPSAM DIŞI:**
- **Dobutamin:** `infusion_rate_mcg_per_kg_min=5.0` -- kod tabanında ZATEN
  var olan "klasik 5 mcg/kg/dk infüzyon" değeri (önceden `dose_mg_per_kg`'yi
  geriye türetmek için kullanılıyordu, şimdi doğrudan kullanılıyor). Kaynağı
  hâlâ zayıf (isimsiz "pediatrik YBÜ çalışması") -- ⚠️ temsili.
- **Nitroprussid:** `infusion_rate_mcg_per_kg_min=5.15` -- **FDA onaylı doz
  aralığının (0.3-10 mcg/kg/dk) ortancası**, gerçek bir doz-yanıt
  çalışmasından (Kessler 1987 veya esmolol Faz 3 gibi) TÜRETİLMEDİ,
  düzenleyici aralıktan alınan tek-nokta temsili bir değer -- nicardipine/
  dobutamine EC50'leriyle AYNI güven kategorisinde: ⚠️ temsili. Ayrıca bu
  ilaç artık `configs/drugs.yaml`'a taşındı (önceden sadece provenance-only
  `drugs_verified.yaml`'daydı, ana simülasyon motorundan hiç seçilemiyordu --
  bu, sürekli infüzyon görevi sırasında ortaya çıkan ayrı bir bulguydu).
- **Nikardipin:** BİLİNÇLİ OLARAK KAPSAM DIŞI. Klinikte sıkça sürekli
  infüzyonla da verilir, ama kod tabanındaki mevcut değer gerçek bir bolus
  dozunu (30 mcg/kg, DailyMed) temsil ediyor -- bunu "aslında infüzyon"
  diye değiştirmek ayrı bir tartışmalı karar gerektirir, bu görevin
  kapsamına dahil edilmedi.

**Doğrulama:** Süreklilik testi (formülün `t=T_inf` anında ATLAMA
yapmadığı, iki dalın o noktada aynı değeri verdiği) ve kararlı-duruma
yakınsama testi (`infusion_duration_hr=None` iken `t` büyüdükçe `C(t) ->
R/Cl`) geçti. Mevcut bolus ilaçlar (esmolol dahil) için `get_plasma_
concentration()` çıktısının eski `plasma_concentration()` çağrısıyla
BİREBİR aynı olduğu doğrulandı (regresyon yok).

**Dürüstçe kalan kapsam sınırı:** `run_reference_trace()` ("Dünya Modelini
Gözlemle" sayfasının tek-iz fonksiyonu) bu değişikliğe DAHİL EDİLMEDİ --
sadece `run_monte_carlo`, `run_polypharmacy_simulation`, `run_polypharmacy_
simulation_loewe` güncellendi (görevin kapsamı buydu). Yani şu an
dobutamin/nitroprussid'i "Dünya Modelini Gözlemle" sayfasında tek-iz olarak
incelerseniz, hâlâ ESKİ bolus eğrisini görürsünüz -- istatistiksel
simülasyon (Monte Carlo) sonuçları doğru, ama gözlemleme sayfası henüz
tutarlı değil. Bu ayrı bir küçük takip görevi olarak kalıyor.

---

## 8. Discrete (all-or-nothing) AV Blok (Gap #3)

ADIM 0'da izole bir CircAdapt deneyi yapıldı: `Timings.c_tau_av1`
çarpanını kademeli büyütürken (1x, 3x, 5x, 7x, 9x, 15x, 50x), model 5x'te
stabil kalırken **7x'te ve üstünde SAYISAL OLARAK ÇÖKÜYOR** (`ModelCrashed`)
-- yani CircAdapt'in kendisinde "sürekli büyüyen bir gecikme"nin fizyolojik
olarak anlamlı bir üst platosu YOK, sadece "çalışıyor" ve "çöktü" iki
durumu var. Bu, gerçek klinikteki "tam/3. derece AV blok" (iletim TAMAMEN
kesilir, ventriküller kendi -- çok daha yavaş -- kaçış ritmiyle atar)
olgusunun matematiksel bir yansıması olarak yorumlandı: sürekli bir
gecikme fonksiyonu, belli bir noktadan sonra fizyolojik olarak anlamsız
hale geliyor, ayrık bir durum değişikliği gerekiyor.

**Mekanizma (`pd.py`):**
- `AV_BLOCK_THRESHOLD_MULTIPLIER = 3.0` -- izole deneyde gözlemlenen EN
  DÜŞÜK çöküş sınırının (5x, bu turdaki tekrar denemede 7x çıktı, farklı
  başlangıç durumu nedeniyle) ALTINDA bir güvenlik payı. Hastalar arası
  değişkenlik nedeniyle "çöküşün hemen altı" güvenli sayılmadı -- bu,
  `dose_min_mg`/`scale_min` gibi mevcut "keyfi ama belgelenmiş" tarama/eşik
  sabitleriyle AYNI kategoride, klinik bir kesinlik iddiası taşımaz.
- `AV_BLOCK_ESCAPE_RHYTHM_HR = 35.0` -- idioventriküler kaçış ritmi
  (standart kardiyoloji literatüründe 20-40 bpm aralığı, 35 bu aralığın
  temsili orta noktası). ⚠️ TEMSİLİ SABİT -- esmolol/nikardipin EC50'lerinde
  kullanılan ⚠️ temsili kategorisiyle AYNI güven seviyesinde, hasta/vaka-özel
  bir ölçümden gelmiyor.
- `av_conduction_cumulative_multiplier()` -- CircAdapt'in `c_tau_av1`
  üzerinde biriktirdiği kümülatif çarpanın (elektrolit `k_factor` × her
  AV-duyarlı ilacın `1/hr_fraction` terimi) istatistiksel motordaki
  KARŞILIĞI. **GÜNCELLEME (N-ilaç genellemesi, ADIM 3.5 / ADR-3, bkz.
  N_DRUG_AUDIT.md Şüphe E ve RESEARCH_N_DRUG.md):** bu fonksiyon ESKİDEN
  (bu güncellemeden önce) SADECE bir YAKLAŞIKLIKTI -- additive motorun
  ZATEN toplamış olduğu TEK bir birleşik `hr_fraction`'ı k_factor'a
  bölüyordu, CircAdapt'in ilaç-başına SIRAYLA çarpımsal birikimini
  (`run_with_multiple_drugs`) TEKRARLAMIYORDU. İzole ölçüm bu sapmanın
  N ile hızla büyüdüğünü gösterdi: N=2'de %1.7, **N=5'te %52**. Artık
  fonksiyon HER AV-duyarlı ilacın İZOLE hr_fraction'ını AYRI AYRI, SIRAYLA
  çarpımsal olarak biriktiriyor -- `integrate_drug_with_circadapt.py >
  cumulative_av_conduction_multiplier()` İLE BİREBİR AYNI matematik. İki
  motor artık AYNI formülü paylaşıyor -- sapma ölçülüp belgelenmekle
  kalmadı, GİDERİLDİ.
- `discrete_av_block_mask()` -- eşik aşıldığı İLK noktadan İTİBAREN
  (mandal/latch, titreşen bir maske DEĞİL) True döner. Gerekçe: AV bloğu
  klinikte ilaç konsantrasyonu düştükçe anlık "açılıp kapanmaz" -- kademeli
  düzelmeyi (Wenckebach tipi periyodiklik) ayrıca modellemek bu basit/lump
  soyutlamanın ötesinde (aşağıya bakın).

**Entegrasyon noktaları:**
- İstatistiksel motor (`simulation.py > run_monte_carlo`,
  `run_polypharmacy_simulation`, `run_polypharmacy_simulation_loewe`):
  her Monte Carlo denemesinin zaten örneklediği ke/sensitivity'den üretilen
  hr zaman dizisi, eşik aşılan noktadan itibaren `AV_BLOCK_ESCAPE_RHYTHM_HR`
  ile maskeleniyor. YENİ bir rastgele değişken EKLENMEDİ -- eşik aşımı
  mevcut örneklemeden DETERMİNİSTİK olarak türüyor.
- CircAdapt entegrasyonu (`integrate_drug_with_circadapt.py >
  cumulative_av_conduction_multiplier()`): `apply_patient_electrolytes_
  to_circadapt`/`apply_drug_effect_to_circadapt` ÇAĞRILMADAN ÖNCE, modele
  hiç dokunmadan aynı çarpanı hesaplar. Eşik aşılırsa (ya da hasta
  `known_av_block_degree=="third"` ise) `run_comparison`/
  `run_polypharmacy_comparison`, CircAdapt'i çökertecek fonksiyonları
  (`run_with_drug`/`run_with_multiple_drugs`) **HİÇ ÇAĞIRMAZ** -- mock/spy
  testiyle doğrulandı (`ModelCrashed` hiç tetiklenmiyor). "İlaçlı" sonuç,
  baseline'ın p/v eğrilerini güvenli bir yer tutucu olarak yeniden kullanıp
  `hr_drug_model=35.0`, `tau_av_drug_ms=None`, `av_block_triggered=True`
  döndürerek temsil edilir.
- `Patient.known_av_block_degree` (`patient.py`, yeni alan, varsayılan
  `None`) -- `patient_profile/` modülünün extraction şemasındaki AYNI
  isimli alanla (`patient_profile/schema.py`) UYUMLU, bu iki modülü
  birbirine bağlayan İLK gerçek köprü. **SADECE `"third"` değeri** simülasyon
  mantığına bağlı -- t=0'dan itibaren, eşik hesabına hiç girmeden doğrudan
  kaçış ritmi. `"first"`/`"second"` HİÇBİR YERE bağlanmadı -- büyüklük
  tahmini gerektiren bir sayı uydurmak yerine, bu bilinçli olarak boş
  bırakıldı.

**Dürüstçe kabul edilmesi gereken kapsam sınırları:**
1. **1. ve 2. derece AV blok modellenmiyor.** Sadece 3. derece (tam blok,
   ikili: var/yok) var. 1./2. derece için büyüklük (örn. PR aralığının ne
   kadar uzadığı, Wenckebach'ın kaçıncı vuruşta düştüğü) tahmin etmek
   ayrı bir doz-yanıt/literatür çalışması gerektirir -- bu proje bunu
   uydurmadı.
2. **Wenckebach tipi periyodiklik kasıtlı olarak modellenmedi.** Gerçek 2.
   derece Mobitz I AV blok, periyodik/olasılıksal bir "bazı vuruşlar düşer,
   bazıları düşmez" davranışı gösterir. Bu modelin mevcut soyutlama
   seviyesi (organ/lump, deterministik eşik) bu davranışı doğal olarak
   temsil edemez -- eklemek, ayrı bir olasılıksal/periyodik alt-model
   gerektirirdi, mimari uyumsuzluk nedeniyle kapsam dışı bırakıldı.
3. **CircAdapt tarafında `known_av_block_degree=="third"` sadece İLAÇ
   sonucuna uygulandı, `run_baseline()`'a değil.** Yani CircAdapt'te bu
   hastanın "ilaçsız" baseline'ı hâlâ normal iletim gösteriyor -- gerçekte
   kalıcı 3. derece AV bloğu olan bir hastanın baseline'ı da kaçış
   ritminde olmalıydı. Bu, `hr_base`'in `t_cycle`'dan (c_tau_av1'den
   BAĞIMSIZ) türetilmesinden kaynaklanan bir mimari sınırlama -- düzeltmek
   `run_baseline()`'ın kendisini yeniden yapılandırmayı gerektirir, bu
   turun kapsamı dışında bırakıldı.
4. Hastanın kendi elektrolitleri TEK BAŞINA (hiç ilaçsız) eşiği aşacak
   kadar aşırı olursa (`k_factor>=3.0`, yani K≈15 mEq/L -- fizyolojik
   olarak yaşamla bağdaşmayan bir düzey) `run_baseline()` hâlâ CircAdapt'i
   çağırıyor, bir pre-check yok -- ama bu düzeyde bir hiperkalemi zaten
   klinik olarak anlamsız/imkânsız olduğu için pratikte bir risk
   oluşturmuyor.

---

## 9. Bilinen UI Kısıtları

Bu bölüm, backend'de doğru çalışan ama mevcut Streamlit slider/seçenek
aralıklarıyla kullanıcı tarafından tetiklenemeyen (ya da erişilemeyen)
mekanizmaları listeler -- bunlar bilinçli olarak DÜZELTİLMEDİ, sadece
dürüstçe belgeleniyor.

### 9a. CircAdapt "AV blok tetiklendi" mesajı mevcut slider aralıklarıyla ulaşılamıyor (Bulgu 2)

`integrate_drug_with_circadapt.py > cumulative_av_conduction_multiplier()`
ve `discrete_av_block_mask()` (bkz. §8) doğru çalışıyor ve birim
testleriyle kanıtlanmış durumda -- ama CircAdapt sekmesindeki "AV blok
tetiklendi" mesajını tetiklemek için gereken kümülatif çarpan (~3.0,
`AV_BLOCK_THRESHOLD_MULTIPLIER`) mevcut Streamlit slider üst sınırlarıyla
(elektrolit K≤8.0 mEq/L, ilaç dozu≤referans dozun 3 katı) HİÇBİR
kombinasyonda aşılamıyor -- gereken çarpana ulaşmak için K≈10+ mEq/L
gibi, mevcut slider aralığının (2.0-8.0) dışında bir değer gerekiyor.

**Sonuç:** İstatistiksel motorun (Monte Carlo) kaçış-ritmi tetiklenmesi
mevcut Streamlit slider aralıklarıyla (K≤8.0) canlı doğrulandı ve
tetiklenebiliyor -- bu ayrım SADECE CircAdapt sekmesindeki mesaj için
geçerli. Bu, ayrı bir küçük takip görevi (slider üst sınırlarını
genişletmek ya da eşiği yeniden kalibre etmek) olarak kalıyor, bu
turun kapsamına dahil edilmedi.

---

## 10. N-İlaç Genellemesi — CircAdapt Parametre-Başına Çöküş Eşikleri (ADIM 4.1)

`N_DRUG_AUDIT.md` Şüphe F: §8'deki 5x-7x çöküş eşiği SADECE
`Timings.c_tau_av1` için ölçülmüştü. N ilaçta `Patch.Sf_act`
(kontraktilite), `ArtVen.p0[0]` (sistemik direnç) ve `General.t_cycle`
(kalp siklus süresi -- nabız) de çarpımsal biriktiği için
(`apply_drug_effect_to_circadapt`), bu üçünün de izole çöküş eşiği
ölçüldü: `scripts/circadapt_parameter_crash_thresholds.py` (yukarı yönlü,
parametreyi büyüten çarpanlar: 1x-500x) ve
`scripts/circadapt_parameter_crash_thresholds_downward.py` (aşağı yönlü,
parametreyi sıfıra yaklaştıran çarpanlar: 1x-0.01x), hasta_a baseline
modelinden tek-parametre izolasyonuyla.

**Ölçülen sonuçlar** (`logs/circadapt_parameter_crash_thresholds*.json`):

| Parametre | Yukarı yönlü ilk çöküş | Aşağı yönlü ilk çöküş | Not |
|---|---|---|---|
| `Timings.c_tau_av1` | 7.0x (5.0x stabil) | test edilmedi (fizyolojik olarak ilgisiz -- hiçbir ilaç/elektrolit bu parametreyi azaltmıyor) | §8'deki bilinen sonucu doğruladı (kontrol ölçümü) |
| `General.t_cycle` | **3.0x** (2.8x stabil) | **0.15x** (`ModelNotStable`, 0.25x stabil) | **EN KIRILGAN parametre** -- ve HER ilaç (sınıfından bağımsız) bunu etkiliyor |
| `Patch.Sf_act` | 100.0x (50.0x stabil) | 0.10x (0.15x stabil) | c_tau_av1'den çok daha dayanıklı |
| `ArtVen.p0[0]` | test edilen aralıkta (500x'e kadar) hiç çökmedi | test edilen aralıkta (0.01x'e kadar) hiç çökmedi | en dayanıklı -- ama "sonsuz güvenli" değil, sadece test edilen aralıkta çökmedi |

**En önemli bulgu:** `General.t_cycle`, `c_tau_av1`'den (7.0x) çok daha
kırılgan (3.0x) -- ve bugüne kadar SADECE `c_tau_av1` için bir ön-kontrol
vardı, `t_cycle` için HİÇ yoktu. Bu, güçlü negatif kronotropların N=2-3
kombinasyonunun (her ilaç `t_cycle`'ı `1/hr_fraction` ile çarpımsal
büyüttüğü için) CircAdapt'i ön-kontrolsüz ÇÖKERTEBİLECEĞİ anlamına
geliyordu -- pratikte bu projenin mevcut ilaç dozlarında (bkz.
tests/test_no_regression_n_drug.py, N=1/2 golden senaryoları) eşiğe
yaklaşılmıyor, ama N≥3 güçlü doz kombinasyonlarında risk gerçek.

**Düzeltme (ADIM 4.2):**
`integrate_drug_with_circadapt.cumulative_parameter_multipliers()` --
`cumulative_av_conduction_multiplier()`'ın (SADECE c_tau_av1)
genelleştirilmiş hâli -- artık `t_cycle`/`Sf_act`/`ArtVen.p0`/`c_tau_av1`
için TÜMÜ için CircAdapt'e HİÇ dokunmadan kümülatif çarpanı hesaplıyor.
`circadapt_instability_risk()` bunlardan HERHANGİ biri ölçülmüş güvenli
aralığın (yukarıdaki tablonun bir güvenlik payı altındaki karşılığı --
`T_CYCLE_MULTIPLIER_SAFE_RANGE=(0.20, 2.5)`,
`SF_ACT_MULTIPLIER_SAFE_RANGE=(0.12, 40.0)`,
`ARTVEN_P0_MULTIPLIER_SAFE_RANGE=(0.01, 500.0)`) dışına çıkarsa,
`run_polypharmacy_comparison()` CircAdapt'i HİÇ çağırmadan, temiz bir
`instability_triggered=True` sonucu döndürür (AV blok deseninin
genelleştirilmiş hali). c_tau_av1 (AV blok) klinik olarak en anlamlı
yorumu taşıdığı için öncelikli kontrol ediliyor; diğer üç parametrenin
çöküşü için klinik bir "kaçış ritmi" karşılığı yok -- bu durumda
`hr_drug_model`, PK/PD zincirinin (CircAdapt'siz) additive tahminine
düşer, `unstable_parameter` alanı HANGİ parametrenin sorumlu olduğunu
açıkça taşır.

**Güvenlik payı metodolojisi:** §8'deki `AV_BLOCK_THRESHOLD_MULTIPLIER=3.0`
(ölçülen 5x-7x çöküşün altında) ile AYNI mantık -- ölçülen ilk çöküşün
BİR MİKTAR içinde bir eşik, "çöküşün hemen altı" güvenli sayılmadı (hasta
varyasyonu payı). Bu eşikler klinik bir kesinlik iddiası taşımaz, sadece
CircAdapt'in SAYISAL kararlılığını korur.

**Test kapsamı:** `tests/test_n_drug_circadapt.py` -- permütasyon
değişmezliği (Şüphe G, gerçek CircAdapt ile N=4, 8 permütasyon),
N=1-6 için "hiç çökme yok" (`@pytest.mark.slow`, yine de CI'da çalışır),
`cumulative_parameter_multipliers`/`circadapt_instability_risk`'in saf
mantık testleri, ve `t_cycle` eşiğinin CircAdapt'i HİÇ ÇAĞIRMADAN devreye
girdiğini doğrulayan bir spy testi.

---

## 11. N-İlaç İstatistiksel Motor -- Seçilen Yöntem ve Gerekçesi (Özet)

Bu bölüm §8-10'un CircAdapt-tarafı bulgularını, istatistiksel motor
tarafındaki mimari kararla (ADR) tamamlar. Tam literatür taraması ve
karşılaştırma tablosu `RESEARCH_N_DRUG.md`'de; burada sadece SONUÇ ve
KAYNAK özetleniyor.

**Seçilen yöntem:** Loewe additivity (`pd.py > loewe_combined_effect`,
Tallarida'nın N-ilaca genellenmiş izobol denklemi) -- literatürde N ilaca
en iyi belgelenmiş, düşük hesap maliyetli yöntem. Alternatifler (MuSyC,
ZIP, Chou-Talalay CI, 3. parti `synergy`/`SynergyFinder` paketleri)
kalibrasyon verisi eksikliği ve/veya N>2 için olgunlaşmamış formülasyon
nedeniyle REDDEDİLDİ (bkz. RESEARCH_N_DRUG.md §1, ADR-1).

**`min(Emax)` tavanı KALDIRILMADI:** Grabovsky & Tallarida (2004, *J
Pharmacol Exp Ther* 310(3):981-986) tam-agonist/kısmi-agonist için eğri
izobol yöntemi bile bu tavanı kaldırmıyor -- literatürde bunu prensipli
şekilde kaldıran bir yöntem yok. N büyüdükçe (en düşük-Emax'lı TEK ilaç
kombinasyonun tavanını belirlediği için) bu kısıt DAHA SIK bağlayıcı hale
gelir -- streamlit_app.py'de hangi ilacın tavanı belirlediği artık açıkça
gösteriliyor (ADR-5).

**Zıt yönlü ilaçlar için gruplama+fark** (`grouped_loewe_combined_effect`):
literatürde (Loewe, Bliss, HSA, MuSyC, ZIP -- hepsi tarandı) zıt yönlü
etkileri ele alan STANDART bir yöntem YOK. Bu, projenin kendi, kaynaksız
mühendislik kararı -- kod ve arayüzde `⚠️` işaretiyle böyle belirtiliyor.

**PD interaction teriminin simetrikleştirilmesi:** `run_polypharmacy_
simulation()`'daki additive-interaction terimi eskiden `emax[a]`'yı
kullanıp `emax[b]`'yi yok sayıyordu (YAML kaydının drug_a/drug_b sırasına
bağlı, belgesiz bir asimetri). N≥3'te artık `(emax[a]+emax[b])/2`
kullanılıyor (`symmetric_interaction_terms=True`, streamlit_app.py'de
`len(drugs)>=3` için otomatik) -- N=1/2 golden-snapshot davranışı
`False` varsayılanıyla korundu.

**Kaynakça:** tam liste ve URL'ler `RESEARCH_N_DRUG.md` §5'te; öne çıkanlar:
Foucquier ve ark. 2015 (Pharmacol Res Perspect); Grabovsky & Tallarida
2004 (J Pharmacol Exp Ther); Wooten ve ark. 2021 (Nat Commun, MuSyC);
Yadav ve ark. 2015 (Comput Struct Biotechnol J, ZIP); FDA/ICH M12 (2024,
Drug Interaction Studies).
