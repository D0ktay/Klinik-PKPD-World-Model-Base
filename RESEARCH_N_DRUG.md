# N-İlaç Kombinasyon Yöntemi — Literatür Araştırması ve Mimari Karar (ADR)

Bu belge ADIM 1'in çıktısıdır: N ilaç (N=1..8) kombinasyon etkisini
birleştirmek için hangi matematiksel yöntemi kullanacağımıza, yayınlanmış
kaynaklara dayanarak karar veriyoruz. `N_DRUG_AUDIT.md`'de doğrulanan iki
somut bulgu (Şüphe E: iki motor arası AV-blok sapması N=5'te %52; Şüphe A:
PK-DDI'nin N≥3'te sessizce yok sayılması) bu kararın merkezinde.

---

## 1. Araştırılan Yöntemler

### 1.1 Loewe Additivity / Genel İzobol Denklemi (GIE)

Klasik doz-eşdeğerliği (isobole) yöntemi. İki ilaç için `a/A_E + b/B_E = 1`
(a,b: kombinasyondaki dozlar; A_E,B_E: aynı etkiyi tek başına veren dozlar).

**N ilaca genelleme:** Doğrudan ve iyi belgelenmiş. Foucquier ve ark. (2015)
N ilaç için toplamı genelleştiriyor: `a/A_E + b/B_E + c/C_E + ... = 1` — 3
ilaçta bu, üç eksende A,B,C noktalarından geçen bir düzlem denklemine
karşılık geliyor (2 ilaçta bir doğru yerine). [Foucquier ve ark. 2015,
*Pharmacology Research & Perspectives*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492765/).
Bizim projedeki `loewe_combined_effect()` (`pd.py:78-152`) zaten bisection
ile bu N-ilaca-genellenmiş denklemi çözüyor — bu KISIM zaten literatürle
uyumlu.

**Farklı Emax sorunu (Grabovsky & Tallarida 2004):** İki ilacın Emax'ı
farklıysa (tam agonist + kısmi agonist), additive izobol artık düz bir
çizgi değil, **eğri** bir izobol oluyor — potens oranı sabit değil.
[Grabovsky & Tallarida 2004, *J Pharmacol Exp Ther* 310(3):981-986](https://jpet.aspetjournals.org/article/S0022-3565(24)31524-1/abstract).
**KRİTİK BULGU:** Bu eğri-izobol yöntemi `min(Emax)` tavanını
KALDIRMIYOR — sadece "additive" tanımının etki-seviyesine göre nasıl
değiştiğini yeniden şekillendiriyor; ulaşılabilir maksimum birleşik etki
yine düşük-Emax'lı ilacın tavanıyla sınırlı kalıyor
([Frontiers 2019 derlemesi, Isobologram Analysis](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2019.01222/full)).
Yani **Şüphe C'deki `min(emax_i)` tavanı, GIE ile de ortadan kalkmıyor** —
bu, Loewe ailesinin (klasik ya da eğri-izobol) yapısal bir özelliği.

**Zıt yönlü etkiler:** Hem klasik Loewe hem GIE, iki ilacın AYNI YÖNDE
etki ettiğini varsayıyor. Zıt yönlü etkiler (bir ilaç azaltır, biri
artırır) bu çerçevelerin **teorik kapsamı dışında** — kaynaklarda ele
alınmıyor (Frontiers 2019 derlemesi, aynı kaynak, madde 3).

### 1.2 Bliss Independence (çarpımsal)

İki ilacın birbirinden bağımsız, olasılıksal olarak etkilediği varsayımı.
N ilaca genelleme: `S_Bliss = E_combo - [1 - ∏(1 - E_i)]` (fraksiyon
cinsinden). [Foucquier ve ark. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492765/);
formül notasyonu ayrıca [AACR 2023 Cancer Research Communications](https://aacrjournals.org/cancerrescommun/article/3/10/2146/729692/Statistical-Assessment-of-Drug-Synergy-from-In)'da.

**Önemli gözlem:** "Kalan fraksiyon" (`1 - E_i`, ya da bizim terminolojide
`hr_fraction`, `sbp_fraction`) çarpımsal olarak birikmesi, N ilaca doğrudan
genellenen, komütatif (sıra bağımsız) bir işlem. PBPK-PD literatüründe de
"aynı hedefi paylaşan ilaçların fraksiyonel etkisinin çarpımsal
birleşmesi" Bliss-tipi bir yaklaşım olarak tanımlanıyor (bkz. §1.6).
**Bu, `integrate_drug_with_circadapt.py`'nin BUGÜN zaten yaptığı şeyle
BİREBİR AYNI matematiksel yapı** — `apply_drug_effect_to_circadapt()`
her ilaç için `t_cycle`'ı kendi `hr_fraction`'ına bölüyor
(`integrate_drug_with_circadapt.py:194-203`), bu tam olarak Bliss'in
"kalan fraksiyonların çarpımı" formülasyonu.

### 1.3 Highest Single Agent (HSA)

Kombinasyon etkisi = tek ilaçların en büyüğü. N ilaca genelleme trivial
(`max(E_1,...,E_N)`) ama bilgi kaybı yüksek — sinerji/antagonizma ayrımı
yapamıyor, sadece "kombinasyon en kötü tekliden daha mı kötü" sorusuna
cevap veriyor. Klinik doz önerisi için yetersiz (Foucquier ve ark. 2015).

### 1.4 MuSyC (Meyer/Wooten ve ark., Nature Communications 2021)

Potens sinerjisi (α) ile etkinlik/Emax sinerjisini (β) ayrı parametrelerle
modelleyen, durum-geçiş (state-transition) tabanlı parametrik model.
[Wooten ve ark. 2021, *Nature Communications*](https://www.nature.com/articles/s41467-021-24789-z);
[synergy paketi dokümantasyonu](https://synergy.readthedocs.io/en/latest/models/synergy/musyc.html).

**N ilaca genelleme:** MuSyC'nin YAYINLANMIŞ, olgunlaşmış formülasyonu
**temelde 2-ilaçlıdır** — potens/etkinlik/kooperativite parametreleri iki
ilaç arasındaki etkileşim için tanımlı. `synergy` Python paketi de MuSyC'yi
iki-ilaçlı bir model olarak implemente ediyor (bkz. §1.6, PyPI/Bioinformatics
makalesi). N>2 için resmi, konsensüs bir genelleme literatürde bulunamadı.

**Uygunluk değerlendirmesi:** MuSyC, `min(Emax)` tavanını AŞABİLECEK bir
çerçeve sunuyor (etkinlik sinerjisi ayrı parametrelendirildiği için), AMA
bunun için HER ilaç çiftinin (α, β, γ) parametrelerinin **deneysel
kalibrasyonunu** gerektiriyor. Bizim projede bu kalibrasyon verisi YOK ve
"KIRMIZI ÇİZGİ" kuralı (`configs/drug_pk_interactions.yaml`) uydurma
katsayı eklemeyi yasaklıyor — MuSyC parametrelerini tahmin ederek
uygulamak bu kuralı doğrudan ihlal eder. **Bu yüzden MuSyC bu proje için
şu an UYGULANABİLİR DEĞİL** (veri eksikliği, N>2 için olgun formülasyon
eksikliği).

### 1.5 ZIP (Zero Interaction Potency, Yadav ve ark. 2015)

Bliss'in çarpımsal (multiplicative survival) prensibini kullanan, doz-etki
eğrisi POTENSİNDEKİ kaymayı ölçen parametrik bir model.
[Yadav ve ark. 2015, *Computational and Structural Biotechnology Journal*](https://www.sciencedirect.com/science/article/pii/S2001037015000422).
Temelde Bliss ailesinden — N ilaca genellemesi Bliss'inkiyle aynı
çarpımsal mantığı izliyor, ama yine ikili doz-yanıt matrisleri (heatmap)
üzerinde ekran-tarama (screening) verisi için tasarlanmış; bizim tek-nokta
(pik konsantrasyon) PK/PD zincirimize doğrudan uygulanabilir bir avantaj
sağlamıyor.

### 1.6 Chou-Talalay Kombinasyon İndeksi (CI)

Medyan-etki denklemine (`D = Dm[fa/(1-fa)]^(1/m)`) dayanan, "mutually
exclusive/non-exclusive" ayrımı yapan yöntem.
[ResearchGate özet, Chou-Talalay yöntemi](https://www.researchgate.net/publication/41000828_Drug_Combination_Studies_and_Their_Synergy_Quantification_Using_the_Chou-Talalay_Method).
N>2 için "polygonogram/Chou-Chou grafiği" ile otomatik bilgisayar
simülasyonu öneriliyor ama bu esasen Loewe'nin (aynı `Dx` mantığı)
farklı bir sunumu — CI=1 additive, CI<1 sinerji, CI>1 antagonizma.
Bizim projeye Loewe'ye ek bir şey katmıyor, sadece yorumlama katmanı
farklı.

### 1.7 SynergyFinder / `synergy` (Python paketi)

`synergy` paketi (PyPI, [Bioinformatics 2021 makalesi](https://academic.oup.com/bioinformatics/article/37/10/1473/5909985))
Loewe, Bliss, HSA, ZIP, MuSyC, Chou-Talalay CI'ı tek bir arayüzde
implemente ediyor — ama **hepsi 2-ilaçlı (pairwise) doz-yanıt matrisi
varsayımıyla** tasarlanmış (deneysel ekran-tarama verisi için). SynergyFinder
3.0 "pairwise ve higher-order kombinasyon verisi" desteklediğini iddia
ediyor ([Ianevski ve ark. 2022, *Nucleic Acids Research*](https://academic.oup.com/nar/article/50/W1/W739/6586861))
ama makalede N>2 için matematiksel detay verilmiyor — kapalı kutu.
Projeye bağımlılık olarak eklemek, doğrulayamadığımız bir N>2 formülasyonuna
güvenmek anlamına gelir; bu projenin "bilmiyorsan ölç" kuralına aykırı.
**Bu paketleri bağımlılık olarak eklemiyoruz** — kendi Loewe/GIE ve Bliss
implementasyonlarımızı (zaten kısmen mevcut) N-ilaca doğru şekilde
genişletiyoruz.

### 1.8 QSP Kardiyovasküler Modellerde Çoklu-İlaç Pratiği

Kardiyovasküler QSP modelleri (Fu ve ark. 2022, [*CPT: Pharmacometrics &
Systems Pharmacology*](https://ascpt.onlinelibrary.wiley.com/doi/10.1002/psp4.12774);
genel QSP iş akışı, [Musante ve ark. 2019, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6617832/))
tipik olarak sinerji-skorlama çerçevelerini (Loewe/Bliss/MuSyC) DEĞİL,
doğrudan **mekanistik parametre pertürbasyonu** kullanır: her ilaç, kendi
hedef fizyolojik parametresini (kontraktilite, periferik direnç, kalp
hızı vb.) kendi mekanizmasına göre değiştirir, ve AYNI parametreyi
hedefleyen birden fazla ilacın etkisi genellikle **çarpımsal (fraksiyonel
kalan) birikim** ile birleştirilir — bu, yukarıdaki Bliss formülasyonuyla
aynı matematiksel yapı (§1.2). Bu, **CircAdapt motorumuzun bugün zaten
yaptığı şeyin** literatürdeki QSP pratiğiyle uyumlu olduğunu doğruluyor.

### 1.9 Regülasyon: FDA / ICH M12

ICH M12 "Drug Interaction Studies" (2024 nihai rehber,
[FDA M12 sayfası](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m12-drug-interaction-studies);
[nihai rehber PDF](https://database.ich.org/sites/default/files/ICH_M12_Step4_Guideline_2024_0521_0.pdf))
"kokteyl yaklaşımı" (cocktail approach) bölümünde, BİR test ilacının
BİRDEN FAZLA substrat/obje ilaç üzerindeki etkisinin AYNI ANDA test
edilmesini tarif ediyor — ama bu, N ilacın birleşik farmakodinamik
etkisini birleştirecek bir MATEMATİKSEL FORMÜL önermiyor; sadece klinik
çalışma TASARIMI (kaç ilaç aynı anda verilsin, hangi PK parametreleri
ölçülsün) için rehberlik. Yani ICH M12, bizim ADR'imizin matematiksel
kararına doğrudan girdi sağlamıyor — sadece PK-DDI'nin (Şüphe A) klinik
pratikte de N ilaçlı senaryolarda ayrıca ölçülmesi gereken, ihmal
EDİLEMEYECEK bir konu olduğunu teyit ediyor.

---

## 2. Karşılaştırma Tablosu

| Yöntem | N'e genelleme | Farklı Emax'ı kaldırır mı | Zıt yönü kaldırır mı | Hesap maliyeti | Bizim projeye uygunluk |
|---|---|---|---|---|---|
| Loewe / GIE | Net, iyi belgelenmiş (bisection zaten var) | HAYIR (eğri izobol bile tavanı korur) | HAYIR | Düşük (bisection, N-ilaç toplamı) | Zaten kısmen uygulanmış, N'e açık |
| Bliss (çarpımsal fraksiyon) | Net, trivial (`∏(1-E_i)`) | Yapısal olarak sorun değil (tavan kavramı yok) | HAYIR | Çok düşük (çarpım) | CircAdapt motoru ZATEN bunu yapıyor |
| HSA | Trivial ama bilgi kaybı yüksek | N/A | HAYIR | Çok düşük | Doz önerisi için yetersiz, kullanılmıyor |
| MuSyC | Literatürde N>2 için olgun değil | EVET (ama kalibrasyon verisi gerektirir) | Kısmen (durum-geçiş modelinde teorik olarak mümkün ama bizim verimizle kalibre edilemez) | Yüksek (parametrik fit) | UYGULANAMAZ (veri yok, KIRMIZI ÇİZGİ ihlali olur) |
| ZIP | Bliss ailesinden, N'e Bliss gibi genellenir | Kısmi | HAYIR | Orta (doz-yanıt matrisi fit) | Ekran-tarama verisi için tasarlı, bize uygun değil |
| Chou-Talalay CI | Loewe'nin başka sunumu | HAYIR | HAYIR | Loewe ile aynı | Ek değer yok |
| 3. parti paket (synergy/SynergyFinder) | Belirsiz/doğrulanamaz N>2 formülasyonu | Pakete bağlı | Hayır | Bağımlılık riski | Kullanılmıyor — "ölçmeden güvenme" kuralına aykırı |

---

## 3. Karar (ADR)

### ADR-1: İstatistiksel motor (HR/SBP büyüklüğü) → Loewe/GIE, N-ilaca genelleştirilmiş biçimde, KALIR

**Karar:** `pd.loewe_combined_effect()`'in bugünkü Loewe/bisection
yaklaşımı (zaten N-ilaca matematiksel olarak açık) **birincil yöntem
olarak KALIYOR**. Yeni bir sinerji modeli (MuSyC vb.) İTHAL EDİLMİYOR.

**Gerekçe:** (a) Loewe/GIE literatürde N ilaca en iyi belgelenmiş,
en düşük hesap maliyetli, projenin zaten kısmen doğru uyguladığı yöntem.
(b) MuSyC gibi alternatifler kalibrasyon verisi gerektiriyor ve bizde bu
veri yok — uydurma katsayı yasağını ihlal etmeden uygulanamaz.
(c) `min(Emax)` tavanı KALDIRILMIYOR (hiçbir literatür yöntemi bunu
prensipli şekilde kaldırmıyor) — bunun yerine N büyüdükçe daha sık
bağlayıcı hale geldiği arayüzde AÇIKÇA gösterilecek (ADIM 6), docstring'de
zaten var olan uyarı korunacak/güçlendirilecek.

### ADR-2: PK-DDI eksikliği (Şüphe A) → yöntem seçimi DEĞİL, saf düzeltme

**Karar:** `run_polypharmacy_simulation_loewe()`'ye `drug_keys` ve
`pk_interaction_matrix` parametreleri eklenip, additive yoldaki
(`run_polypharmacy_simulation()`, `simulation.py:351-357`) ile BİREBİR
AYNI `pk_interaction_adjusted_ke()` çağrısı buraya da taşınacak.
Varsayılan `None` → N=1/2 golden-snapshot davranışı DEĞİŞMEZ.

Bu bir mimari karar değil, tespit edilmiş bir eksik-parametre hatasının
düzeltilmesi — ADIM 3.1'e doğrudan bağlanıyor.

### ADR-3: CircAdapt'e uygulanan parametre çarpanları (AV-blok dahil) → Bliss-tipi, ilaç-başına çarpımsal birikim, TEK FORMÜL

**Karar:** `integrate_drug_with_circadapt.py`'nin bugün zaten uyguladığı
"her ilaç için ayrı ayrı, sırayla `t_cycle`/`Sf_act`/`ArtVen.p0`/
`c_tau_av1`'i kendi fraksiyonuyla çarp/böl" yaklaşımı **CircAdapt'e
uygulanan TÜM parametreler için kanonik/tek doğru formül** olarak
kabul ediliyor — çünkü (a) Bliss independence'ın "kalan fraksiyonların
çarpımı" formülasyonuyla birebir örtüşüyor (§1.2, §1.8), (b) QSP
kardiyovasküler pratiğiyle uyumlu (§1.8), (c) N=4 ilaçla 24 permütasyonun
TAMAMEN sıra-bağımsız olduğu bu oturumda ÇALIŞMA-ZAMANINDA doğrulandı
(`N_DRUG_AUDIT.md` §2 Şüphe G güncellemesi).

**Şüphe E'nin (%52 sapma, N=5) doğrudan çözümü:** İstatistiksel motorun
`pd.av_conduction_cumulative_multiplier()` fonksiyonu -- bugün TEK bir
lumped (additive/Loewe kombine) `hr_fraction`'ı bir kez `k_factor`'a
bölüyor -- **kaldırılıp yerine, `integrate_drug_with_circadapt.py`'deki
`cumulative_av_conduction_multiplier()` ile AYNI matematiği (ilaç-başına
AYRI AYRI, sırayla çarpımsal bölme) uygulayan bir versiyonla
DEĞİŞTİRİLECEK.** Bu, "yaklaşıklık" olmaktan çıkıp iki motorun BİREBİR
AYNI formülü paylaşmasını sağlıyor — sapma ölçüp belgelemek yerine
DOĞRUDAN GİDERİLİYOR. (İstatistiksel motorun kendisi CircAdapt'i
çalıştırmadığı için, bu fonksiyon ilaç-başına `hr_fraction` LİSTESİNİ
parametre olarak alacak şekilde imzası değişecek — çağıran kodun
`effect_hr_list`'i zaten elinde var, `simulation.py:373` civarı.)

**Not — HR/SBP büyüklüğü ile AV-multiplier'ın AYRI kombinasyon kuralları
kullanması NEDEN tutarsızlık değil:** HR/SBP'nin kendisi (Loewe ile
birleştiriliyor) ile CircAdapt'e uygulanan parametre ÇARPANI (Bliss-tipi
çarpımsal) FARKLI ŞEYLER ölçüyor — biri "birleşik etkinin BÜYÜKLÜĞÜ ne
olmalı" (klinik doz-yanıt sorusu, Loewe'nin cevapladığı), diğeri "bu
büyüklüğe ulaşmak için CircAdapt parametresi NASIL DEĞİŞTİRİLMELİ" (bir
uygulama detayı, önceden zaten çarpımsal). Loewe'den çıkan BİRLEŞİK
`hr_fraction`'ı CircAdapt'e uygularken yine TEK SEFERDE (lumped)
uygulamak yerine, ilaç-başına ayrıştırıp çarpımsal biriktirmek —
tam olarak CircAdapt motorunun zaten yaptığı şey.

### ADR-4: Zıt yönlü Emax'lı ilaçlar → literatürde çözüm yok, PRAGMATİK mühendislik kararı (açıkça etiketlenmiş)

**Karar:** Hiçbir klasik yöntem (Loewe, Bliss, HSA, MuSyC, ZIP) zıt yönlü
etkileri formel olarak ele almıyor (§1.1-§1.5, doğrulandı). Bu nedenle:
ilaçlar Emax işaretine göre iki gruba ayrılacak (azaltanlar / artıranlar),
HER GRUP kendi içinde Loewe ile birleştirilecek (aynı yön içinde Loewe
geçerli), İKİ GRUBUN NET SONUCU ise BASİT FARK (grup A birleşik etki -
grup B birleşik etki) ile hesaplanacak.

**Bu literatürden gelen bir yöntem DEĞİL** — projenin kendi mühendislik
kararı. `RESEARCH_N_DRUG.md`'de ve ilgili kod docstring'inde `⚠️ temsili /
mühendislik kararı, literatür kaynağı yok` şeklinde AÇIKÇA etiketlenecek
(bu bir sayısal katsayı değil bir KOMBİNASYON KURALI olduğu için
"KIRMIZI ÇİZGİ" kuralının harfi bu maddeye uygulanmıyor, ama ruhu
uygulanıyor — belirsizlik gizlenmeyecek, streamlit arayüzünde "karma
yönlü kombinasyon: bu bir mühendislik yaklaşımıdır, klinik literatürde
doğrudan karşılığı yoktur" uyarısı gösterilecek).

### ADR-5: `min(Emax)` tavanı → KALDIRILMIYOR, belgelenmesi güçlendiriliyor

**Karar:** Şüphe C zaten çürütülmüştü (tavan zaten belgeli). Literatür
araştırması bunu teyit ediyor: hiçbir yöntem (GIE dahil) bu tavanı
prensipli şekilde kaldırmıyor. N büyüdükçe (N=5,6,7,8'de en düşük-Emax'lı
TEK ilaç tüm kombinasyonun tavanını belirleme olasılığı artıyor) bu kısıt
DAHA SIK bağlayıcı hale geliyor — bu yüzden ADIM 6'da arayüzde N≥3'te
"kombinasyonun tavanını hangi ilaç belirliyor" bilgisi AÇIKÇA
gösterilecek (docstring'deki mevcut uyarı UI'a taşınacak).

### ADR-6: PD `interaction_matrix` terimi asimetrisi (Şüphe B) → simetrikleştir + doygunluk kuralı

**Karar:** `simulation.py:378`'deki terim `emax_a` yerine
`(emax_a+emax_b)/2` kullanacak şekilde simetrikleştirilecek (`build_
interaction_matrix()` (a,b) VE (b,a) ikisini de üretecek biçimde
güncellenecek ya da tek girişte simetrik ortalama uygulanacak — ADIM 3'te
karar verilecek implementasyon detayı). Doygunluk kuralı: toplam
interaction-term katkısı, additive terimin KENDİSİYLE aynı `np.clip(...,
0, None)` sınırını zaten paylaşıyor (crash yok) — ancak N≥3'te birden
fazla PD interaction kaydı varsayıldığında toplamın fizyolojik olmayan
büyüklüklere ulaşabildiği (§Şüphe B, N=6 deneyinde -232.6 bpm) ADIM 6'da
kullanıcıya AÇIKÇA gösterilecek bir "aşırı doygunluk" uyarısı olarak ele
alınacak; şu an config'te sadece 1 kayıt olduğu için bu N=1/2 golden
davranışını ETKİLEMİYOR.

### ADR-7 (2026-08-01 revizyon): ADR-4'ün "gruplama+fark"ı → CircAdapt-mirror'lı mekanistik fraksiyon çarpımı

**Gerekçe:** ADR-4'te kabul edilen "gruplama+fark" (ilaçları Emax işaretine
göre iki gruba ayırıp her grubu Loewe ile birleştirdikten sonra BASİT
FARKINI almak), o an bilinen tüm klasik yöntemlerin (Loewe, Bliss, HSA,
MuSyC, ZIP) bu senaryoyu kapsamadığı doğru tespitiyle alınmış bir karardı.
Ancak proje, kendi CircAdapt entegrasyonunda (mekanik kalp simülasyonu)
ZATEN daha iyi bir çözüme sahipti: `integrate_drug_with_circadapt.py >
apply_drug_effect_to_circadapt()`/`run_with_multiple_drugs()`, nabzı bpm
deltası toplayarak DEĞİL, her ilacın kendi İZOLE `hr_fraction`'ının
(yeni_hr/bazal_hr) `General.t_cycle` üzerinde ÇARPIMSAL olarak
birikmesiyle birleştiriyor — bu formül fraksiyonun yönünden (1'in üstünde/
altında olmasından) tamamen bağımsız, yani zıt yönlü ilaçlar için hiçbir
özel gruplama/işaret ayrımına ihtiyaç duymuyor. Bu formülün order-
independent olduğu ölçülerek doğrulanmıştı (N_DRUG_AUDIT.md, 24/24
permütasyon bit-identical) — yani zaten proje-içi kanonik/güvenilir kabul
edilen bir mekanizma.

**Karar:** `pd.grouped_loewe_combined_effect()`'in karma-yönlü dalı, ADR-4'ün
"gruplama+fark" formülü YERİNE, yeni `pd.mechanistic_fraction_combined_effect()`
fonksiyonunu kullanacak şekilde değiştirildi — bu fonksiyon CircAdapt'in
t_cycle formülünün istatistiksel katmandaki BİREBİR mirror'ı. Aynı-yönlü
dal (`loewe_combined_effect()`) DOKUNULMADI (MUTLAK KURAL #1, N=1/N=2
davranışı sessizce değişmiyor).

**HR/SBP ayrımı — dürüstlük notu:** CircAdapt'te HR için gerçek bir
kanonik formül var (t_cycle), ama SBP CircAdapt'te Sf_act/ArtVen.p0'dan
PV-loop simülasyonuyla EMERGENT olarak çıkıyor — kapalı formda bir "SBP
fraksiyonu" formülü YOK, yani SBP tarafında mirror'lanacak bir referans
yok. `mechanistic_fraction_combined_effect()` SBP'ye de aynı çarpımsal-
fraksiyon yaklaşımını uyguluyor, ama bu HR ile TUTARLILIK ve
öngörülebilirlik (keyfi işaret-gruplaması olmaması) gerekçesiyle alınmış
AYRI bir mühendislik kararı — CircAdapt mirror'ı olduğu iddia edilmiyor,
kod docstring'inde ve streamlit uyarısında bu ayrım açıkça belirtiliyor.

**Doğrulama:** `tests/test_n_drug_statistical.py >
test_mechanistic_fraction_combined_effect_matches_circadapt_t_cycle`, aynı
senaryoda istatistiksel motorun HR sonucunun CircAdapt'in `t_cycle`
çarpanından türetilen HR ile SAYISAL OLARAK eşleştiğini doğruluyor.
Regresyon: hiçbir golden snapshot bu karma-yönlü sonucu SABİT bir sayısal
değere pinlemiyordu (`tests/test_no_regression_n_drug.py`'deki tek test
sadece gevşek `30 < mean < 130` bpm sınırı kontrol ediyor) — bu yüzden N=1/
N=2 aynı-yönlü davranışı hiç etkilenmeden değişiklik yapılabildi.

---

## 4. Seçilen Yöntemin Mevcut Koda İlişkisi

- `loewe_combined_effect()` (istatistiksel HR/SBP büyüklüğü): **KORUNUYOR**,
  değiştirilmiyor — zaten doğru N-ilaca genellenmiş Loewe.
- `run_polypharmacy_simulation()` (additive+interaction_matrix yolu):
  **N=1/2'de KORUNUYOR** (golden snapshot), N≥3 için interaction terimi
  ADR-6'daki simetrikleştirme ile düzeltiliyor.
- `run_polypharmacy_simulation_loewe()`: PK-DDI parametreleri EKLENİYOR
  (ADR-2), davranış değişikliği sadece bu parametreler verildiğinde devreye
  giriyor.
- `pd.av_conduction_cumulative_multiplier()`: **YENİDEN YAZILIYOR** —
  CircAdapt'in ilaç-başına çarpımsal formülüyle (ADR-3) birebir aynı hale
  getiriliyor. Bu, "yaklaşıklık" olan mevcut fonksiyonun YERİNİ ALIYOR.
- `integrate_drug_with_circadapt.cumulative_av_conduction_multiplier()` ve
  `apply_drug_effect_to_circadapt()`: **KORUNUYOR**, değiştirilmiyor —
  zaten kanonik doğru formül bu (ADR-3).
- Zıt yönlü ilaçlar için yeni bir yardımcı fonksiyon (`pd.py` içine)
  eklenecek — ADR-4'teki gruplama/fark mantığını uygulayan, `⚠️ temsili`
  etiketli.

---

## 5. Kaynakça (tüm URL'ler bu oturumda WebSearch/WebFetch ile erişildi)

1. Foucquier J, Guedj M, Rey A ve ark. (2015). *Analysis of drug
   combinations: current methodological landscape.* Pharmacol Res
   Perspect. https://pmc.ncbi.nlm.nih.gov/articles/PMC4492765/
2. Grabovsky Y, Tallarida RJ (2004). *Isobolographic Analysis for
   Combinations of a Full and Partial Agonist: Curved Isoboles.* J
   Pharmacol Exp Ther 310(3):981-986.
   https://jpet.aspetjournals.org/article/S0022-3565(24)31524-1/abstract
3. Alsahli, ve ark. *Isobologram Analysis: A Comprehensive Review of
   Methodology and Current Research.* Front Pharmacol (2019).
   https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2019.01222/full
4. Wooten DJ, Meyer CT, Lubbock ALR ve ark. (2021). *MuSyC is a consensus
   framework that unifies multi-drug synergy metrics for combinatorial
   drug discovery.* Nat Commun 12:4607.
   https://www.nature.com/articles/s41467-021-24789-z
5. synergy paketi dokümantasyonu (MuSyC modeli).
   https://synergy.readthedocs.io/en/latest/models/synergy/musyc.html
6. Yadav B, Wennerberg K, Aittokallio T, Tang J (2015). *Searching for
   Drug Synergy in Complex Dose–Response Landscapes Using an Interaction
   Potency Model.* Comput Struct Biotechnol J 13:504-513.
   https://www.sciencedirect.com/science/article/pii/S2001037015000422
7. Wooten DJ, ve ark. *synergy: a Python library for calculating,
   analyzing and visualizing drug combination synergy.* Bioinformatics
   37(10):1473-1475 (2021).
   https://academic.oup.com/bioinformatics/article/37/10/1473/5909985
8. Ianevski A, Giri AK, Aittokallio T (2022). *SynergyFinder 3.0: an
   interactive analysis and consensus interpretation of multi-drug
   synergies across multiple samples.* Nucleic Acids Res 50(W1):W739-W743.
   https://academic.oup.com/nar/article/50/W1/W739/6586861
9. Chou TC. *Drug Combination Studies and Their Synergy Quantification
   Using the Chou-Talalay Method* (özet).
   https://www.researchgate.net/publication/41000828_Drug_Combination_Studies_and_Their_Synergy_Quantification_Using_the_Chou-Talalay_Method
10. Efe Y, ve ark. *Statistical Assessment of Drug Synergy from In Vivo
    Combination Studies Using Mouse Tumor Models.* Cancer Res Commun
    3(10):2146 (2023) — Bliss/HSA formülasyonları.
    https://aacrjournals.org/cancerrescommun/article/3/10/2146/729692/
11. Fu F, ve ark. (2022). *A novel cardiovascular systems model to
    quantify drugs effects on the inter-relationship between
    contractility and other hemodynamic variables.* CPT Pharmacometrics
    Syst Pharmacol.
    https://ascpt.onlinelibrary.wiley.com/doi/10.1002/psp4.12774
12. Musante CJ, ve ark. (2019). *Quantitative Systems Pharmacology: An
    Exemplar Model-Building Workflow With Applications in Cardiovascular,
    Metabolic, and Oncology Drug Development.* CPT Pharmacometrics Syst
    Pharmacol. https://pmc.ncbi.nlm.nih.gov/articles/PMC6617832/
13. FDA / ICH. *M12 Drug Interaction Studies* (nihai rehber, Ağustos
    2024). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m12-drug-interaction-studies
    ; nihai metin: https://database.ich.org/sites/default/files/ICH_M12_Step4_Guideline_2024_0521_0.pdf

---

## 6. Onay Bekleniyor

Bu ADR'nin özeti:
1. **Loewe/GIE korunuyor** (istatistiksel HR/SBP), min(Emax) tavanı
   kaldırılmıyor ama arayüzde daha görünür hale getiriliyor.
2. **PK-DDI eksikliği (Şüphe A)** — yöntem kararı değil, doğrudan bug fix.
3. **AV-blok sapması (Şüphe E, %52)** — istatistiksel motor CircAdapt'in
   ilaç-başına çarpımsal (Bliss-tipi) formülüyle BİREBİR HİZALANARAK
   ÇÖZÜLÜYOR, sadece ölçülüp belgelenmiyor.
4. **Zıt yönlü ilaçlar** — literatürde çözüm yok, gruplama+fark şeklinde
   AÇIKÇA ETİKETLENMİŞ bir mühendislik kararı.
5. **MuSyC/ZIP/3. parti paketler** — kalibrasyon verisi eksikliği ve
   "ölçmeden güvenme" kuralı gereği KULLANILMIYOR.

ADIM 2'ye (golden snapshot testleri) geçmeden önce bu kararların onayını
bekliyorum.
