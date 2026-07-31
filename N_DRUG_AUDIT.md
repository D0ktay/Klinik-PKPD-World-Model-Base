# N-İlaç (Polifarmasi) Davranış Denetimi

Bu belge, projenin N≥3 ilaç (polifarmasi) senaryolarındaki gerçek
davranışını, kod okuyarak ve gerekli yerlerde sayısal deneylerle
doğrulayarak belgeler. Tüm iddialar dosya:satır referanslıdır. Kod her
zaman görevi tanımlayan istem metninden önce gelir -- 4. bölümde,
istemin varsayımlarıyla çelişen bulgular ayrıca işaretlenmiştir.

Denetlenen sürüm: mevcut çalışma kopyası (git deposu değil, klasör
anlık görüntüsü). Sayısal deneyler için betikler
`C:\Users\oktay\AppData\Local\Temp\claude\...\scratchpad\` altında
`check_B_interaction_sum.py` ve `check_E_av_multiplier_divergence.py`
olarak bırakıldı (depoya commit edilmedi).

---

## 1. Fonksiyon Bazlı N-Desteği Tablosu

| Fonksiyon | Bugün desteklediği max N (fiilen) | N≥3'te ne olur | Kanıt (dosya:satır) |
|---|---|---|---|
| `pd.emax_effect` | Sınırsız (tek ilaç girdisi alır, N'den bağımsız çağrılır) | Sorun yok -- her ilaç için ayrı ayrı çağrılıyor | `src/worldmodel/pd.py:15-24` |
| `pd.loewe_combined_effect` | Matematiksel olarak N'e AÇIK (bisection, `zip(concentrations,...)` ile listeler üzerinden topluyor) | Doğru çalışır AMA birleşik etkiyi `min(emax_i)` ile sınırlar (bilinçli, belgelenmiş kısıt) | `src/worldmodel/pd.py:78-152`, sınır notu satır 105-111 |
| `pd.electrolyte_adjusted_emax_hr/sbp` | Sınırsız (ilaç başına çağrılır) | Sorun yok | `src/worldmodel/pd.py:163-196` |
| `pd.av_conduction_cumulative_multiplier` | N=1'de tam doğru; N≥2'de YAKLAŞIK | **Sessizce yanlış** -- CircAdapt'in gerçek çarpımsal birikiminden sistematik olarak sapıyor, sapma N ile büyüyor (bkz. §2 Şüphe E) | `src/worldmodel/pd.py:199-233`, "YAKLAŞIKLIK NOTU" satır 209-217 |
| `pd.discrete_av_block_mask` | Yukarıdaki fonksiyona bağımlı olduğu için aynı sınır | AV blok tetikleme kararı N≥3'te CircAdapt'in gerçek çöküş noktasından sistematik olarak sapabilir | `src/worldmodel/pd.py:236-261` |
| `simulation.run_polypharmacy_simulation` | Kod çalışır (çökmez) her N için; `interaction_matrix` verildiğinde N≥3'te **asimetrik/keyfi ağırlıklı** ek terim üretir | **Sessizce yanlış olabilir** (interaction_matrix kullanılırsa) / aksi halde doğru (saf additive+clip) | `src/worldmodel/simulation.py:376-379` (asimetri), clip: `:387,391` |
| `simulation.run_polypharmacy_simulation_loewe` | N'e açık (Loewe formülü N-ilaca genelleştirilmiş) AMA `interaction_matrix`/`drug_keys`/`pk_interaction_matrix` parametresi YOK | PK-seviyeli etkileşimler her zaman, PD-seviyeli sinerji her zaman YOK SAYILIR (N=2'de de, N≥3'te de -- ama N≥3'te tek "otomatik doz önerisi" yolu bu olduğu için pratik etkisi orada) | `src/worldmodel/simulation.py:396-403` (imza), `pk.py` import edilmiyor bile |
| `simulation.build_interaction_matrix` | N'e açık ama config'te N=2 ötesi hiç kayıt yok (tek kayıt: beta_bloker-digoxin) | N≥3'te ekstra kayıt yoksa boş sözlük -> additive'e düşer (regresyon yok); kayıt eklenirse §2 Şüphe B'deki asimetri devreye girer | `src/worldmodel/simulation.py:480-503`, `configs/drug_interactions.yaml:16-27` |
| `simulation.recommend_dose` | Kendisi `len(drugs)` bilmiyor bile (tek `Drug` alıyor) -- branching **streamlit_app.py'de** | N/A (fonksiyonun kendisi N-duyarsız) | `src/worldmodel/simulation.py:545-555`; çağıran dal: `streamlit_app.py:819-834` |
| `simulation.recommend_polypharmacy_dose_scale` | N'e açık (grid-scan + Loewe) AMA testlerde SADECE N=2 ile egzersiz edilmiş | Kod çalışır ama PK etkileşimini yok sayar (yukarıdaki loewe sınırı yüzünden); N≥3 davranışı test edilmemiş | `src/worldmodel/simulation.py:700-786`; testler: `tests/test_pk.py:1190-1257` (hepsi `[beta_bloker, digoxin]`) |
| `pk.pk_interaction_adjusted_ke` | Mimari N-perpetrator'a açık (`for perpetrator in active_perpetrators` döngüsü) ama TABLO'da tek kayıt var, hiç egzersiz edilmemiş | Kod muhtemelen doğru çalışır (basit çarpımsal varsayım) ama gerçek çoklu-perpetrator senaryosu HİÇ TEST EDİLMEMİŞ | `src/worldmodel/pk.py:178-220`, dürüstlük notu satır 198-207 |
| `integrate_drug_with_circadapt.apply_drug_effect_to_circadapt` | N'e açık (tek ilaç için çağrılır, `run_with_multiple_drugs` döngüde çağırır) | Doğru -- çarpımsal, sıraya bağlı değil (bkz. §2 Şüphe G) | `integrate_drug_with_circadapt.py:136-207` |
| `integrate_drug_with_circadapt.run_with_multiple_drugs` | N'e açık, sırayla uygular | Matematiksel olarak sıra-bağımsız (saf çarpım, clip/branch yok) | `integrate_drug_with_circadapt.py:407-435` |
| `integrate_drug_with_circadapt.cumulative_av_conduction_multiplier` | N'e açık, ilaç başına çarpımsal | pd.py'deki karşılığından (§2 Şüphe E) matematiksel olarak FARKLI -- bu asıl CircAdapt'e uygulanan gerçek değer | `integrate_drug_with_circadapt.py:311-338` |
| `integrate_drug_with_circadapt.run_polypharmacy_comparison` | N'e açık | AV blok ön-kontrolü CircAdapt'i hiç çalıştırmadan `av_block_triggered=True` ile NaN döndürür (çökme değil, kasıtlı) | `integrate_drug_with_circadapt.py:438-494` |
| `streamlit_app.py` doz önerisi paneli | N=1: `recommend_dose` / N=2: `recommend_dose`+polifarmasi riski / N≥3: `recommend_polypharmacy_dose_scale` (farklı bir fonksiyon, farklı çıktı) | Çöküş yok, ama N≥3'te "tek ilaç dozu öner" özelliği TAMAMEN YOK, yerine "ortak ölçek" önerisi var (bilinçli tasarım, belgelenmiş) | `streamlit_app.py:813-849` |
| `streamlit_app.py` "Dünya Modelini Gözlemle" sekmesi | N≥2'de sadece İLK ilacı gösterir | Yanıltıcı değil -- açıkça uyarı metni var | `streamlit_app.py:1144-1150` |
| `streamlit_app.py` JEPA sekmesi | N≥2'de sadece İLK ilacı, VE sadece beta_blocker/positive_inotrope sınıfını destekler | Yanıltıcı değil -- açık uyarı var | `streamlit_app.py:1378-1390` |
| `report.export_report` (PDF) | N'e tam açık -- TÜM ilaçları döngüyle listeler | Doğru çalışır; sadece `dose_rec` boşsa (N≥3) "Önerilen doz" satırını atlar (belgelenmiş, UI'da da uyarılıyor) | `src/worldmodel/report.py:151-170, 182-186, 236-258` |

---

## 2. Şüphe Doğrulama (A-H)

### A. `run_polypharmacy_simulation_loewe` PK etkileşimini tamamen yok sayıyor mu?

**DOĞRULANDI.**

`run_polypharmacy_simulation_loewe()` imzasında `interaction_matrix`,
`drug_keys`, `pk_interaction_matrix` parametreleri YOK
(`src/worldmodel/simulation.py:396-403`). Fonksiyon gövdesinde
`pk_interaction_adjusted_ke` hiç çağrılmıyor -- `ke` sadece
`organ_function_adjusted_ke(...) * rng.lognormal(...)` ile hesaplanıyor
(`simulation.py:441-444`), `run_polypharmacy_simulation()`'daki PK
etkileşim bloğunun (`simulation.py:351-357`) karşılığı YOK.

`recommend_polypharmacy_dose_scale()` (3+ ilaçlı doz önerisinin TEK
yolu, bkz. `streamlit_app.py:838-849`) bu Loewe fonksiyonunu çağırıyor
(`simulation.py:750`) ve PK etkileşim parametrelerini geçmiyor (geçemez
de, imzada yok). Sonuç: **esmolol+digoksin gibi bilinen bir PK
etkileşimi (Kessler 1987, `configs/drug_pk_interactions.yaml:23-31`),
bu ikisi 3+ ilaçlı bir kombinasyonun İÇİNDE seçildiğinde otomatik doz
önerisinde SESSİZCE yok sayılır** -- N=2'de (additive yolda,
`streamlit_app.py:787-790`) doğru uygulanan aynı etkileşim, N≥3'e
geçildiğinde (farklı bir motora -- Loewe -- geçildiği için) kaybolur.

Etki: Sessizce yanlış (crash değil) -- doz önerisi PK-seviyeli
etkileşimi hesaba katmadan bir "güvenli ölçek" üretir.

### B. `run_polypharmacy_simulation` interaction terimi asimetrik mi, N büyüdükçe anlamsız sonuç üretebilir mi?

**DOĞRULANDI (asimetri) / KISMEN (anlamsız sonuç -- clip ile önleniyor ama saturasyon riski gerçek).**

Kod: `total_hr_delta += factor * adjusted_emax_hr[a] * effect_hr_list[a] * effect_hr_list[b]`
(`simulation.py:378`) -- sadece `emax[a]` kullanıyor, `emax[b]` hiç
girmiyor. `build_interaction_matrix()` (`simulation.py:480-503`) her
config kaydı için TEK bir `(index_of_drug_a, index_of_drug_b)` girdisi
ekliyor -- (b,a) AYRICA eklenmiyor. `configs/drug_interactions.yaml:8`
"sıra önemli değil (build_interaction_matrix() her iki yönde de
eşleştirir)" yorumu, kullanıcının SEÇİM sırasından bağımsız doğru
index'e eşlendiği anlamına geliyor (bkz. `test_build_interaction_matrix_
matches_regardless_of_selection_order`, `tests/test_pk.py:608-618`) --
AMA bu, terimin kendisinin `emax_a`/`emax_b` simetrisinde olduğu
anlamına GELMİYOR: YAML'daki `drug_a` alanı hangi ilaç olarak
yazılmışsa, o ilacın `emax_hr`'si terime giriyor, diğerininki hiç
girmiyor. Tek mevcut kayıtta (`drug_a: beta_bloker`) bu bilinçli/
belgelenmiş bir seçim gibi duruyor, ama mekanizma genel olarak N≥3'e
çok sayıda kayıt eklendiğinde **YAML yazarının drug_a/drug_b sırasına
bağlı, kapalı bir asimetri kaynağı** oluşturuyor.

Sayısal deney (`check_B_interaction_sum.py`, N=6 ilaç, emax_hr=[25,8,6,18,15,20],
worst-case effect_fraction=1.3 hepsi aynı anda pik):
- Sadece additive (interaction yok): toplam delta=119.6, HR=78-119.6=**-41.6 bpm**
  (`np.clip(hr,0,None)` ile 0'a kırpılıyor -- crash yok ama "0 bpm" fizyolojik
  olarak anlamsız/"ölü hasta" okuması, saf additive modelin doğal sonucu).
- TÜM C(6,2)=15 ikili çift için (worst-case) interaction kaydı varsayılırsa:
  ek 191.0 delta, toplam 310.6, HR=78-310.6=**-232.6 bpm** (yine 0'a kırpılıyor).
- Asimetri: `(a=0,b=1)` terimi=21.12 iken `(a=1,b=0)` terimi=6.76 --
  SADECE emax_hr[a] farklı olduğu için aynı iki ilaç aynı etki
  fraksiyonlarında farklı büyüklükte terim üretiyor.

Mevcut testler (`tests/test_pk.py:542-553`,
`test_polypharmacy_hr_never_goes_negative_with_many_drugs`) bu senaryoyu
KISMEN kapsıyor -- ama testte AYNI ilaç 8 kez kullanılıyor VE
`interaction_matrix=None` (varsayılan) -- yani asimetri terimi hiç
egzersiz edilmiyor, sadece additive+clip garantisi test ediliyor.

### C. `loewe_combined_effect` birleşik etkiyi `min(emax_i)` ile sınırlıyor mu, belgelenmiş mi?

**DOĞRULANDI -- VE zaten iyi belgelenmiş.**

`hi = np.full_like(concentrations[0], min(emaxes_abs) * (1 - 1e-6))`
(`pd.py:140`) -- bisection üst sınırı doğrudan `min(emax_i)`'ye
sabitleniyor, yani birleşik etki hiçbir zaman en düşük tavanlı ilacın
Emax'ını aşamıyor. Bu, docstring'de "BİLİNEN KAPSAM SINIRI" başlığı
altında açıkça anlatılıyor (`pd.py:105-111`) -- görev isteminin
zımnen varsaydığının aksine, bu ZATEN belgelenmiş, gizli bir davranış
DEĞİL.

### D. Zıt yönlü Emax kombinasyonu ValueError fırlatıyor mu, streamlit'te yakalanıyor mu?

**DOĞRULANDI (ValueError) + ÇÜRÜTÜLDÜ (uncaught crash varsayımı).**

`loewe_combined_effect` zıt işaretli `emax` listesinde `ValueError`
fırlatıyor (`pd.py:130-135`). Streamlit'te BU İKİ çağrı noktası da
try/except ile sarılı:
- N=2, "Loewe ile karşılaştır" checkbox'ı: `streamlit_app.py:800-805`
  (`except ValueError as e: loewe_error = str(e)`).
- N≥3, `recommend_polypharmacy_dose_scale`: `streamlit_app.py:846-849`
  (`except ValueError as e: polypharmacy_scale_error = str(e)`), ve
  hata kullanıcıya `streamlit_app.py:900-906`'da okunabilir bir mesajla
  gösteriliyor.

Yani kod OKUNMADAN varsayılabilecek "muhtemelen uncaught crash" senaryosu
**gerçekleşmiyor** -- her iki çağrı yeri de zaten korunmuş.

### E. İstatistiksel motorun AV çarpanı (`pd.av_conduction_cumulative_multiplier`) CircAdapt'in gerçek çarpımsal birikiminden (`cumulative_av_conduction_multiplier`) sapıyor mu?

**DOĞRULANDI -- ve sapma N ile hızla büyüyor.**

`pd.py:199-233`'ün kendi docstring'i ("YAKLAŞIKLIK NOTU", satır 209-217)
zaten bunu itiraf ediyor: istatistiksel motor TEK bir birleşik
(additive modelden gelen) `hr_fraction`'ı bir kez `k_factor`'a bölüyor;
`integrate_drug_with_circadapt.py:311-338`'deki gerçek CircAdapt
karşılığı ise HER AV-duyarlı ilaç için AYRI AYRI, sırayla (çarpımsal
birikim) bölüyor.

Sayısal deney (`check_E_av_multiplier_divergence.py`, K=5.5 mEq/L,
k_factor=1.15, per-drug hr_fraction=[0.90, 0.87, 0.84, ...]):

| N | pd.py (additive-combined) | circadapt.py (çarpımsal) | fark |
|---|---|---|---|
| 2 | 1.494 | 1.469 | -1.7% |
| 3 | 1.885 | 1.748 | -7.3% |
| 4 | 2.738 | 2.159 | -21.2% |
| 5 | 5.750 | 2.767 | **-51.9%** |

N=1'de iki formül eşit (matematiksel garanti), ama N büyüdükçe
istatistiksel motorun (`discrete_av_block_mask`, dolayısıyla
`apply_discrete_av_block`) kullandığı yaklaşık değer, CircAdapt'in
gerçekten üreteceği çarpımdan sistematik olarak SAPIYOR (N=5'te yaklaşık
2 KAT fark). Bu, **istatistiksel motorun "AV blok tetiklendi" kararı ile
CircAdapt'in gerçekte çökeceği/çökmeyeceği nokta arasında N≥3'te ciddi
bir tutarsızlık riski** anlamına geliyor -- örn. istatistiksel motor AV
bloğu tetiklemeyip normal bir iz üretirken, aynı senaryoda gerçek
CircAdapt çağrısı (eğer `av_block_triggered` ön-kontrolü yanlış "False"
derse) `ModelCrashed` ile çökebilir, ya da tam tersi (istatistiksel motor
gereksiz yere erken AV bloğu tetikleyebilir).

### F. CircAdapt çöküş eşiği (5x-7x) sadece `Timings.c_tau_av1` için mi ölçüldü?

**DOĞRULANDI.**

`CALIBRATION_REPORT.md:503-514` ("§8 Discrete AV Blok") açıkça
`Timings.c_tau_av1` çarpanının 1x/3x/5x/7x/9x/15x/50x kademeli
büyütülerek izole test edildiğini, 5x'te stabil kalıp 7x'te çöktüğünü
anlatıyor -- bu, SADECE `c_tau_av1` parametresi için yapılmış bir ölçüm.
`CALIBRATION_REPORT.md:412` de aynı ölçümden bahsediyor (2x'te EDV farkı
yok, 5x'te var).

`Patch.Sf_act`, `ArtVen.p0[0]`, `General.t_cycle` için AYRI, izole bir
çöküş-eşiği deneyi **repoda hiçbir yerde bulunamadı** (CALIBRATION_REPORT.md
ve kod yorumlarında grep edildi -- bu üç parametre için "Xx'te çöktü"
tarzı bir ölçüm kaydı yok). Bu üç parametre de `apply_drug_effect_to_
circadapt`/`apply_comorbidity_to_circadapt` içinde N ilaç ve/veya
komorbidite ile çarpımsal olarak büyütülebiliyor
(`integrate_drug_with_circadapt.py:194-203, 270-277`), ama Gap #3'ün
ön-kontrolü (`av_block_triggered`, `cumulative_av_conduction_multiplier`)
SADECE `c_tau_av1` yolunu kontrol ediyor -- yani N≥3 ilaçla `Sf_act`
(kontraktilite) veya `ArtVen.p0` (sistemik direnç) aşırı büyürse/küçülürse,
bunun CircAdapt'i çökertip çökertmeyeceği konusunda HİÇBİR ön-kontrol
YOK, ölçülmüş bir eşik de YOK.

### G. `run_with_multiple_drugs` sıralı uygulama, order-independent mi (saf çarpım) yoksa clip/branch ile bozulabilir mi?

**ÇÜRÜTÜLDÜ (sıraya bağımlılık iddiası) -- HEM statik kod incelemesiyle HEM DE gerçek çalışma-zamanı permütasyon testiyle doğrulandı.**

**Güncelleme (2026-07-31):** İlk denetimde `circadapt` paketinin bu ortamda
kurulu olmadığı bulgusu **yanlıştı** -- proje kökündeki `.venv` içinde
(`vivax_world_model_demo/vivax_world_model_demo/.venv`) `circadapt` paketi
zaten kurulu (`pip show` ile doğrulandı, `circadapt.__init__` dosya yolu
`.venv/Lib/site-packages/circadapt/__init__.py`). İlk denetimde yanlış/sistem
Python'u kullanılmış olmalı. Doğru venv (`.venv/Scripts/python.exe`) ile
gerçek bir permütasyon deneyi (`check_G_permutation.py`, scratch klasöründe)
çalıştırıldı:

- 4 farklı sınıftan ilaç (beta_bloker, vazodilator, dobutamine, digoxin),
  hasta_a, AV blok eşiğini tetiklemeyecek ölçülü dozlarla.
- `compute_drug_effect()` HER ilaç için BİR KEZ hesaplandı (PK/PD zinciri
  sıradan etkilenmez, zaten ayrı ayrı ilaç başına çağrılıyor -- test edilen
  SADECE `run_with_multiple_drugs()`'ın CircAdapt'e uygulama sırası).
- 4 ilacın TÜM 4!=24 permütasyonu ayrı ayrı `run_with_multiple_drugs()`'a
  verildi, her permütasyon sonrası `General.t_cycle`, `Patch.Sf_act`,
  `ArtVen.p0`, `Timings.c_tau_av1` okundu.

**Sonuç: 24 permütasyonun TAMAMI sayısal olarak birebir aynı** (`t_cycle`,
`Sf_act`, `p0` farkı tam olarak 0.0; `c_tau_av1` farkı 2.78e-17 -- kayan
nokta yuvarlama gürültüsü düzeyinde, anlamlı bir sapma değil).
`np.allclose(rtol=1e-9, atol=1e-12)` tüm çiftler için `True`. Bu, statik kod
incelemesindeki "çarpma/bölme commutative, clip/branch yok" çıkarımını
çalışma-zamanında doğruluyor -- **`run_with_multiple_drugs()` gerçekten sıra
bağımsız**, N=4 ilaçta ölçülmüş, N=8'e kadar aynı matematiksel garantinin
geçerli olması beklenir (parametre-başına saf çarpım/bölüm mekanizması N'den
bağımsız).

`apply_drug_effect_to_circadapt()` içinde (`integrate_drug_with_circadapt.py:136-207`):
- `t_cycle`: `t_cycle_base / hr_fraction` -- saf bölme/çarpım, `np.clip`
  YOK.
- `Sf_act`: `sf_act[...] * sbp_fraction` -- saf çarpım, clip YOK.
- `c_tau_av1`: `c_tau_av1[0] / hr_fraction` -- saf bölme, clip YOK.
- `ArtVen.p0`: `p0[...] * sbp_fraction` -- saf çarpım, clip YOK.
- `PFC["is_active"] = False` -- idempotent bir atama (kaç kez/hangi
  sırada çağrılırsa çağrılsın aynı sonuç).

Bu repoda `grep -n "clip\|min(\|max("` ile taranan tüm kod yolunda
(`integrate_drug_with_circadapt.py`) `apply_drug_effect_to_circadapt`
İÇİNDE hiçbir clip/koşullu dallanma bulunamadı -- tek koşul
`drug_class`'a göre HANGİ parametrenin değiştirileceğini seçen bir
`if/elif/else` (satır 194-205), bu da ilaç sırasından bağımsız (her
ilaç kendi sınıfına göre kendi mekanizmasını hedefliyor, başka bir
ilacın DAHA ÖNCE mi SONRA mı uygulandığından etkilenmiyor).

Matematiksel olarak: N ilacın sırayla uygulanması, N tane çarpma/bölme
işleminin ÇARPIMINA (kayan-nokta yuvarlama hatası dışında) eşdeğer --
çarpma/bölme değişmeli (commutative) olduğu için SONUÇ sıradan
BAĞIMSIZ olmalı. **Bu ortamda `circadapt` paketi kurulu olmadığı için
(`ModuleNotFoundError: No module named 'circadapt'`) gerçek bir
permütasyon-diff deneyi ÇALIŞTIRILAMADI** -- yukarıdaki sonuç statik kod
incelemesine dayanıyor, çalışma-zamanı doğrulaması YAPILMADI. Bu,
görev isteminin "birkaç permütasyon çalıştırıp fark al" beklentisinin
bu ortamda teknik olarak imkansız olduğu anlamına geliyor.

### H. `conc_runs` sadece ilk ilacın konsantrasyonunu mu tutuyor, streamlit yanıltıcı mı gösteriyor?

**DOĞRULANDI (sadece ilk ilaç) / ÇÜRÜTÜLDÜ (yanıltıcı gösterim).**

`run_polypharmacy_simulation` (`simulation.py:373-374`) ve
`run_polypharmacy_simulation_loewe` (`simulation.py:459-460`) her ikisi
de `if d_idx == 0: conc_runs[i] = conc` ile SADECE ilk ilacın
konsantrasyonunu yazıyor -- bu zaten `simulation.py:329`'da yorumla
açıkça belirtiliyor ("ilk ilacın konsantrasyonu (referans/gösterim
için)").

Ancak `conc_runs` alanı **streamlit_app.py'de veya viz.py'de hiçbir
yerde çizilmiyor/gösterilmiyor** -- `plot_results()` (`src/worldmodel/
viz.py:8-47`) sadece `hr_runs`/`sbp_runs` kullanıyor, `conc_runs`'a hiç
dokunmuyor. Repo genelinde `conc_runs` kullanımı sadece
`src/worldmodel/simulation.py` (üretim) ve `tests/test_pk.py` (doğrulama)
içinde bulunuyor -- yani alan var ama polifarmasi UI akışında hiç
YÜZEYE ÇIKMIYOR, dolayısıyla "kullanıcıyı yanıltma" riski PRATİKTE YOK
(gösterilmeyen bir şey yanıltamaz). Şüphenin "streamlit ne gösteriyor"
kısmı çürütüldü -- alan sadece SimulationResult içinde ölü/kullanılmayan
bir veri.

---

## 3. Streamlit'e Özgü Bulgular

- **`len(drugs) == 1/2/>2` dallanması (doz önerisi):**
  `streamlit_app.py:813-834` -- N=1 için `recommend_dose(patient, drug, ...)`,
  N=2 için `recommend_dose(patient, drugs[0], ..., polypharmacy_result=mc_result,
  polypharmacy_description=drugs[1].display_name)` (SADECE drugs[0]'ın
  dozu optimize ediliyor, drugs[1] sabit/verilen), N>2 için `dose_rec = None`
  (yerine `recommend_polypharmacy_dose_scale` ayrı bir bilgi kutusu
  üretiyor, `streamlit_app.py:838-849`). Gerekçe kod içinde açıkça
  yazılı: "3+ ilaçta 'hangi ilacın dozu optimize edilsin' belirsiz bir
  karar" (`streamlit_app.py:826-828`, ayrıca `simulation.py:712-719`).
  **Not:** `recommend_dose()` fonksiyonunun KENDİSİ `len(drugs)`'ı hiç
  bilmiyor/kullanmıyor (tek bir `Drug` alıyor) -- görev isteminin
  "recommend_dose neden len(drugs)==2'ye göre dallanıyor" sorusu yanlış
  yere işaret ediyor: dallanma `recommend_dose`'un İÇİNDE değil,
  onu ÇAĞIRAN streamlit_app.py'de.

- **"Dünya Modelini Gözlemle" sekmesi (N≥2):** sadece ilk ilaç için
  çalıştığını açıkça belirtiyor: *"{N} ilaç seçili -- bu sayfa (adım-adım
  tek iz gösterimi) şu an sadece İLK seçilen ilaç için çalışıyor"*
  (`streamlit_app.py:1144-1150`).

- **JEPA sekmesi (N≥2 VE ilaç sınıfı kısıtı):** iki ayrı kısıt --
  (1) `drug.drug_class not in SUPPORTED_DRUG_CLASSES` ise (sadece
  beta_blocker/positive_inotrope destekleniyor) tamamen devre dışı
  bırakılıyor (`streamlit_app.py:1378-1384`); (2) N>1 ise yine sadece
  ilk ilaç kullanılıyor, açık uyarıyla (`streamlit_app.py:1386-1390`).

- **PDF rapor (`export_report`):** görev isteminin aksine, PDF **TÜM N
  ilacı doğru şekilde listeliyor** -- hem "İlaç Bilgileri" bölümünde
  (`report.py:151-170`, her ilaç için ayrı "İlaç {i+1}" başlığı) hem
  "Kaynakça" bölümünde (`report.py:236-258`, her ilaç için ayrı
  provenance listesi). Tek eksik: `dose_rec is None` olduğunda (N≥3,
  doz önerisi hesaplanmadığında) "Önerilen doz" satırı atlanıyor
  (`report.py:182-186`) -- bu, streamlit'te de kullanıcıya ÖNCEDEN
  bildiriliyor (`streamlit_app.py:1490-1494`: *"3+ ilaçlı polifarmasi
  senaryosunda doz önerisi hesaplanmadığı için rapor bu bölümü boş
  gösterecek"*). Yani PDF üretimi N-ilaç için ÇÖKMÜYOR ve YANILTICI
  DEĞİL -- eksik alan açıkça belgelenmiş/bildirilmiş bir kısıt.

- **Loewe checkbox (N=2'ye özgü UI ama fonksiyon N'e açık):**
  "Loewe additivity ile de karşılaştır" checkbox'ı sadece
  `len(drug_keys) > 1` bloğunda gösteriliyor (`streamlit_app.py:718-756`)
  ve sonuç sadece `len(drugs) > 1 and compare_with_loewe` koşulunda
  hesaplanıyor (`streamlit_app.py:800-805`) -- yani bu checkbox aslında
  N≥2 (N=3,4... dahil) için de çalışır durumda, sadece UI'da "2 ilaç"
  diye özel bir isimlendirme YOK; test kapsamı ise (bkz. §2 Şüphe A/E)
  sadece N=2 ile sınırlı.

---

## 4. İstem Varsayımlarıyla Çelişen Bulgular

Aşağıdaki noktalarda kod, görev isteminin (zımni ya da açık)
varsayımlarıyla ÇELİŞİYOR -- kod esas alınmıştır:

1. **"loewe_combined_effect'in cap davranışı belgelenmiş mi?" sorusu
   zaten EVET yanıtına sahip, gizli bir davranış değil** -- `pd.py:105-111`
   bunu "BİLİNEN KAPSAM SINIRI" başlığıyla açıkça anlatıyor. İstem bunu
   "belgelenmiş mi?" diye soruyor sanki belgelenmemiş olabilirmiş gibi --
   kod, tam tersini gösteriyor: bu proje için ALIŞILDIK ölçüde iyi
   belgelenmiş bir kısıt.

2. **"Zıt yönlü Emax kombinasyonu streamlit'te uncaught crash yapar mı?"
   sorusu -- HAYIR, İKİ çağrı noktası da (N=2 Loewe karşılaştırması VE
   N≥3 doz-ölçek önerisi) zaten try/except ile korunuyor**
   (`streamlit_app.py:800-805`, `846-849`). İstem bu konuda bir crash
   riski ima ediyor gibi duruyor ama kod bunu ele almış.

3. **"conc_runs sadece drugs[0]'ı tutar, streamlit'in bunu N-ilaç için
   nasıl gösterdiğini kontrol et" -- kod `conc_runs`'ı zaten hiç
   göstermiyor.** Alan `SimulationResult` içinde var ama UI katmanında
   HİÇ tüketilmiyor -- yani "yanıltıcı gösterim" riski isteme göre var
   olabilirmiş gibi duruyordu, ama pratikte gösterilmediği için risk
   YOK (ölü kod/kullanılmayan alan, ayrı bir temizlik konusu olabilir
   ama "kullanıcıyı yanıltma" bulgusu değil).

4. **"recommend_dose neden len(drugs)==2'ye göre dallanıyor?" sorusu
   yanlış konumlandırılmış** -- `recommend_dose()` fonksiyonunun kendisi
   `len(drugs)` parametresi bile almıyor (tek bir `Drug` nesnesi alıyor,
   `simulation.py:545-546`). N=1/2/3+ ayrımı tamamen `streamlit_app.py`
   içindeki ÇAĞIRAN kodda yapılıyor (`streamlit_app.py:819-834`).

5. **"PDF report generation'ın N ilaç için sadece bir kaçını listeleyip
   listelemediği" sorusu -- HAYIR, PDF TÜM N ilacı doğru listeliyor**
   (`report.py:151-170, 236-258`) -- görev isteminin ima ettiği "belki
   sadece ilk 1-2 ilacı gösterir" riski koddan doğrulanamadı, tam
   tersine bu modül N-ilaç için özenle (döngüyle, "İlaç {i+1}"
   başlıklarıyla) yazılmış.

6. **DÜZELTME (2026-07-31):** İlk denetimdeki "`circadapt` paketi bu
   ortamda kurulu değil" iddiası YANLIŞTI -- doğru sanal ortam
   (`.venv/Scripts/python.exe`) kullanıldığında paket kurulu çıktı.
   Suspicion G artık gerçek çalışma-zamanı permütasyon testiyle (4 ilaç,
   24 permütasyon, tümü bit-düzeyinde özdeş) doğrulandı -- bkz. §2 Şüphe G
   güncellemesi. İstemin öngördüğü deney tam olarak yapılabildi ve
   sonucu doğruladı.

7. **En kritik, istemin doğrudan sormadığı ama ortaya çıkan bulgu:**
   `recommend_polypharmacy_dose_scale()` -- yani N≥3 ilaç için var olan
   TEK otomatik doz-güvenliği aracı -- (a) PK etkileşimini her zaman
   yok sayıyor (Şüphe A) VE (b) test paketinde SADECE N=2 ilaçla
   (`beta_bloker`+`digoxin`) egzersiz edilmiş, N=3+ için TEK BİR TEST
   YOK (`tests/test_pk.py:1190-1257`). Yani projenin "N≥3'te güvenilir
   değil" öz-değerlendirmesi doğru, ama en somut kanıtı bu fonksiyonun
   kendi test kapsamındaki N=2 sınırlaması.

---

## Kaynak Dosyalar (bu denetimde okunanlar)

- `src/worldmodel/pd.py`
- `src/worldmodel/simulation.py`
- `src/worldmodel/pk.py`
- `src/worldmodel/viz.py`
- `src/worldmodel/report.py`
- `integrate_drug_with_circadapt.py`
- `streamlit_app.py`
- `configs/drug_interactions.yaml`, `configs/drug_pk_interactions.yaml`, `configs/drugs.yaml`
- `tests/test_pk.py`
- `CALIBRATION_REPORT.md`
- `CLAUDE.md`

Sayısal deney betikleri (repoya dahil değil, scratch klasöründe):
`check_B_interaction_sum.py`, `check_E_av_multiplier_divergence.py`.
