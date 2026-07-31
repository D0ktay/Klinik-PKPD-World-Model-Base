# N-İlaç Genellemesi — Nihai Rapor

Bu rapor, sistemi N=1..8 ilaç için doğru, test edilmiş ve Streamlit'te
çalışır hale getirme görevinin (ADIM 0-7) sonucunu özetler. Ayrıntılı
kanıtlar için: `N_DRUG_AUDIT.md` (ADIM 0 denetimi), `RESEARCH_N_DRUG.md`
(ADIM 1 literatür + ADR), commit geçmişi (`c793d00`..`378ea26`).

## Ne değişti (ADIM bazında)

| ADIM | Commit | Özet |
|---|---|---|
| 0 | — | Denetim: 8 şüphenin (A-H) doğrulama/çürütme, N_DRUG_AUDIT.md |
| 1 | — | Literatür araştırması + ADR-1..6, RESEARCH_N_DRUG.md |
| 2 | `c793d00` | Golden-snapshot regresyon testleri (N=1/2, AV-duyarlı+hiperkalemi, zıt yön) |
| 3 | `8b72c5b` | PK-DDI'yi Loewe yoluna taşıma (Şüphe A), interaction terimi simetrikleştirme (Şüphe B), zıt-yön gruplama (Şüphe D), AV-blok formül hizalama (Şüphe E — en kritik düzeltme), conc_runs genişletme (Şüphe H), 33 yeni test |
| 4 | `400c1c1` | CircAdapt parametre-başına ölçülmüş çöküş eşikleri (Şüphe F), genelleştirilmiş ön-kontrol (t_cycle en kırılgan çıktı — 3.0x), 14 yeni test |
| 5 | `cb0bb20` | Doz önerisi panelinin N≥3'e açılması, hedef-ilaç seçimi, 3 yeni test |
| 6 | `378ea26` | Streamlit UI: N-ilaç grafikleri, etkileşim tablosu, karma-yön/kararsızlık uyarıları, cache, gözlemleme/JEPA sekmelerinin genelleştirilmesi |

Toplam: 267 mevcut test + 70 yeni test = **337/337 yeşil** (tüm ADIM'lar
sonunda doğrulandı, `pytest -q`).

## Hangi kanıta dayanarak

- **Ölçülmeden hiçbir sayısal iddia yapılmadı.** AV-blok sapması (N=5'te
  %52), CircAdapt parametre çöküş eşikleri (t_cycle 3.0x, Sf_act 100x,
  c_tau_av1 7.0x, ArtVen.p0 500x+'de hâlâ stabil), permütasyon
  değişmezliği (N=4, 24 permütasyon, bit-exact), grid-scan maliyeti
  (N=2→5: ~10.6s→24.4s, kabaca doğrusal) — hepsi izole betiklerle
  ölçüldü (`scripts/circadapt_parameter_crash_thresholds*.py`,
  `tests/test_n_drug_circadapt.py`).
- **Literatür kararları kaynaklı.** RESEARCH_N_DRUG.md'de 13 kaynak
  (Foucquier 2015, Grabovsky & Tallarida 2004, Wooten 2021/MuSyC, Yadav
  2015/ZIP, FDA/ICH M12 2024, vb.) taranıp özetlendi; her ADR maddesi bu
  kaynaklara veya "literatürde yok, mühendislik kararı" etiketine
  dayanıyor.
- **Uçtan uca doğrulama gerçek tarayıcıda yapıldı.** Playwright/Chromium
  (headless) ile N=3 (Esmolol+Digoksin+Dobutamin, karma yönlü) senaryosu
  gerçekten tıklanarak geçildi: İlaç Seçimi (etkileşim tablosu, hedef-ilaç
  seçimi), Simülasyon (konsantrasyon grafikleri, katkı dökümü, karma-yön
  uyarısı), CircAdapt Sonuçları (PV loop, EF/CO), Dünya Modelini Gözlemle
  (ilaç seçici), JEPA (ilaç seçici), Rapor İndir (PDF, 3 sayfa, tüm
  ilaçlar + kaynakça) — konsol hatası YOK. N=5 (karma yönlü) ve N=8
  (tekrarlı ilaç havuzu) CLI üzerinden (`compare_n_drug_polypharmacy.py`,
  ad-hoc betik) uçtan uca (istatistiksel + CircAdapt) çalıştırıldı, çökme
  yok.

## Hangi testler bunu koruyor

- `tests/test_no_regression_n_drug.py` (19 test) — N=1/N=2 golden
  snapshot, `rtol=0, atol=0` tam eşitlik. Tek KASITLI istisna: zıt-yönlü
  Loewe artık ValueError yerine tanımlı sonuç döndürüyor (ADR-4,
  kullanıcı onaylı davranış değişikliği).
- `tests/test_n_drug_statistical.py` (33 test) — N=1 tutarlılığı,
  sıfır-varyans permütasyon değişmezliği (N=2-6), monotonluk (N=1-8),
  fizyolojik sınırlar (N=8), sınır durumlar, Loewe izobol residual'i
  (N=2-8), hypothesis property-based testler.
- `tests/test_n_drug_circadapt.py` (14 test) — gerçek CircAdapt ile
  permütasyon değişmezliği (N=4), N=1-6 çökme yok, eşik-mantığı birim
  testleri, t_cycle eşiğinin CircAdapt'i hiç çağırmadan devreye girdiğini
  doğrulayan spy testi.
- `tests/test_n_drug_dose_recommendation.py` (3 test) — PK-DDI'nin
  gerçekten hesaba katıldığı (N=3), N=1/2 regresyonu, hedef-ilaç mantığı.

## Neyi yapmadım ve neden

1. **MuSyC/ZIP/3. parti sinerji paketleri entegre edilmedi.** Kalibrasyon
   verisi eksikliği ve "uydurma katsayı yasağı" nedeniyle — bkz.
   RESEARCH_N_DRUG.md ADR-1.
2. **`min(Emax)` tavanı kaldırılmadı.** Hiçbir literatür yöntemi bunu
   prensipli şekilde kaldırmıyor — sadece görünürlüğü artırıldı (ADR-5).
3. **Zıt yönlü ilaç birleştirme kuralı literatür kaynaklı değil** —
   projenin kendi mühendislik kararı, böyle etiketlendi (ADR-4).
4. **`compare_polypharmacy.py` (orijinal, N=2'ye özel demo betiği)
   DEĞİŞTİRİLMEDİ** — anlamı/amacı (esmolol+digoksin'in spesifik
   "tehlikeli kombinasyon" gösterimi) N-ilaca genellemekle bozulurdu.
   Bunun yerine YENİ bir betik (`compare_n_drug_polypharmacy.py`, N=5,
   karma yönlü) eklendi.
5. **Streamlit Cloud'a deploy YAPILMADI** — görev talimatına göre bu adım
   için kullanıcı onayı gerekiyor, henüz istenmedi/onaylanmadı.
6. **İstatistiksel motorun Monte Carlo yolu tam bit-exact permütasyon
   değişmez YAPILMADI** (mimari değişiklik gerektirir — her ilaca
   kimliğine göre bağımsız bir RNG akışı vermek) — bunun yerine
   sıfır-varyans koşulda (rastgelelik kaynağı ortadan kalktığında)
   permütasyon değişmezliği test edildi ve doğrulandı; gerçek (sigma>0)
   Monte Carlo modunda istatistiksel olarak aynı dağılım ama farklı dizi
   üretebileceği AÇIKÇA belgelendi (README §16, CALIBRATION_REPORT §10).
7. **`ArtVen.p0[0]` için "gerçek" çöküş eşiği bulunamadı** (0.01x-500x
   aralığında hiç çökmedi) — bu sınırların ÖTESİ test edilmedi, "sonsuz
   güvenli" iddia edilmiyor.
8. **`run_reference_trace()` (Dünya Modelini Gözlemle'nin bazı alt
   akışları) dobutamin/nitroprussid'in infüzyon-PK modelini kullanmıyor**
   — bu, önceki bir turdan (Faz 5.x) kalan, bu görevin kapsamına dahil
   edilmeyen bilinen bir sınır (CALIBRATION_REPORT.md §7'de zaten
   belgeli).

## Kalan bilinen sınırlar (dürüstçe)

- N büyüdükçe `min(Emax)` tavanı daha sık bağlayıcı olur — kaldırılmadı,
  sadece arayüzde gösteriliyor.
- Zıt yönlü kombinasyon birleştirme kuralı literatürden gelmiyor.
- Monte Carlo yolu ilaç seçim sırasından bit-düzeyinde bağımsız değil
  (istatistiksel olarak aynı, dizi olarak farklı).
- `ArtVen.p0[0]` çöküş eşiği tam olarak bilinmiyor (sadece ≥500x'te hâlâ
  stabil olduğu ölçüldü).
- Grid-scan (doz tarama) maliyeti N=8'de ~35-40 saniyeye kadar
  uzayabilir (ölçülen N=2-5 eğiliminin ekstrapolasyonu) — Streamlit
  spinner'ı bunu gösteriyor ama optimize edilmedi.
- ~~Streamlit Cloud'daki canlı davranış test edilmedi~~ -- **artık test
  edildi, bkz. aşağıdaki "Deploy" bölümü.**

"Her şey mükemmel" değildir — yukarıdaki sınırlar bilinçli, ölçülmüş ve
belgelenmiş kararlardır.

---

## Deploy

- **Push edilen commit:** `927f7cf12a0880d77c0616291fcc6e50e6801b3b` (ADIM 7
  sonu, `origin/master`) — `git push origin master` ile
  `github.com/D0ktay/Klinik-PKPD-World-Model-Base` reposuna gönderildi,
  `d31cbb0..927f7cf` (7 commit: baseline + ADIM 2-7).
- **Canlı URL:** https://klinik-pkpd-world-model-base-ahrm47cwvjar9cgogclrz9.streamlit.app/
  (Streamlit Community Cloud, GitHub push'a otomatik bağlı -- ayrı bir
  manuel "redeploy" tetiklemesi gerekmedi).
- **Deploy tarihi/doğrulama:** 2026-07-31. Uygulama push sonrası
  inaktivite nedeniyle "uykuda" bulundu ("Zzzz -- bu app inaktivite
  nedeniyle uykuya daldı"), "Yes, get this app back up!" ile uyandırıldı
  ve yeni commit'le başarıyla ayağa kalktı.
- **N=3 doğrulaması (canlı, Playwright/Chromium):** Esmolol+Digoksin+
  Dobutamin (karma yönlü) -- İlaç Seçimi (etkileşim tablosu "simetrik
  (N≥3)" doğru gösterdi), Simülasyon (konsantrasyon grafiği, katkı
  dökümü, karma-yön uyarısı) sorunsuz çalıştı. Konsol hatası yok.
- **N=5 doğrulaması (canlı):** +Nikardipin+Sodyum Nitroprussid eklenerek
  N=5'e çıkarıldı -- "N≥5" performans/tavan uyarısı doğru göründü,
  konsantrasyon grafiğinde 5 ayrı eğri, katkı dökümünde 5 satır, doz
  önerisi ("Önerilen en iyi doz: 20.0 mg") ve CircAdapt sekmesi (PV loop,
  EF %65, CO 6.3 L/dk, tüm 5 ilaç listeli) doğru render oldu. Konsol
  hatası yok (gözlenen birkaç "404" kaynak hatası Streamlit Cloud'un
  kendi statik varlıklarına -- ör. favicon/analytics -- ait, uygulama
  işlevselliğini etkilemedi).
- **Bulunan/düzeltilen bir test-altyapısı detayı (uygulama kodu
  DEĞİL):** Streamlit Community Cloud, uygulamayı üst sayfanın (Fork/
  GitHub butonlu chrome) İÇİNDE ayrı bir iframe'de (`/~/+/` yolu)
  render ediyor -- yerel `streamlit run` sunucusunda bu iframe sarmalayıcı
  yok. Playwright betiklerinin doğru iframe'i hedeflemesi gerekti; bu
  SADECE canlı doğrulama betiğini etkiledi, uygulamanın kendisinde bir
  sorun değil.

**Sonuç: canlı deploy, N=3 ve N=5 için yerel ortamdaki davranışla
BİREBİR tutarlı çalışıyor.**
