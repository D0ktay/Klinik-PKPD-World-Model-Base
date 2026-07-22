# Kalibrasyon Raporu

Bu rapor, projedeki HER parametrenin nereden geldiğini (gerçek literatür /
temsili varsayım / kullanıcı girdisi) ve modelin çıktılarının yayınlanmış
klinik veriyle nerede uyuştuğunu, nerede uyuşmadığını **dürüstçe**
belgeliyor. Amaç: "kör kör her şey doğru" iddiası yerine, "şuraya kadar
güvenebilirsin, burada dikkatli ol" diyen bir mühendislik duruşu (Faz 13).

Canlı, otomatik bir versiyonu için: `provenance_report()` (Faz 14) ve
Streamlit arayüzündeki "Bu sonuç neye dayanıyor?" bölümü.

---

## 1. İlaç Parametreleri — Kaynak Durumu

| İlaç | PK parametreleri | PD parametreleri (emax/ec50) | Doğrulama |
|---|---|---|---|
| **Esmolol** (`beta_bloker`) | ✅ Gerçek (Wiest 1991, FDA label; Faz 9'da openFDA ile bağımsız doğrulandı) | ⚠️ Temsili | ✅ Faz 13: onset/duration testleri geçti |
| **Nikardipin** | ✅ Gerçek (Clinical Pharmacokinetics 2006, DailyMed) | ⚠️ Temsili | ❌ Doğrulanmadı (Faz 13 kapsamı sadece esmolol) |
| **Dobutamin** | ✅ Gerçek (Kates & Leier 1978), ama **dose_mg_per_kg bolus-eşdeğeri bir yaklaşıklık** (gerçek klinik kullanım sadece infüzyon) | ⚠️ Temsili | ❌ Doğrulanmadı |
| **Digoksin** | ✅ Gerçek (standart ders kitabı: t½, Vd, renal atılım fraksiyonu) | ⚠️ Temsili | ❌ Doğrulanmadı |
| **Metoprolol** (`drugs_verified.yaml`) | ✅ Gerçek (FDA etiketi: doz, onset zamanlaması); t½/Vd ders kitabı | ⚠️ Temsili | ❌ Doğrulanmadı |
| **Sodyum Nitroprussid** | ✅ Gerçek (FDA etiketi: t½=2dk, Vd=ekstrasellüler sıvı) ama **dose_mg_per_kg bolus-eşdeğeri bir yaklaşıklık** | ⚠️ Temsili | ❌ Doğrulanmadı |
| **"Örnek Vazodilatör"** (`vazodilator`) | ❌ Tamamen temsili/uydurma | ⚠️ Temsili | ❌ |

**Okuma:** "✅ Gerçek" işaretli parametreler (`ka`/`ke_mean`/`vd_per_kg`/doz),
gerçekten yayınlanmış bir kaynaktan (FDA etiketi, hakemli çalışma, ya da
standart ders kitabı) geliyor -- `configs/drugs.yaml` ve
`configs/drugs_verified.yaml`'daki yorumlarda tek tek kaynak gösterildi.
**`emax_hr`/`emax_sbp`/`ec50` (doz-yanıt büyüklüğü) HİÇBİR İLAÇTA gerçek
bir doz-yanıt çalışmasından kalibre edilmedi** -- bunlar "makul görünen,
yönü doğru" temsili sayılar. Bu, projenin en büyük tek kalibrasyon
boşluğu.

---

## 2. Diğer Fizyolojik Parametreler

| Mekanizma | Durum |
|---|---|
| Böbrek/karaciğer fonksiyonunun ke üzerindeki etkisi (Faz 8) | ✅ Yön/varlık gerçek (esmolol=etkilenmez, digoksin=%65 renal), kesin oran (`renal_clearance_fraction`) ders kitabı |
| Keo (etki bölgesi gecikmesi, Faz 5) | ⚠️ Tamamen temsili -- hiçbir ilaçta yayınlanmış bir Keo çalışmasından gelmedi |
| Potasyum -> AV iletim gecikmesi (Faz 11) | ✅ Yön gerçek (hiperkalemi AV iletimini yavaşlatır, iyi bilinen fizyoloji), eğim (0.3) temsili |
| Kalsiyum -> kontraktilite (Faz 11) | ✅ Yön gerçek (Ca-kontraktilite ilişkisi, iyi bilinen fizyoloji), eğim (0.08) temsili |
| Kalp yetmezliği / hipertansiyon profilleri (Faz 12) | ✅ Mekanizma gerçek (KY=azalmış kontraktilite, HT=artmış direnç+basınç setpoint'i), büyüklük (%40, %30) temsili |
| Polifarmasi interaction_matrix (Faz 10) | ⚠️ Varsayılan olarak KULLANILMIYOR (saf toplamsal); örnek çarpan (0.5) tamamen temsili |
| İki-kompartmanlı esmolol k10/k12/k21 (Faz 4) | ✅ alpha/beta gerçek, ama santral hacim (Vc=0.5 L/kg) VARSAYIM (esmolole özgü ayrı bir Vc yayını bulunamadı) |

---

## 3. Faz 13 Doğrulama Sonuçları — `tests/test_clinical_validation.py`

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
çalışmadan türetilmedi (Faz 1'de temsili olarak seçildi), yakınlık
büyük olasılıkla TESADÜFİ. Gerçek, sağlam kanıt Test #5'te: modelin
konsantrasyonu 60 dakikada pik değerin ~%2'sine iniyor (tek-bolus PK
kinetiği), oysa published çalışmadaki etki İNFÜZYON SÜRESİNCE
SÜRDÜRÜLDÜ -- bu, modelin **idame infüzyon tedavisini modelleyemediği**
anlamına gelen, dürüstçe kabul edilmesi gereken bir kapsam sınırı.

**Test yazılırken keşfedilen ayrı bir nüans:** Etki (`effect_fraction`),
konsantrasyon kadar hızlı sönmüyor (60 dk'da hâlâ pik etkinin ~%11'i
kalıyor, konsantrasyonun ~%2'sine karşın) -- çünkü Emax modelinde pik
etki zaten tam doygunluğa ulaşmıyor (%84.5, %100 değil). Bu gerçek bir
PK/PD davranışı, bug değil, ama saf konsantrasyon karşılaştırmasından
farklı yorumlanmalı.

---

## 4. Genel Sonuç ve Öneriler

1. **PK'nin "iskelet" kısmı (ka/ke/Vd, dozlar) genel olarak güvenilir** --
   çoğu gerçek literatürden, birden fazla bağımsız kaynaktan (FDA etiketi
   + RxNorm + ders kitabı) çapraz doğrulandı.
2. **PD'nin "büyüklük" kısmı (emax/ec50) HİÇBİR İLAÇTA gerçek bir doz-
   yanıt çalışmasından kalibre edilmedi.** Bu projenin gerçek bir klinik/
   araştırma aracına dönüşmesi için yapılması gereken EN ÖNEMLİ iş budur.
3. **Model, tek-doz/bolus senaryolarını temsil ediyor, sürekli infüzyon
   tedavisini DEĞİL.** İdame infüzyonu olan ilaçlar (dobutamin,
   nitroprussid) için kullanılan "bolus-eşdeğeri" dozlar birer
   yaklaşıklıktır.
4. Elektrolit/komorbidite çarpanlarının YÖNÜ gerçek fizyolojiye dayanıyor,
   ama KESİN büyüklükleri (eğimler, yüzdeler) hiçbir hasta kohortundan
   kalibre edilmedi -- bunlar "doğru yönde, kalibrasyon gerektiren"
   demonstratif parametreler.

Bu proje bir **kavram-kanıtı (proof-of-concept)** olarak tasarlandı; yukarıdaki
sınırlamalar bilinçli mühendislik kararlarıdır, gözden kaçan hatalar değil.
