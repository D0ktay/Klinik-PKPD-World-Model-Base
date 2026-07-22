"""
Gerçek İlaç Veritabanlarına Bağlantı (RxNorm + openFDA)
==========================================================

Faz 9: Şu ana kadar her ilacı elle YAML'a giriyorduk. Bu modül, halka
açık iki ücretsiz tıbbi veri API'sinden gerçek veri çekmeyi sağlıyor:

1. RxNorm (ABD Ulusal Tıp Kütüphanesi, api anahtarı gerekmez) --
   bir ilaç adından standart bir tanımlayıcı (RxCUI) çeker. Bu,
   "bu ilaç gerçekten var ve şu resmi kimliğe sahip" doğrulamasını
   sağlar.

2. openFDA (api anahtarı gerekmez) -- bir ilacın FDA prescribing
   information'ından (etiket) İLGİLİ METİN BÖLÜMLERİNİ getirir
   ("clinical_pharmacology", "dosage_and_administration" vb.).

ÖNEMLİ TASARIM KARARI: openFDA'dan gelen metin TAM OTOMATİK
parse EDİLMEZ (örn. "half-life: 3.5 hours" gibi bir cümleden regex'le
sayı çıkarmaya çalışmak). Bunun nedeni: FDA etiketleri serbest metin,
ilaçtan ilaca format/birim/cümle yapısı çok farklı -- otomatik
sayısal çıkarım güvenilir olmaz ve SESSİZCE YANLIŞ bir PK parametresi
üretme riski taşır (bu proje boyunca defalarca gördüğümüz gibi,
"gerçek görünen ama yanlış" bir sayı, hiç sayı olmamasından daha
tehlikelidir). Bunun yerine bu modül YARI-OTOMATİK bir akış sağlar:
ilgili metin bölümünü getirir, geliştirici bunu OKUYUP PK
parametrelerini kendisi (Phase 1-8'de esmolol/nikardipin/dobutamin/
digoksin için yapıldığı gibi) elle kalibre eder ve kaynağı
configs/drugs_verified.yaml'a source_url + retrieved_date ile kaydeder.
"""

import json
import urllib.parse
import urllib.request
from datetime import date, datetime

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

DEFAULT_LABEL_SECTIONS = [
    "clinical_pharmacology",
    "dosage_and_administration",
    "indications_and_usage",
]


def lookup_rxcui(drug_name: str, timeout: float = 10.0) -> dict:
    """
    RxNorm'dan bir ilaç adının standart RxCUI tanımlayıcısını çeker.
    Bu, "bu ilaç RxNorm'da tanınan gerçek bir ilaç mı?" sorusuna
    resmi bir cevap verir -- kod bir uydurma isimle çalıştırılırsa
    rxcui=None döner, bu da yanlış/uydurma bir ilaç adı girildiğini
    fark etmeyi sağlar.

    Dönüş: {"drug_name", "rxcui", "source_url", "retrieved_date"}
    """
    url = f"{RXNORM_BASE}/rxcui.json?name={urllib.parse.quote(drug_name)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read())

    rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
    return {
        "drug_name": drug_name,
        "rxcui": rxcui_list[0] if rxcui_list else None,
        "source_url": url,
        "retrieved_date": date.today().isoformat(),
    }


def fetch_fda_label_sections(drug_name: str, sections: list[str] | None = None,
                              route: str | None = None, timeout: float = 10.0) -> dict:
    """
    openFDA'dan bir ilacın FDA etiketinden ilgili metin bölümlerini çeker.

    drug_name: jenerik ad (openfda.generic_name alanında aranır)
    sections: hangi etiket bölümlerinin isteneceği (varsayılan:
        DEFAULT_LABEL_SECTIONS)
    route: opsiyonel -- "INTRAVENOUS" gibi bir uygulama yolu filtresi
        (aynı ilacın oral/IV formları farklı PK profiline sahip olabilir,
        örn. metoprolol tartrate IV vs oral)

    Dönüş: {"drug_name", "found", "sections" (bulunan bölümlerin ham
        metni -- PARSE EDİLMEMİŞ), "source_url", "retrieved_date"}

    NOT: Bu fonksiyon sayısal PK parametresi ÇIKARMAZ, sadece ham metni
    getirir. Geliştirici bu metni okuyup parametreleri kendisi kalibre
    etmelidir (bkz. modül docstring'i).
    """
    if sections is None:
        sections = DEFAULT_LABEL_SECTIONS

    # Çok kelimeli ilaç adları tırnaksız verilirse openFDA kelimeleri AYRI
    # AYRI eşleştirir (ör. "sodium nitroprusside" -> "sodium fluoride" gibi
    # alakasız bir sonuç döndürebilir) -- tam ifade eşleşmesi için tırnak
    # gerekiyor. Bunu deneyerek bulduk (bkz. modül testleri).
    quoted_name = f'"{drug_name}"' if " " in drug_name else drug_name
    query = f"openfda.generic_name:{urllib.parse.quote(quoted_name)}"
    if route:
        query += f"+AND+openfda.route:{urllib.parse.quote(route)}"
    url = f"{OPENFDA_BASE}?search={query}&limit=1"

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read())

    results = data.get("results", [])
    retrieved = date.today().isoformat()
    if not results:
        return {
            "drug_name": drug_name, "found": False, "sections": {},
            "source_url": url, "retrieved_date": retrieved,
        }

    label = results[0]
    extracted = {}
    for section in sections:
        value = label.get(section)
        extracted[section] = value[0] if value else None

    return {
        "drug_name": drug_name,
        "found": True,
        "generic_name": label.get("openfda", {}).get("generic_name"),
        "sections": extracted,
        "source_url": url,
        "retrieved_date": retrieved,
    }


def summarize_for_calibration(drug_name: str, route: str | None = None,
                               max_chars_per_section: int = 500) -> str:
    """
    lookup_rxcui + fetch_fda_label_sections'ı birleştirip, bir geliştiricinin
    terminalde okuyup PK parametrelerini elle kalibre etmesi için
    kısaltılmış, okunabilir bir özet üretir. Hiçbir sayı otomatik
    ÇIKARILMAZ -- bu fonksiyon sadece "nereye bakman gerektiğini" gösterir.
    """
    rxcui_info = lookup_rxcui(drug_name)
    label_info = fetch_fda_label_sections(drug_name, route=route)

    lines = [
        f"İlaç: {drug_name}",
        f"RxCUI: {rxcui_info['rxcui']} (kaynak: {rxcui_info['source_url']})",
    ]
    if not label_info["found"]:
        lines.append("openFDA'da etiket bulunamadı.")
        return "\n".join(lines)

    lines.append(f"openFDA jenerik adı: {label_info.get('generic_name')}")
    lines.append(f"Kaynak: {label_info['source_url']}")
    for section, text in label_info["sections"].items():
        lines.append(f"\n--- {section} ---")
        if text:
            lines.append(text[:max_chars_per_section] +
                          ("..." if len(text) > max_chars_per_section else ""))
        else:
            lines.append("(bu bölüm etikette yok)")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # FDA etiket metinleri rastgele Unicode karakterler içerebilir (ör. "≥")
    # -- Windows konsolunun varsayılan kod sayfası bunları encode edemeyip
    # çökebiliyor. errors="replace" ile script'in bu yüzden çökmesini önlüyoruz.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Faz 9'da configs/drugs_verified.yaml'a eklenen 5 ilaç için hızlı
    # bir keşif çıktısı -- kalibrasyon sırasında gerçekten böyle kullanıldı.
    for name, route in [
        ("esmolol", None),
        ("metoprolol", "INTRAVENOUS"),
        ("sodium nitroprusside", None),
        ("dobutamine", None),
        ("digoxin", None),
    ]:
        print("=" * 70)
        print(summarize_for_calibration(name, route=route))
        print()
