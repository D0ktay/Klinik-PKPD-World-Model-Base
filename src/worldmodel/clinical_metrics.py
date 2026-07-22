"""
Klinik Olarak Anlamlı Metrikler (EK GÖREV B)

Şu ana kadar CircAdapt sonuçlarını ham basınç/hacim sayılarıyla
gösteriyorduk. Bu dosya, bunları kardiyolojinin gerçekte kullandığı
STANDART metriklere çeviriyor -- "kalp iyi mi kötü mü çalışıyor"
sorusuna, bir doktorun da tanıyacağı tek bir sayıyla cevap vermek için.
"""

# EF (ejeksiyon fraksiyonu) normal aralıkları -- klinik literatürde
# yaygın kabul gören eşikler.
EF_NORMAL_MIN = 55.0     # >= bu değer: normal pompalama
EF_MILD_MIN = 40.0        # bu değer - EF_NORMAL_MIN arası: hafif azalmış
# < EF_MILD_MIN: düşük (kalp yetmezliği belirtisi)

# CO (kardiyak output) normal aralığı.
CO_NORMAL_MIN = 4.0
CO_NORMAL_MAX = 8.0
CO_BORDERLINE_MIN = 2.5
CO_BORDERLINE_MAX = 10.0


def ejection_fraction(edv: float, esv: float) -> float:
    """
    EF / ejeksiyon fraksiyonu -- kalbin HER ATIŞTA içindeki kanın
    YÜZDE KAÇINI pompaladığı. Kardiyolojide "kalp ne kadar iyi
    pompalıyor" sorusunun standart cevabıdır.

    edv: EDV / end-diastolic volume (kalbin DOLDUĞU andaki hacmi --
        "diastol" kalbin gevşeyip kanla dolduğu faz; bu yüzden EN
        YÜKSEK hacim burada ölçülür)
    esv: ESV / end-systolic volume (kalbin BOŞALDIĞI andaki hacmi --
        "sistol" kalbin kasılıp kan pompaladığı faz; bu yüzden EN
        DÜŞÜK hacim burada ölçülür)

    Formül: EF = (EDV - ESV) / EDV * 100

    Normal: %55-70 (kalp iyi pompalıyor)
    Hafif azalmış: %40-54
    Düşük (kalp yetmezliği belirtisi): <%40 (kalp yeterince güçlü
        pompalayamıyor -- örn. sistolik kalp yetmezliğinde görülen tablo)
    """
    if edv <= 0:
        return 0.0
    return (edv - esv) / edv * 100.0


def cardiac_output(edv: float, esv: float, heart_rate_bpm: float) -> float:
    """
    CO / kardiyak output (kalp debisi) -- kalbin DAKİKADA pompaladığı
    TOPLAM kan miktarı, litre cinsinden. Vücudun organlarını yeterince
    kanlandırıp kanlandıramadığının temel göstergesi.

    Önce atım hacmi (stroke volume -- kalbin TEK BİR atışta pompaladığı
    kan miktarı) hesaplanır: SV = EDV - ESV (mL). Sonra bu, kalp hızıyla
    (dakikadaki atış sayısı) çarpılır:

    CO (L/dk) = SV (mL) * nabız (bpm) / 1000

    Normal aralık: 4-8 L/dk (yetişkin, dinlenme halinde).
    """
    stroke_volume_ml = edv - esv
    return stroke_volume_ml * heart_rate_bpm / 1000.0


def classify_cardiac_function(ef: float, co: float) -> dict:
    """
    EF ve CO'yu normal aralıklarına göre YEŞİL (normal) / SARI (dikkat)
    / KIRMIZI (anormal) olarak etiketler -- kullanıcı tek bakışta
    "bu kalp iyi mi kötü mü çalışıyor" sorusuna cevap alabilsin diye.

    Dönüş: {"overall_color", "ef_color", "co_color", "ef_label", "co_label", "summary"}
    """
    if ef >= EF_NORMAL_MIN:
        ef_color = "yeşil"
    elif ef >= EF_MILD_MIN:
        ef_color = "sarı"
    else:
        ef_color = "kırmızı"

    if CO_NORMAL_MIN <= co <= CO_NORMAL_MAX:
        co_color = "yeşil"
    elif CO_BORDERLINE_MIN <= co < CO_NORMAL_MIN or CO_NORMAL_MAX < co <= CO_BORDERLINE_MAX:
        co_color = "sarı"
    else:
        co_color = "kırmızı"

    priority = {"kırmızı": 2, "sarı": 1, "yeşil": 0}
    overall_color = max((ef_color, co_color), key=lambda c: priority[c])

    summaries = {
        "yeşil": "Kalp normal aralıkta pompalıyor.",
        "sarı": "Kalp fonksiyonunda hafif bir sapma var, izlenmeli.",
        "kırmızı": "Kalp fonksiyonunda belirgin bir sapma var (örn. düşük EF -- kalp yeterince güçlü pompalayamıyor olabilir).",
    }

    return {
        "overall_color": overall_color,
        "ef_color": ef_color,
        "co_color": co_color,
        "ef_label": f"EF %{ef:.0f}",
        "co_label": f"CO {co:.1f} L/dk",
        "summary": summaries[overall_color],
    }
