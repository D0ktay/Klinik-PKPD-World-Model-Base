"""
JEPA Mimarisi (Faz 3)
========================

Üç parça:

  Encoder    -- durum vektörünü (bkz. state_repr.STATE_DIM) düşük boyutlu bir
                "embedding"e (soyut temsil) sıkıştırır. Gradyanla eğitilir.
  Predictor  -- [embedding, aksiyon] alıp bir FARK (delta/residual) tahmin
                eder -- "sonraki embedding" DEĞİL, "embedding NE KADAR
                değişecek". Gerçek sonraki embedding, `predict_next_embedding()`
                yardımcı fonksiyonuyla `embedding + delta` olarak hesaplanır
                (bkz. o fonksiyonun docstring'i -- bu tasarım kararının
                gerekçesi: zayıf/gürültülü sinyalde doğrudan-tahmin modelleri
                "ortalamaya kaçma" eğiliminde, delta-tahmini bu sorunu
                yapısal olarak önlüyor).
  TargetEncoder -- Encoder ile AYNI mimari, ama ağırlıkları backprop ile
                DEĞİL, online Encoder'ın ağırlıklarının hareketli ortalamasıyla
                (EMA -- exponential moving average) güncellenir ve bu dala
                gradyan HİÇ akmaz (stop-gradient). Bu, JEPA'yı JEPA yapan asıl
                mekanizma: eğer hedefi üreten ağ da serbestçe eğitilseydi,
                model her durumu aynı sabit embedding'e eşleyip kaybı sıfırlar
                ("collapse" / çökme) ama hiçbir şey öğrenmezdi. EMA + stop-
                gradient bu kısayolu engeller.

  DecoderHead -- (Faz 4'te train_decoder.py tarafından ayrıca eğitilir)
                embedding'i geri, insanın okuyabileceği fizyolojik sayılara
                (EF, CO, HR, EDV, ESV) çevirir.
"""

import torch
import torch.nn as nn

from worldmodel.learned_dynamics.state_repr import STATE_DIM, ACTION_DIM, SCALAR_TARGET_FIELDS

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 128


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class Encoder(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, embedding_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Predictor(nn.Module):
    """forward()'ın çıktısı bir DELTA'dır (sonraki embedding DEĞİL) --
    bkz. predict_next_embedding()."""

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.net = _mlp(embedding_dim + action_dim, hidden_dim, embedding_dim)

    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([embedding, action], dim=-1))


def predict_next_embedding(predictor: Predictor, embedding: torch.Tensor,
                            action: torch.Tensor) -> torch.Tensor:
    """
    TEK doğru kaynak: `Predictor.forward()`'ın DELTA (fark) döndürdüğünü,
    gerçek "sonraki embedding"in `embedding + delta` olduğunu HER çağıran
    kod (train_jepa.py, evaluate.py, rollout_evaluate.py) burada, TEK
    yerden alır -- aksi halde bir dosyada residual eklenip başka birinde
    unutulursa (tutarsız yorumlama), eğitim ve çıkarım birbirini
    YANLIŞ anlardı. Bu fonksiyon o riski yapısal olarak ortadan kaldırıyor.

    NEDEN delta (doğrudan "sonraki embedding" tahmini yerine): Aksiyon
    zayıf/etkisi küçükken (örn. ilaç konsantrasyonu düşükken), MSE kaybı
    minimize eden bir "doğrudan tahmin" modeli en güvenli seçenek olarak
    eğitim kümesinin ORTALAMA sonraki-embedding'ine yakınsar -- hastadan
    bağımsız SABİT bir noktaya çekilir (gözlemlenen "collapse" belirtisi).
    Delta-tahmininde ise aynı durumda en güvenli/kolay çözüm delta≈0 --
    yani "durum değişmedi" -- bu FİZİKSEL OLARAK DOĞRU davranışla örtüşüyor
    (zayıf aksiyon = küçük değişim), modelin rastgele bir ortalamaya
    savrulmasını yapısal olarak engelliyor.
    """
    return embedding + predictor(embedding, action)


class DecoderHead(nn.Module):
    """embedding -> [EF, CO, HR, EDV, ESV] (normalize edilmiş, sıra
    SCALAR_TARGET_FIELDS ile birebir aynı -- bkz. train_decoder.py)."""

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.net = _mlp(embedding_dim, hidden_dim, len(SCALAR_TARGET_FIELDS))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


@torch.no_grad()
def update_target_encoder(target_encoder: Encoder, online_encoder: Encoder, momentum: float) -> None:
    """
    target_ağırlık <- momentum * target_ağırlık + (1 - momentum) * online_ağırlık

    momentum tipik olarak 1'e yakın (örn. 0.99) -- target encoder online
    encoder'ı YAVAŞÇA takip eder, bu yavaşlık collapse'e karşı ikinci bir
    güvence (hedef, tahmin edilenle aynı anda değişmiyor).
    """
    for target_param, online_param in zip(target_encoder.parameters(), online_encoder.parameters()):
        target_param.data.mul_(momentum).add_(online_param.data, alpha=1 - momentum)


def get_device() -> torch.device:
    """GPU varsa kullanır, yoksa CPU'da çalışmaya devam eder -- hiçbir yerde
    GPU ZORUNLULUĞU yok (bkz. plan Context: donanım masaüstü, GPU durumu
    belirsiz)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
