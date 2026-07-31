"""
Öğrenilmiş Dinamik Model (JEPA) -- worldmodel'in İLK alt-paketi.

Bu paket, src/worldmodel/'ün geri kalanından (patient.py, pk.py, pd.py,
simulation.py) KAVRAMSAL OLARAK AYRI: onlar insan tarafından yazılmış,
veriden hiç öğrenmeyen matematiksel/fiziksel modeller (Emax/Hill,
Cockcroft-Gault benzeri deterministik formüller). Bu paket ise CircAdapt'in
(gerçek kalp mekaniği motoru) ürettiği sentetik verilerden JEPA (Joint
Embedding Predictive Architecture) mimarisiyle bir "öğrenilmiş dinamik
model" eğitir -- projenin ML/derin öğrenme anlamında GERÇEK bir öğrenilmiş
world model içeren tek parçası (patient_profile/llm_extraction.py'deki
Gemini kullanımı hariç).

Modüller:
  state_repr.py   -- hasta/trajectory verisini sabit-boyutlu vektörlere çevirir
  dataset.py       -- PyTorch Dataset/DataLoader
  model.py          -- Encoder, Predictor, TargetEncoder (EMA), DecoderHead
  train_jepa.py     -- self-supervised JEPA eğitim döngüsü
  train_decoder.py  -- embedding -> yorumlanabilir sayı (EF, CO, HR...) eğitimi
  evaluate.py       -- held-out test kümesinde CircAdapt'e karşı doğruluk raporu

Bağımlılık: bu paket `torch` gerektirir (projenin geri kalanı gerektirmez) --
requirements.txt'te ayrıca not edildi.
"""
