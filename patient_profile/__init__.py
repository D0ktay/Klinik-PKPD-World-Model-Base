"""
Hasta Profili LLM Extraction modülü.

Amaç: kullanıcının yüklediği hasta dosyalarından (PDF -- muayene raporu,
laboratuvar sonucu, epikriz vb.) yapılandırılmış hasta verisi çıkarmak,
bu veriyi doğrulamak, ve mevcut PK/PD + CircAdapt katmanına (bkz.
src/worldmodel/patient.py > Patient, Drug) hastaya-özel parametre olarak
beslemek.

MİMARİ PRENSİP: LLM burada karar verici değil, veri çıkarıcıdır. Hiçbir
klerens/doz/klinik yargı LLM tarafından üretilmez -- tüm hesaplamalar
(Cockcroft-Gault, Child-Pugh, allometrik ölçekleme) covariate_mapping.py
içindeki deterministik Python fonksiyonlarıyla yapılır.

Akış: file_ingestion -> llm_extraction -> temporal_merge -> validation
      -> (kullanıcı onayı, review_data) -> covariate_mapping
      -> src.worldmodel.patient.Patient / mevcut PK parametreleri
"""
