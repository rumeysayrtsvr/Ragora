# Nicel Ön Retrieval Değerlendirmesi

Bu değerlendirme, mevcut MongoDB ve Qdrant indeksleri üzerinde hazırlanan küçük bir etiketli test kümesiyle yapılmıştır. Metin tarafında 20 Türkçe sorgu; görsel tarafta 16 indeksli kaynak görselinden üretilen hafif kırpılmış/yeniden sıkıştırılmış sorgu görseli kullanılmıştır. Metin sorgularında doğru kabul edilen sonuçlar kategori, kaynak URL parçası ve beklenen kanıt terimleriyle; görsel sorgularda ise beklenen görsel hash'i, kategori ve kaynak/başlık eşleşmesiyle işaretlenmiştir.

| Arama yöntemi | Soru sayısı | Recall@5 | MRR@5 | Precision@5 |
| --- | --- | --- | --- | --- |
| Semantik arama (Qdrant) | 20 | 1.00 | 1.00 | 0.71 |
| Anahtar kelime araması (MongoDB) | 20 | 1.00 | 0.97 | 0.81 |
| Hibrit RRF | 20 | 1.00 | 1.00 | 0.84 |

| Görsel arama yöntemi | Görsel sayısı | Hit@5 | MRR@5 | Precision@5 |
| --- | --- | --- | --- | --- |
| CLIP image-to-image | 16 | 0.88 | 0.88 | 0.39 |

Ön sonuçlar, metin tarafında hibrit RRF yaklaşımının semantik arama ve anahtar kelime aramasının güçlü yönlerini birleştirdiğini göstermektedir. Semantik arama, doğal dille sorulan açıklayıcı sorgularda kategori düzeyinde güçlü sonuçlar üretirken; MongoDB text search özellikle ürün adı, parça adı, hata/özellik terimi veya tablo değeri içeren sorgularda tamamlayıcı sinyal sağlamıştır. Görsel tarafta CLIP tabanlı image-to-image arama, indeksli görsellerden oluşturulan varyantlarda yüksek isabet üretmiş; bununla birlikte bu sonuçlar aynı kaynak koleksiyonundan türetilen görseller üzerinde ölçüldüğü için açık dünya fotoğraflarıyla ayrıca genişletilmelidir.
