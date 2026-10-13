# SRNet + E-NAS

Bu paket, gönderilen kısa MATLAB eğitim koduna E-NAS eklemek için yeni yazılmış
bir uygulamadır. Makalenin sonuçlarını üreten özgün kod olduğu iddia edilmez.

MATLAB R2023b hedeflenmiştir. Burada MATLAB olmadığı için eğitim ve MATLAB
testleri çalıştırılamadı. Önce README.md içindeki `test_graphs`, `smoke_test` ve
gerçek veride `smoke` komutlarını çalıştırın; hata çıktısını paylaşın.

`enas_config.m`: ayarlar ve açık mimari katalogları.
`enas_decode.m`: 10 genin mimariye dönüştürülmesi.
`enas_graph.m`: bağlantıları ve boyut eşleştirmelerini kuran ağ üreticisi.
`srnet_enas.m`: GA araması, son eğitim, bağımsız test ve kayıtlar.

Fitness: 0.85*(1-doğrulama doğruluğu) + 0.15*(blok sayısı/160).
Tam modda 20 birey, 50 nesil, aday başına 50 epoch ve son eğitimde 500 epoch var.
Bu işlem uzun sürebilir; önce bir epoch kullanan smoke modunu çalıştırın.

Özgün kodda bulunmayan dizi katalogları ve bağlantı haritaları yeniden tanımlandı;
ayrıntıları README.md'de açıkça listelendi. Bu seçenekleri makalenin geçmişte
gerçekleştirilmiş araması gibi sunmayın. Bu paket 10-fold değil, eğitim/doğrulama/
bağımsız test ayrımı uygular. Panel bazlı gruplanmış ayrım ayrıca gereklidir.

Paylaşımdan önce: MATLAB kontrolü, gerçek veri denemesi, yazar bilgileri ve lisans
seçimi tamamlanmalı. Henüz GitHub'a veya Zenodo'ya yayımlanmadı; DOI oluşturulmadı.
