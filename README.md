# 🏠 Istanbul Airbnb Price Analysis & Price Prediction

## 📌 Proje Hakkında

Bu proje, **Istanbul Airbnb ilanlarının fiyatlarını etkileyen faktörleri analiz etmek** ve çeşitli özellikleri kullanarak **gecelik fiyat tahmini** yapmak amacıyla geliştirilmiştir.

Proje kapsamında veri temizleme, keşifsel veri analizi (EDA), özellik mühendisliği ve makine öğrenmesi adımları uygulanmıştır.

Çalışmada Airbnb ilanlarının **konum, oda tipi, yorumlar, minimum konaklama süresi, müsaitlik ve diğer özellikleri** incelenerek fiyat üzerindeki etkileri araştırılmıştır.

---

## 🎯 Projenin Amacı

* Istanbul Airbnb piyasasını veri üzerinden incelemek
* Fiyatları etkileyen temel faktörleri belirlemek
* Farklı semt ve ilan özelliklerinin fiyatlara etkisini analiz etmek
* Airbnb fiyatlarını tahmin edebilen bir makine öğrenmesi modeli geliştirmek
* Elde edilen sonuçları kullanıcı, ev sahibi ve yatırımcı açısından yorumlamak

---

## 📊 Veri Seti

Projede Istanbul Airbnb ilanlarından oluşan bir veri seti kullanılmıştır.

Veri setinde yaklaşık **30.000 ilan** bulunmaktadır.

Başlıca değişkenler:

* `neighbourhood` – Mahalle
* `room_type` – Oda tipi
* `price` – Gecelik fiyat
* `minimum_nights` – Minimum konaklama süresi
* `number_of_reviews` – Yorum sayısı
* `reviews_per_month` – Aylık yorum sayısı
* `availability_365` – Yıllık müsaitlik
* `calculated_host_listings_count` – Ev sahibinin ilan sayısı
* `last_review` – Son yorum tarihi
* `license` – Lisans bilgisi

---

## 🧹 Veri Ön İşleme

Modelleme öncesinde veri seti üzerinde aşağıdaki işlemler gerçekleştirilmiştir:

* Gereksiz değişkenlerin kaldırılması
* Eksik değerlerin incelenmesi ve uygun şekilde ele alınması
* `price` değişkenindeki geçersiz değerlerin filtrelenmesi
* `reviews_per_month` eksik değerlerinin değerlendirilmesi
* `last_review` değişkeninden yeni bir `has_review` özelliğinin oluşturulması
* Kategorik değişkenlerin modele uygun hale getirilmesi
* Aykırı değerlerin incelenmesi

---

## 🔎 Keşifsel Veri Analizi (EDA)

Analiz aşamasında aşağıdaki konular incelenmiştir:

* Istanbul'daki Airbnb ilanlarının fiyat dağılımı
* Mahallelere göre fiyat farklılıkları
* Oda tiplerinin fiyat üzerindeki etkisi
* Yorum sayısı ve fiyat arasındaki ilişki
* Müsaitlik ve fiyat ilişkisi
* İlan yoğunluğunun bölgelere göre dağılımı
* Farklı bölgelerdeki fiyat farklılıkları

Özellikle **Sarıyer, Sancaktepe ve diğer bölgeler arasındaki fiyat farklılıkları** incelenerek lokasyonun fiyat üzerindeki etkisi araştırılmıştır.

---

## 🧠 Feature Engineering

Model performansını ve veri analizini geliştirmek amacıyla yeni değişkenler oluşturulmuştur.

Örneğin:

```text
has_review
```

İlanın daha önce yorum alıp almadığını belirleyen binary bir değişkendir.

Ayrıca kategorik değişkenler modelleme öncesinde uygun encoding yöntemleriyle dönüştürülmüştür.

---

## 🤖 Makine Öğrenmesi

Fiyat tahmini için **Random Forest Regressor** kullanılmıştır.

Model geliştirme sürecinde:

* Train / Test ayrımı
* Feature Engineering
* Model eğitimi
* RandomizedSearchCV ile hiperparametre optimizasyonu
* Tahminlerin değerlendirilmesi

adımları uygulanmıştır.

### Kullanılan değerlendirme metrikleri

* **MSE (Mean Squared Error)**
* **R² Score**

Model performansı bu metrikler kullanılarak değerlendirilmiştir.

---

## 🛠️ Kullanılan Teknolojiler

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook / PyCharm
Git & GitHub
```

---

## 📈 Proje Çıktıları

Proje sonucunda:

* Istanbul Airbnb piyasasının genel yapısı analiz edilmiştir.
* Lokasyonun fiyatlar üzerindeki etkisi incelenmiştir.
* Farklı oda tiplerinin fiyat farklılıkları ortaya çıkarılmıştır.
* İlan özellikleri ile fiyat arasındaki ilişkiler araştırılmıştır.
* Makine öğrenmesi kullanılarak fiyat tahmin modeli oluşturulmuştur.

Bu analizler, **seyahat planlaması, bütçe belirleme ve Airbnb ev sahiplerinin fiyatlandırma kararları** açısından kullanılabilecek içgörüler sunmaktadır.

---

## 🚀 Gelecek Geliştirmeler

Projenin sonraki aşamalarında:

* Daha gelişmiş makine öğrenmesi modellerinin denenmesi
* XGBoost / LightGBM gibi modellerle karşılaştırma
* Model performansının iyileştirilmesi
* Feature Importance ve SHAP analizi
* Streamlit kullanılarak interaktif bir fiyat tahmin uygulaması geliştirilmesi
* Kullanıcının konum ve konaklama özelliklerini girerek tahmini fiyat alabileceği bir arayüz oluşturulması

planlanmaktadır.

---

## 

GitHub: [@aleynabass002-lab](https://github.com/aleynabass002-lab)
