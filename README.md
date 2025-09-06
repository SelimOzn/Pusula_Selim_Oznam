# Pusula_Selim_Oznam
# Fiziksel Rehabilitasyon Veri Analizi ve Ön İşleme

## 1. Proje Genel Bakışı
Bu proje, fiziksel tıp ve rehabilitasyon alanındaki bir veri seti üzerinde **exploratory data analysis (EDA)** ve **ön işleme (preprocessing)** süreçlerini gerçekleştirmeyi amaçlamaktadır. Veri seti 2235 gözlem ve 13 özellik içermektedir.  

**Hedef:**  
- Tedavi süresi (`TedaviSuresi`) değişkeni üzerine odaklanarak veriyi modellemeye hazır hale getirmek.  
- Veri temizliği, eksik değerlerin doldurulması, kategorik değişkenlerin işlenmesi ve sayısal değişkenlerin standartlaştırılması gibi ön işlemleri uygulamak.  

---

## 2. Dosya Yapısı

| Dosya | Açıklama |
|-------|----------|
| `eda.py` | Veri keşfi ve görselleştirme işlemlerini yapar. Korelasyon, aykırı değer analizi, histogram, boxplot, scatter plot, Kruskal-Wallis testi, yaş ve kronik hastalık etkileri gibi analizleri içerir. |
| `preprocessing.py` | Veri temizleme ve model hazırlık sürecini içerir. Sayısal ve kategorik sütunların işlenmesi, eksik değerlerin doldurulması, kategori sayısının azaltılması ve One-Hot Encoding işlemleri burada yapılır. |

---

## 3. Kullanılan Kütüphaneler
- Pandas  
- Numpy  
- Matplotlib  
- Seaborn  
- Scikit-learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `SimpleImputer`, `OneHotEncoder`)  
- Unidecode  
- Re  

---

## 4. Kullanım Talimatları

### 4.1. EDA Çalıştırma
```bash
python eda.py
python preprocessing.py
