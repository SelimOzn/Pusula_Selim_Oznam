import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr,kruskal, mannwhitneyu
import textwrap
import re
import unidecode


df = pd.read_excel("Talent_Academy_Case_DT_2025.xlsx")
print(df.describe())
print(df.info())

def clean_text(x):
    if pd.isna(x):
        return x
    x = x.lower()
    x = re.sub(r"\s+", " ", x)
    x = unidecode.unidecode(x)  # Türkçe karakterleri normalize etme (İ ve i dönüşümü için)
    x = x.strip()
    return x

def remove_side_words(x):
    if pd.isna(x):
        return x
    # "sağ" ve "sol" kelimelerini silme
    x = re.sub(r"\b(sağ|sol)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

df["TedaviAdi"] = df["TedaviAdi"].apply(clean_text)
df["TedaviAdi"] = df["TedaviAdi"].apply(remove_side_words)

df["TedaviSuresi_num"] = df["TedaviSuresi"].str.extract(r"(\d+)").astype(float)
df["UygulamaSuresi_num"] = df["UygulamaSuresi"].str.extract(r"(\d+)").astype(float)
df["KronikHastalikCount"] = (
    df["KronikHastalik"]
    .fillna("")
    .apply(lambda x: len([h.strip() for h in x.split(",") if h.strip()!=""]))
)

# Hasta numarası, tedavi adı, tedavi süresi aynı olup tanısı farklı olan kayıtları inceleme
mask = df.groupby(["HastaNo", "TedaviAdi", "TedaviSuresi_num"])["Tanilar"].transform("nunique") > 0
df_conflict = df[mask].drop_duplicates(subset=["HastaNo", "TedaviAdi", "TedaviSuresi_num", "Tanilar"])

# Sayısal ve string sütunlara göre farklı agg fonksiyonu
numeric_cols = df.select_dtypes(include='number').columns.difference(["HastaNo","TedaviSuresi_num"])
string_cols = df.select_dtypes(include='object').columns.difference(["TedaviAdi"])
agg_dict = {col: 'mean' for col in numeric_cols}
agg_dict.update({col: lambda x: x.mode()[0] if not x.mode().empty else None for col in string_cols})  # string için en çok geçen

df_reduced = df.groupby(["TedaviAdi", "TedaviSuresi_num", "HastaNo"], as_index=False).agg(agg_dict)

print(df_reduced.describe())
print(df_reduced.info())

plt.figure(figsize=(8,6))
corr = df_reduced[["Yas","UygulamaSuresi_num","KronikHastalikCount","TedaviSuresi_num"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# Aykırı değer tespiti fonksiyonu
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (series < lower) | (series > upper)
    return mask

# Sayısal sütunlar için aykırı değer yüzdeleri hesaplanıyor
for col in ["Yas", "TedaviSuresi_num", "UygulamaSuresi_num"]:
    mask = detect_outliers_iqr(df_reduced[col].dropna())
    oran = mask.mean() * 100
    print(f"{col}: %{oran:.2f} aykırı değer var")



# Yeterli gözleme sahip tedavilerin seçilmesi
tedavi_counts = df_reduced["TedaviAdi"].value_counts()
valid_tedaviler = tedavi_counts[tedavi_counts > 10].index
df_valid = df_reduced[df_reduced["TedaviAdi"].isin(valid_tedaviler)]

# Karşılaştırma amaçlı tedavilere göre ortalama ve std değerleri
group_stats = df_valid.groupby("TedaviAdi").agg({
    "TedaviSuresi_num": ["mean", "std"],
    "UygulamaSuresi_num": "mean",
    "Yas": "mean",
    "KronikHastalikCount": "mean"
}).reset_index()

group_stats.columns = [
    "TedaviAdi", "TedaviSuresi_num_mean", "TedaviSuresi_num_std",
    "UygulamaSuresi_mean", "Yas_mean", "KronikHastalik_mean"
]

# Orijinal veriye ortalamaları merge et
df_valid = df_valid.merge(group_stats, on="TedaviAdi", how="left")

# Anomali metrikleri
df_valid["diff"] = (df_valid["TedaviSuresi_num"] - df_valid["TedaviSuresi_num_mean"]).abs()
df_valid["ratio"] = df_valid["TedaviSuresi_num"] / df_valid["TedaviSuresi_num_mean"]
df_valid["zscore"] = (df_valid["TedaviSuresi_num"] - df_valid["TedaviSuresi_num_mean"]) / df_valid["TedaviSuresi_num_std"]

# Olası anomali adayları
df_anomalies = df_valid[(df_valid["zscore"].abs() > 3) | (df_valid["ratio"] > 2)]


#Yüksek seviyede anomali yüzdesine sahip sütun için boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df_reduced["TedaviSuresi_num"])
plt.title("Tedavi Süresi - Aykırı Değerler (Boxplot)")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df_reduced["TedaviSuresi_num"], kde=True, bins=30)
plt.title("Tedavi Süresi Dağılımı (Histogram + KDE)")
plt.xlabel("Tedavi Süresi")
plt.ylabel("Frekans")
plt.show()

counts = df_reduced.value_counts("TedaviSuresi_num")
per_15 = counts[15]/len(df_reduced)

df_reduced["Bolum"] = df_reduced["Bolum"].fillna("Bilinmiyor")

print("Kruskal-Wallis: Bölüm")
groups = [g["TedaviSuresi_num"].values for _, g in df_reduced.groupby("Bolum")]
stat, p = kruskal(*groups)
print(f"K-W Bölüm: stat={stat:.3f}, p={p:.4f}")

n_total = len(df_reduced)
k_groups = df_reduced["Bolum"].nunique()
epsilon_sq = (stat - (k_groups - 1)) / (n_total - 1)

print("Bölüm Epsilon squared (ε²): %.6f" % epsilon_sq)

bolum_ort = df_reduced.groupby("Bolum")["TedaviSuresi_num"].mean().reset_index()

plt.figure(figsize=(8,5))
plt.bar(x=bolum_ort["Bolum"], height=bolum_ort["TedaviSuresi_num"], width=0.8,  align="center")
labels = [textwrap.fill(label, 15) for label in bolum_ort["Bolum"].unique()]
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.ylabel("Tedavi Süresi")
plt.title("Bölümlere Göre Tedavi Süresi")
plt.tight_layout()
plt.show()

print(df_reduced.groupby("KronikHastalikCount")["TedaviSuresi_num"].mean().reset_index())

# Virgülle ayrılmış kronik hastalıkları sütunlara dönüştür
df_hastalik = df_reduced["KronikHastalik"].str.get_dummies(sep=",")

# Orijinal dataframe ile birleştir
df_expanded = pd.concat([df_reduced, df_hastalik], axis=1)

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof)
    return (x.mean() - y.mean()) / pooled_std


etki_sonuclar = []

for hastalik in df_hastalik.columns:
    grup0 = df_expanded[df_expanded[hastalik] == 0]["TedaviSuresi_num"]
    grup1 = df_expanded[df_expanded[hastalik] == 1]["TedaviSuresi_num"]

    if len(grup1) > 5:
        u, p = mannwhitneyu(grup0, grup1, alternative="two-sided")
        d = cohens_d(grup1, grup0)  # işaret: pozitifse sürede artış
        etki_sonuclar.append((hastalik, p, d, grup0.mean(), grup1.mean()))

etki_df = pd.DataFrame(etki_sonuclar, columns=["Hastalik", "p", "Cohen_d", "mean_yok", "mean_var"])
etki_df["Anlamlı mı"] = etki_df["p"] < 0.05

# p-değeri küçük ve |d| büyük olanları sırala
etki_df = etki_df.iloc[etki_df["Cohen_d"].abs().argsort()[::-1]]

top5 = etki_df[etki_df["Anlamlı mı"]].head(5)

plt.figure(figsize=(8,5))
sns.barplot(
    data=top5.melt(id_vars=["Hastalik"], value_vars=["mean_yok","mean_var"]),
    x="Hastalik", y="value", hue="variable"
)
plt.title("Tedavi Süresini En Çok Etkileyen İlk 5 Kronik Hastalık")
plt.ylabel("Ortalama Tedavi Süresi")
plt.xlabel("Kronik Hastalık")
wrapped_labels = [ "\n".join(textwrap.wrap(label, width=15)) for label in top5["Hastalik"] ]
plt.xticks(ticks=range(len(wrapped_labels)), labels=wrapped_labels, rotation=45)
plt.legend(title="Hastalık Durumu", labels=["Yok","Var"])
plt.tight_layout()
plt.show()

# Yaşa göre scatterplot
plt.figure(figsize=(7,5))
plt.scatter(df_reduced["Yas"], df_reduced["TedaviSuresi_num"], alpha=0.5)
plt.xlabel("Yaş")
plt.ylabel("Tedavi Süresi (gün)")
plt.title("Yaşa göre Tedavi Süresi")
plt.show()


# Yaş grupları tanımla
bins = [0, 18, 30, 50, 60, 70, 120]
labels = ["Çocuk (0-18)", "Genç (19-30)", "Orta yaş (30-50)", "50-60", "60-70", "70+"]
df_reduced["YasGrup"] = pd.cut(df_reduced["Yas"], bins=bins, labels=labels, right=True)

# Ortalama tedavi süresi
yas_ort = df_reduced.groupby("YasGrup")["TedaviSuresi_num"].mean().reset_index()
print(yas_ort)

# Görselleştirme
plt.figure(figsize=(8,5))
sns.boxplot(x="YasGrup", y="TedaviSuresi_num", data=df_reduced, order=labels)
plt.title("Yaş Gruplarına Göre Tedavi Süresi")
plt.xlabel("Yaş Grubu")
plt.ylabel("Tedavi Süresi (seans)")
plt.show()


# Cinsiyete göre boxplot
plt.figure(figsize=(7,5))
df_reduced.boxplot(column="TedaviSuresi_num", by="Cinsiyet")
plt.title("Cinsiyete göre Tedavi Süresi")
plt.suptitle("")
plt.ylabel("Tedavi Süresi (gün)")
plt.show()

plt.figure(figsize=(8,5))
df_reduced.boxplot(column="TedaviSuresi_num", by="Bolum")
plt.title("Bölümlere göre Tedavi Süresi")
plt.suptitle("")
plt.ylabel("Tedavi Süresi (gün)")
labels = [textwrap.fill(label.get_text(), 15) for label in plt.gca().get_xticklabels()]
plt.gca().set_xticklabels(labels, rotation=45)
plt.tight_layout()
plt.show()

# Kan grubu bazında tedavi süresi
plt.figure(figsize=(8,4))
sns.boxplot(x="KanGrubu", y="TedaviSuresi_num", data=df_reduced)
plt.title("Kan Grubuna Göre Tedavi Süresi")
plt.ylabel("Tedavi Süresi (seans)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(df_reduced.value_counts("KanGrubu"))


plt.figure(figsize=(7,5))
sns.violinplot(x="KronikHastalikCount", y="TedaviSuresi_num", data=df_reduced, inner="box")
plt.title("Kronik Hastalık Sayısına Göre Tedavi Süresi Dağılımı")
plt.ylabel("Tedavi Süresi (seans)")
plt.show()

df_expanded = (
    df.dropna(subset=["KronikHastalik"])
      .assign(KronikHastalik=df["KronikHastalik"].str.split(","))
      .explode("KronikHastalik")
)

df_expanded2 = (
    df.assign(KronikHastalik=df["KronikHastalik"].fillna("Bilinmiyor").str.split(","))
      .explode("KronikHastalik")
)


uyruklar = ['Tokelau', 'Azerbaycan', 'Libya', 'Arnavutluk']
# Türk ve yabancı uyruk olarak kategoriler oluşturuldu
df["UyrukGrup"] = df["Uyruk"].apply(lambda x: "Yabancı" if x in uyruklar else ("Türk" if x=="Türkiye" else "Diğer"))

# Tedavi türü ve uyruk grubuna göre ortalama tedavi süresi
result = (
    df[df["UyrukGrup"].isin(["Türk", "Yabancı"])]
    .groupby(["TedaviAdi", "UyrukGrup"])["TedaviSuresi_num"]
    .mean()
    .reset_index()
    .pivot(index="TedaviAdi", columns="UyrukGrup", values="TedaviSuresi_num")
    .dropna(subset=["Yabancı", "Türk"]) # Sadece Türklere uygulanan tedavileri kaldırma
)

labels = [textwrap.fill(label, 15) for label in result.index]
ax = result.plot(kind="bar", figsize=(12,6))
ax.set_title("Tedavi Adına Göre Ortalama Tedavi Süresi (Türk vs Yabancı)")
ax.set_xlabel("Tedavi Adı")
ax.set_ylabel("Ortalama Tedavi Süresi (gün)")
ax.set_xticklabels(labels, rotation=45, ha="right")
plt.legend(title="Uyruk Grubu")
plt.tight_layout()
plt.show()

# Her tedavi için ortalama uygulama süresi ve tedavi süresi grafiği
plt.scatter(df_reduced["UygulamaSuresi_num"], df_reduced["TedaviSuresi_num"], alpha=0.6)
plt.xlabel("Ortalama Uygulama Süresi (dakika)")
plt.ylabel("Planlanan Tedavi Süresi (seans)")
plt.title("Uygulama Süresi vs Tedavi Süresi")
plt.show()

pearson_corr, pearson_p = pearsonr(df_reduced["UygulamaSuresi_num"], df_reduced["TedaviSuresi_num"])
spearman_corr, spearman_p = spearmanr(df_reduced["UygulamaSuresi_num"], df_reduced["TedaviSuresi_num"])

print(f"Pearson korelasyonu: {pearson_corr:.3f} (p={pearson_p:.3f})")
print(f"Spearman korelasyonu: {spearman_corr:.3f} (p={spearman_p:.3f})")


# NaN olanları doldurma
df_reduced["UygulamaYerleri"] = df_reduced["UygulamaYerleri"].fillna("Bilinmiyor")

# Virgülle ayrılmış uygulama yerlerini ayır
df_exploded = df_reduced.copy()
df_exploded["UygulamaYerleri"] = df_exploded["UygulamaYerleri"].str.split(",")
df_exploded = df_exploded.explode("UygulamaYerleri")

# Fazladan boşluk varsa temizle
df_exploded["UygulamaYerleri"] = df_exploded["UygulamaYerleri"].str.strip()
uygulama_summary = df_exploded.groupby("UygulamaYerleri").agg(
    OrtalamaTedaviSuresi=("TedaviSuresi_num", "mean"),
    KayitSayisi=("TedaviSuresi_num", "count")
).reset_index()
# Ortalama süreye göre sıralayalım
uygulama_summary = uygulama_summary.sort_values("OrtalamaTedaviSuresi")

plt.figure(figsize=(10,6))
plt.bar(uygulama_summary["UygulamaYerleri"], uygulama_summary["OrtalamaTedaviSuresi"], alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Uygulama Yerleri")
plt.ylabel("Ortalama Planlanan Tedavi Süresi (seans)")
plt.title("Uygulama Yerlerine Göre Ortalama Tedavi Süresi")
plt.tight_layout()
plt.show()

