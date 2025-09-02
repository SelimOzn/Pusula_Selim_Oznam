import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Talent_Academy_Case_DT_2025.xlsx")
print(df.describe())
print(df.info())

df["TedaviSuresi_num"] = df["TedaviSuresi"].str.extract(r"(\d+)").astype(float)

# 1. Yaşa göre scatterplot
plt.figure(figsize=(7,5))
plt.scatter(df["Yas"], df["TedaviSuresi_num"], alpha=0.5)
plt.xlabel("Yaş")
plt.ylabel("Tedavi Süresi (gün)")
plt.title("Yaşa göre Tedavi Süresi")
plt.show()

# 2. Cinsiyete göre boxplot
plt.figure(figsize=(7,5))
df.boxplot(column="TedaviSuresi_num", by="Cinsiyet")
plt.title("Cinsiyete göre Tedavi Süresi")
plt.suptitle("")
plt.ylabel("Tedavi Süresi (gün)")
plt.show()

# 3. Bölümlere göre boxplot
plt.figure(figsize=(8,5))
df.boxplot(column="TedaviSuresi_num", by="Bolum")
plt.title("Bölümlere göre Tedavi Süresi")
plt.suptitle("")
plt.ylabel("Tedavi Süresi (gün)")
plt.xticks(rotation=45)
plt.show()

# 4. Kronik hastalık durumuna göre boxplot
plt.figure(figsize=(7,5))
df.boxplot(column="TedaviSuresi_num", by="KronikHastalik")
plt.title("Kronik Hastalık Durumuna göre Tedavi Süresi")
plt.suptitle("")
plt.ylabel("Tedavi Süresi (gün)")
plt.show()


df_expanded = (
    df.dropna(subset=["KronikHastalik"])                # NaN değerleri çıkar
      .assign(KronikHastalik=df["KronikHastalik"].str.split(","))  # liste yap
      .explode("KronikHastalik")                        # her hastalık ayrı satır
)

# Fazladan boşlukları temizle
df_expanded["KronikHastalik"] = df_expanded["KronikHastalik"].str.strip()

# Hastalık bazında tedavi süresini göster (boxplot)
plt.figure(figsize=(10,6))
df_expanded.boxplot(column="TedaviSuresi_num", by="KronikHastalik")
plt.title("Kronik Hastalıklara göre Tedavi Süresi")
plt.suptitle("")
plt.ylabel("Tedavi Süresi (gün)")
plt.xticks(rotation=45)
plt.show()