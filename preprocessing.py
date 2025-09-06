import pandas as pd
import re
import unidecode
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocessing():
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
    numeric_cols = df.select_dtypes(include='number').columns.difference(["HastaNo"])
    string_cols = df.select_dtypes(include='object').columns.difference(["TedaviAdi"])
    agg_dict = {col: 'mean' for col in numeric_cols}
    agg_dict.update({col: lambda x: x.mode()[0] if not x.mode().empty else None for col in string_cols})  # string için en çok geçen

    df_reduced = df.groupby(["HastaNo", "TedaviAdi"], as_index=False).agg(agg_dict)

    #Preprocessing
    print(df_reduced.info())
    groups = df_reduced["HastaNo"]
    #ID ve EDA sonucu etkisiz özellikleri silme
    df_reduced.drop(["HastaNo", "Cinsiyet", "KanGrubu", "Uyruk", "TedaviSuresi", "UygulamaSuresi"], axis=1, inplace=True)

    # Eksik string değerlerini doldurma
    df_reduced["KronikHastalik"] = df_reduced["KronikHastalik"].fillna("Bilinmiyor")
    df_reduced["Alerji"] = df_reduced["Alerji"].fillna("Bilinmiyor")

    counts_tedavi = df_reduced["TedaviAdi"].value_counts()
    rare_tedavi = counts_tedavi[counts_tedavi == 1].index

    df_reduced.loc[(df_reduced["TedaviAdi"].isin(rare_tedavi)) &
                   (df_reduced["TedaviSuresi_num"] == 15), "TedaviAdi"] = "Diğer"

    # Tanilar için
    counts_tanilar = df_reduced["Tanilar"].value_counts()
    rare_tanilar = counts_tanilar[counts_tanilar == 1].index

    df_reduced.loc[(df_reduced["Tanilar"].isin(rare_tanilar)) &
                   (df_reduced["TedaviSuresi_num"] == 15), "Tanilar"] = "Diğer"

    # Metin temizliği: Tanilar, UygulamaYerleri, Bolum için de uygulama
    for col in ["Tanilar", "UygulamaYerleri", "Bolum", "Alerji", "KronikHastalik"]:
        if col in df_reduced.columns:
            df_reduced[col] = df_reduced[col].apply(clean_text)

    #  Eksikleri aynı tedaviyi alan hasta grubunun modu ile doldurma
    def fillna_by_group_mode(df, group_col, target_cols):
        for c in target_cols:
            if c not in df.columns:
                continue
            # grup modu
            mode_per_group = (
                df.groupby(group_col)[c]
                  .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else "Bilinmiyor")
                # Grubtaki tüm değerler nan ise "Bilinmiyor"
            )
            # her satıra grubunun modunu kullan
            df[c] = df[c].fillna(df[group_col].map(mode_per_group))
        return df

    df_reduced = fillna_by_group_mode(
        df_reduced,
        group_col="TedaviAdi",
        target_cols=["Tanilar", "UygulamaYerleri", "Bolum"]
    )

    # Tedavi dışındaki metin alanlarında genel doldurma
    for c in ["Alerji", "KronikHastalik"]:
        if c in df_reduced.columns:
            df_reduced[c] = df_reduced[c].fillna("Bilinmiyor")


    # Hedefte boş değer varsa o satırın silinmesi
    df_reduced = df_reduced[~df_reduced["TedaviSuresi_num"].isna()].copy()

    num_cols = [c for c in ["Yas", "KronikHastalikCount", "UygulamaSuresi_num"] if c in df_reduced.columns]
    cat_cols = [c for c in ["TedaviAdi", "Tanilar", "UygulamaYerleri", "Bolum", "Alerji", "KronikHastalik"] if c in df_reduced.columns]

    X = df_reduced[num_cols + cat_cols].copy()
    y = df_reduced["TedaviSuresi_num"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None  # regresyonda stratify yok
    )

    # Preprocessing: sayısal = median impute + scale, kategorik = most_frequent impute + OHE
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)

if __name__ == "__main__":
    preprocessing()