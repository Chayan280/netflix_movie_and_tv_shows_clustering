"""
Ek baar chalao: raw CSV -> clustering -> netflix_clustered.csv + cluster_terms.json
Notebook ka pipeline same hai: dropna -> features -> TF-IDF -> scale -> PCA(90%) -> KMeans(k=4)
"""
import json, re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

K = 4
raw = pd.read_csv("netflix_titles.csv")
data = raw.dropna().copy()                      # notebook: data.dropna(inplace=True)
original = data.copy()                          # display ke liye original columns bacha lo

# --- date features ---
data["date_added"] = pd.to_datetime(data["date_added"].str.strip(), errors="coerce")
data = data.dropna(subset=["date_added"])
original = original.loc[data.index]
data["month_added"] = data["date_added"].dt.month
data["year_added"] = data["date_added"].dt.year

# --- type / duration / rating ---
data = pd.get_dummies(data, columns=["type"], prefix="type", dtype=int)
data["movie_duration_min"] = 0
data["tv_show_seasons"] = 0
num = data["duration"].str.extract(r"(\d+)")[0].astype(int)
data.loc[data["type_Movie"] == 1, "movie_duration_min"] = num[data["type_Movie"] == 1]
data.loc[data["type_TV Show"] == 1, "tv_show_seasons"] = num[data["type_TV Show"] == 1]
data = data.drop(columns="duration")
data = pd.get_dummies(data, columns=["rating"], prefix="rating", dtype=int)

# --- multi-label one-hot (top N) ---
def process_multi_label(df, col, top_n):
    labels = df[col].str.split(",").explode().str.strip()
    top = labels.value_counts().head(top_n).index
    for c in top:
        name = f"{col}_" + re.sub(r"\W+", "_", c)
        df[name] = df[col].apply(lambda x: int(c in [s.strip() for s in x.split(",")]))
    return df

for col, n in [("country", 20), ("listed_in", 20), ("director", 50), ("cast", 50)]:
    data = process_multi_label(data, col, n)
data = data.drop(columns=["country", "listed_in", "director", "cast"])

# --- text cleaning ---
lemm = WordNetLemmatizer()
stops = set(stopwords.words("english"))
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\b\w*\d\w*\b", "", t)
    toks = [w for w in t.split() if w not in stops]
    return " ".join(lemm.lemmatize(w) for w in toks)

data["combined_text"] = data["description"].apply(clean_text) + " " + data["title"].apply(clean_text)
tfidf = TfidfVectorizer(min_df=5, max_df=0.85, stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["combined_text"])
terms = np.array(tfidf.get_feature_names_out())

# --- feature matrix ---
drop = ["show_id", "title", "description", "date_added", "combined_text"]
feat = data.drop(columns=drop).select_dtypes(include=["number"]).astype(float).reset_index(drop=True)
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=["tfidf_" + t for t in terms])
X = pd.concat([feat, df_tfidf], axis=1)
cols_scale = ["release_year", "year_added", "month_added", "movie_duration_min", "tv_show_seasons"]
X[cols_scale] = StandardScaler().fit_transform(X[cols_scale])

# --- PCA (90% variance) + KMeans ---
pca_full = PCA().fit(X)
n_comp = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90) + 1)
pcs = PCA(n_components=n_comp, random_state=42).fit_transform(X)
km = KMeans(n_clusters=K, init="k-means++", max_iter=300, n_init=10, random_state=42)
labels = km.fit_predict(pcs)

metrics = {
    "rows": int(len(X)), "features": int(X.shape[1]), "pca_components": n_comp,
    "silhouette": float(silhouette_score(pcs, labels)),
    "calinski_harabasz": float(calinski_harabasz_score(pcs, labels)),
    "davies_bouldin": float(davies_bouldin_score(pcs, labels)),
}

# --- top TF-IDF terms per cluster (cluster ko naam dene ke liye) ---
cluster_terms = {}
tm = tfidf_matrix.tocsr()
for c in range(K):
    mean = np.asarray(tm[labels == c].mean(axis=0)).ravel()
    cluster_terms[str(c)] = terms[mean.argsort()[::-1][:10]].tolist()

# --- final output (original readable columns + cluster + 2D map coords) ---
out = original.reset_index(drop=True).copy()
out["cluster"] = labels
pc2 = PCA(n_components=2, random_state=42).fit_transform(X)
out["pc1"], out["pc2"] = pc2[:, 0], pc2[:, 1]
out["year_added"] = data["year_added"].values
out.to_csv("netflix_clustered.csv", index=False)
json.dump({"metrics": metrics, "cluster_terms": cluster_terms}, open("cluster_info.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
print(out["cluster"].value_counts().sort_index())
print(out.groupby(["cluster", "type"]).size().unstack(fill_value=0))
for c, t in cluster_terms.items():
    print(c, t)
