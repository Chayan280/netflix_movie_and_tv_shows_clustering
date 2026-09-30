import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Compatible with both older and newer Streamlit versions
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
W = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Netflix Clustering", page_icon="🎬", layout="wide")

# ---------- data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_clustered.csv")
    df["cluster"] = df["cluster"].astype(int)
    df["cluster_name"] = "Cluster " + df["cluster"].astype(str)
    return df

@st.cache_data
def load_info():
    if os.path.exists("cluster_info.json"):
        return json.load(open("cluster_info.json"))
    return {"metrics": {}, "cluster_terms": {}}

df = load_data()
info = load_info()
metrics, cluster_terms = info["metrics"], info["cluster_terms"]
order = sorted(df["cluster_name"].unique(), key=lambda s: int(s.split()[-1]))

# ---------- sidebar filters ----------
st.sidebar.title("🎬 Filters")
types = st.sidebar.multiselect("Type", sorted(df["type"].unique()), default=sorted(df["type"].unique()))
clusters = st.sidebar.multiselect("Cluster", order, default=order)
yr_min, yr_max = int(df["release_year"].min()), int(df["release_year"].max())
years = st.sidebar.slider("Release year", yr_min, yr_max, (yr_min, yr_max))
ratings = st.sidebar.multiselect("Rating", sorted(df["rating"].unique()))
search = st.sidebar.text_input("Title search")

f = df[df["type"].isin(types) & df["cluster_name"].isin(clusters)
       & df["release_year"].between(*years)]
if ratings:
    f = f[f["rating"].isin(ratings)]
if search:
    f = f[f["title"].str.contains(search, case=False, na=False)]

# ---------- header ----------
st.title("Netflix Movies & TV Shows Clustering")
st.caption("TF-IDF + metadata features → PCA (90% variance) → K-Means (K=4)")

if f.empty:
    st.warning("No titles match the current filters. Try loosening them.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Titles", f"{len(f):,}")
c2.metric("Clusters", f["cluster"].nunique())
c3.metric("Movies", f"{(f['type'] == 'Movie').sum():,}")
c4.metric("TV Shows", f"{(f['type'] == 'TV Show').sum():,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Cluster Map", "Cluster Profiles", "Explore Titles", "Model Scores"])

# ---------- Overview ----------
with tab1:
    a, b = st.columns(2)
    sz = f.groupby(["cluster_name", "type"]).size().reset_index(name="count")
    a.plotly_chart(px.bar(sz, x="cluster_name", y="count", color="type", barmode="group",
                          category_orders={"cluster_name": order},
                          title="Titles per cluster"), **W)
    b.plotly_chart(px.pie(f, names="cluster_name", hole=0.45,
                          category_orders={"cluster_name": order},
                          title="Cluster share"), **W)
    yr = f.groupby(["release_year", "cluster_name"]).size().reset_index(name="count")
    st.plotly_chart(px.line(yr, x="release_year", y="count", color="cluster_name",
                            category_orders={"cluster_name": order},
                            title="Release year trend by cluster"), **W)

# ---------- Cluster map ----------
with tab2:
    st.write("Each dot is one title. Hover to see its title, genre and rating.")
    sample = f if len(f) <= 4000 else f.sample(4000, random_state=1)
    st.plotly_chart(px.scatter(sample, x="pc1", y="pc2", color="cluster_name", symbol="type",
                               hover_name="title", hover_data=["listed_in", "rating", "release_year"],
                               category_orders={"cluster_name": order}, opacity=0.7, height=600,
                               labels={"pc1": "PC 1", "pc2": "PC 2"}),
                    **W)

# ---------- Cluster profiles ----------
with tab3:
    pick = st.selectbox("Select a cluster", order)
    cid = pick.split()[-1]
    sub = f[f["cluster_name"] == pick]
    if sub.empty:
        st.info("No titles from this cluster match the current filters.")
    else:
        l, r = st.columns(2)
        with l:
            st.subheader("Top keywords (TF-IDF)")
            if cid in cluster_terms:
                st.write(", ".join(cluster_terms[cid]))
            else:
                st.caption("cluster_info.json not found.")
            st.subheader("Quick facts")
            st.write(f"- Titles: **{len(sub):,}**")
            st.write(f"- Avg release year: **{sub['release_year'].mean():.0f}**")
            st.write(f"- Most common rating: **{sub['rating'].mode().iat[0]}**")
        with r:
            genres = sub["listed_in"].str.split(",").explode().str.strip().value_counts().head(8)
            st.plotly_chart(px.bar(genres[::-1], orientation="h", title="Top genres",
                                   labels={"value": "titles", "index": ""}),
                            **W)
        countries = sub["country"].str.split(",").explode().str.strip().value_counts().head(8)
        st.plotly_chart(px.bar(countries[::-1], orientation="h", title="Top countries",
                               labels={"value": "titles", "index": ""}),
                        **W)

# ---------- Explore titles ----------
with tab4:
    show = ["title", "type", "cluster", "listed_in", "country", "rating", "release_year"]
    st.dataframe(f[show].sort_values(["cluster", "title"]), **W, height=380)
    st.download_button("Download filtered data as CSV", f[show].to_csv(index=False), "filtered_titles.csv")
    t = st.selectbox("View title details", f["title"].sort_values().unique())
    row = f[f["title"] == t].iloc[0]
    st.markdown(f"**{row['title']}** ({row['release_year']}, {row['rating']}) → **{row['cluster_name']}**")
    st.write(row["description"])
    st.caption(f"Genres: {row['listed_in']}  |  Country: {row['country']}")

# ---------- Model scores ----------
with tab5:
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette (↑ better)", f"{metrics['silhouette']:.3f}")
        m2.metric("Calinski-Harabasz (↑ better)", f"{metrics['calinski_harabasz']:.1f}")
        m3.metric("Davies-Bouldin (↓ better)", f"{metrics['davies_bouldin']:.3f}")
        st.write(f"Rows: **{metrics['rows']:,}**, features: **{metrics['features']:,}**, "
                 f"PCA components: **{metrics['pca_components']}**")
    else:
        st.info("cluster_info.json not found; skipping model scores.")