# Netflix Movies & TV Shows Clustering: Streamlit App

An interactive web app for exploring the results of the Netflix Movies & TV Shows clustering project.
Instead of re-running notebook cells one by one, open the app and explore the clusters with filters, charts and search.

> **Live demo:** _add your Streamlit Community Cloud link here_

## Features

| Tab | What it shows |
|---|---|
| **Overview** | Titles per cluster, cluster share, and release-year trend by cluster |
| **Cluster Map** | 2D PCA scatter plot of all titles, coloured by cluster (hover for title, genre, rating) |
| **Cluster Profiles** | Top TF-IDF keywords, top genres and top countries for a selected cluster |
| **Explore Titles** | Filterable table, per-title details, and CSV download of the filtered data |
| **Model Scores** | Silhouette, Calinski-Harabasz and Davies-Bouldin scores, plus feature/PCA summary |

Sidebar filters: content type, cluster, release year range, rating and title search.

## Quick start

```bash
cd netflix_app
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
netflix_movie_and_tv_shows_clustering/
├── Netflix_Movies_and_TV_Shows_Clustering.ipynb   # original analysis notebook
├── <raw dataset>.csv                              # original Netflix dataset
└── netflix_app/
    ├── app.py                 # Streamlit app (this is what you run)
    ├── netflix_clustered.csv  # clustered data that the app reads
    ├── cluster_info.json      # model scores + top keywords per cluster
    ├── prepare_data.py        # rebuilds the two files above from the raw CSV
    ├── requirements.txt
    └── README.md
```

## How the clustering works

The pipeline follows the notebook:

1. **Cleaning:** drop rows with missing values, parse `date_added` into year and month.
2. **Feature engineering:** one-hot encode `type` and `rating`; split `duration` into movie minutes and TV seasons;
   one-hot encode the top-N values of the multi-label columns (`country` and `listed_in`: top 20 each, `director` and `cast`: top 50 each).
3. **Text features:** clean `title` + `description` (lowercase, remove punctuation, digits and stopwords, lemmatize), then apply TF-IDF (`min_df=5`, `max_df=0.85`).
4. **Scaling:** standardize the numeric columns (release year, year added, month added, duration, seasons).
5. **Dimensionality reduction:** PCA keeping 90% of the variance.
6. **Clustering:** K-Means with K = 4 (chosen with the elbow method).

## Regenerating the data

`netflix_clustered.csv` and `cluster_info.json` are produced by `prepare_data.py`:

```bash
python prepare_data.py path/to/raw_dataset.csv
```

If you omit the path it looks for `netflix_titles.csv` in the current folder.
The script needs `scikit-learn` and `nltk` (both are in `requirements.txt`) and downloads the NLTK `stopwords` and `wordnet` data on first run.

### Using the exact labels from your notebook

`prepare_data.py` re-implements the notebook pipeline, so cluster sizes and scores can differ slightly from a given notebook run.
To show the notebook's own labels, run this at the end of the notebook and replace `netflix_clustered.csv`:

```python
out = original_data.loc[data_with_clusters.index].copy()   # original readable columns
out["cluster"] = data_with_clusters["cluster_label"].values
out["pc1"] = pca_feature_matrix.iloc[:, 0].values
out["pc2"] = pca_feature_matrix.iloc[:, 1].values
out.to_csv("netflix_clustered.csv", index=False)
```

The app expects these columns: `title`, `type`, `cluster`, `listed_in`, `country`, `rating`, `release_year`, `description`, `pc1`, `pc2`.
If you use the snippet above, also make sure `cluster_info.json` matches the new labels
(the Model Scores and Cluster Profiles tabs read from it), or delete it and those sections will show a notice instead.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repository and branch `main`.
4. Set **Main file path** to `netflix_app/app.py` and deploy.

## Known limitations

- **Missing values:** the notebook drops every row with any missing value, which removes most TV shows
  (about 135 of 2,410 remain), because TV shows often have no director listed. As a result, three of the four clusters contain only movies.
  Imputing `director`, `cast` and `country` with `"Unknown"` instead of dropping rows would keep far more data.
- **Cluster quality:** silhouette scores are low (around 0.04 to 0.12), which means the clusters overlap heavily.
  Treat them as broad groupings, not sharp categories.
- **Overlapping keywords:** the top keywords of different clusters are similar (`life`, `young`, `love`, `family`),
  so cluster names are not assigned automatically.

## Tech stack

Python, pandas, scikit-learn, NLTK, Plotly, Streamlit