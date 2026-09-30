# Netflix Clustering App

## Run karne ke steps
    pip install -r requirements.txt
    streamlit run app.py

## Files
- app.py                 : Streamlit web app (presentation ke time yahi chalana hai)
- netflix_clustered.csv  : clustered data (app yahi padhta hai)
- cluster_info.json      : scores + cluster keywords
- prepare_data.py        : raw CSV se clustering dobara chalane ke liye (optional)
- netflix_titles.csv     : raw dataset

## Apne notebook ke exact labels use karne hain?
Notebook ke end mein ye chalao aur CSV app folder mein daal do:
    out = original_data.loc[data_with_clusters.index].copy()   # original readable columns
    out["cluster"] = data_with_clusters["cluster_label"].values
    out["pc1"] = pca_feature_matrix.iloc[:, 0].values
    out["pc2"] = pca_feature_matrix.iloc[:, 1].values
    out.to_csv("netflix_clustered.csv", index=False)
