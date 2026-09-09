# Netflix Movies and TV Shows Clustering

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An exploratory data analysis and unsupervised machine learning project that studies Netflix movies and TV shows and groups similar titles using content metadata.

## Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tools and Libraries](#tools-and-libraries)
- [Workflow](#workflow)
- [Key Findings](#key-findings)
- [Clustering Results](#clustering-results)
- [Getting Started](#getting-started)
- [Future Improvements](#future-improvements)

## Project Overview

This project analyzes a Netflix catalog sourced from Flixable. It explores content trends by type, country, genre, release year, and maturity rating, then applies text processing and clustering techniques to identify groups of similar titles.

The complete analysis is available in the [Jupyter notebook](Netflix_Movies_and_TV_Shows_Clustering.ipynb).

## Objectives

- Explore the distribution and growth of Netflix movies and TV shows.
- Identify patterns in countries, genres, ratings, and release years.
- Test relationships between content type, country, and maturity rating.
- Build clusters using textual, categorical, and numerical features.
- Compare K-Means, Agglomerative Clustering, and DBSCAN.

## Dataset

The project uses the following dataset:

**File name:** [`NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv`](NETFLIX MOVIES AND TV SHOWS CLUSTERING.CSV)

The dataset was sourced from **Flixable**, a third-party Netflix search engine, and contains Netflix catalog information up to 2019.

### Dataset Features

`show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, and `description`.

The dataset contains **7,787 rows** and **12 columns**. The CSV file is not currently included in this repository; place it in the project directory before running the notebook. Update the notebook's `pd.read_csv()` path if the file is stored in another location.

## Tools and Libraries

- **Python** for analysis and modeling
- **Pandas** and **NumPy** for data preparation
- **Matplotlib** and **Seaborn** for visualization
- **NLTK** for text preprocessing
- **Scikit-learn** for TF-IDF, encoding, PCA, and clustering
- **Jupyter Notebook** for interactive development

## Workflow

1. Load and inspect the Netflix catalog.
2. Clean missing values and prepare the data.
3. Perform exploratory data analysis with visualizations.
4. Test relationships between selected catalog attributes.
5. Combine text, categorical, and numerical features.
6. Apply TF-IDF, one-hot encoding, scaling, and PCA.
7. Train and compare multiple clustering algorithms.

## Key Findings

- Movies make up most of the catalog, while TV show additions increased in later years.
- The United States is the largest content-producing country, followed by India and the United Kingdom.
- Dramas, international movies, and comedies are among the most common categories.
- `TV-MA` and `TV-14` are the most represented maturity ratings.
- Country and content type show a statistically significant association.
- Movie and TV show maturity-rating distributions differ significantly.

## Clustering Results

The project compared three unsupervised learning algorithms after feature engineering and PCA dimensionality reduction.

| Model | Configuration | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | --- | ---: | ---: | ---: |
| K-Means | 4 clusters | 0.112 | 581.788 | 2.131 |
| Agglomerative Clustering | 4 clusters, Ward linkage | 0.080 | 416.634 | 2.371 |
| DBSCAN | Initial parameters | Ineffective | Not meaningful | Not meaningful |

K-Means produced the strongest measured clustering scores in the notebook. DBSCAN classified most records as noise under the initial parameters, showing that it requires further tuning for this dataset.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook Netflix_Movies_and_TV_Shows_Clustering.ipynb
```

Run the notebook cells from top to bottom to reproduce the analysis.

## Future Improvements

- Use word embeddings or transformer-based representations for richer text features.
- Tune DBSCAN and compare additional clustering algorithms.
- Add external metadata such as IMDb ratings and audience reviews.
- Build an interactive recommendation or cluster-exploration dashboard.

## Author

Created as a data science and machine learning portfolio project.
