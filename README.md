```markdown
# Netflix Movies and TV Shows Clustering

## Project Overview
This project, "Netflix Movies and TV Shows Clustering," is an end-to-end Data Science and Machine Learning initiative focused on analyzing Netflix's extensive content catalog up to the year 2019. Utilizing a dataset sourced from Flixable, a third-party Netflix search engine, the project delves into content consumption trends, regional distributions, and employs Machine Learning techniques to cluster similar titles based on their textual attributes. Historical data reveals a significant strategic shift in Netflix’s catalog, with a notable increase in TV shows and a decrease in movies since 2010.

## Table of Contents
1.  [Project Summary](#project-summary)
2.  [Key Objectives & Methodology](#key-objectives--methodology)
3.  [Data Sources](#data-sources)
4.  [Tech Stack & Tools](#tech-stack--tools)
5.  [Exploratory Data Analysis (EDA) Insights](#exploratory-data-analysis-eda-insights)
6.  [Hypothesis Testing Conclusions](#hypothesis-testing-conclusions)
7.  [Clustering Model Outcomes](#clustering-model-outcomes)
8.  [Business Impact and Recommendations](#business-impact-and-recommendations)
9.  [Future Work](#future-work)

## Project Summary
This project demonstrates how data analytics and machine learning can decode streaming industry strategies. Beyond uncovering historical catalog shifts, it offers a functional clustering solution to group similar content, backed by a production-ready interactive interface ideal for technical interviews and practical portfolio demonstrations.

## Key Objectives & Methodology
The project execution is divided into four critical phases:

1.  **Exploratory Data Analysis (EDA)**: Deep-dive analysis into content types, genre distributions, release trends, and maturity ratings to extract strategic insights.
2.  **Content Shift & Regional Analysis**:
    *   Analyzing country-specific content trends (e.g., US vs. India vs. UK).
    *   Validating the hypothesis regarding Netflix's pivot towards TV shows over movies in recent years.
3.  **Text-Based Content Clustering**: Implementing Natural Language Processing (NLP) techniques on textual features—such as plot descriptions, genres, cast, and directors—to build unsupervised clustering models. This forms the foundation for a content recommendation system.
4.  **External Dataset Integration**: Assessing how integrating third-party metrics like IMDb ratings and Rotten Tomatoes scores can add qualitative depth to content evaluation.

## Data Sources
The primary dataset was sourced from **Flixable**, a third-party Netflix search engine, covering content up to 2019. The dataset includes `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, and `description`.

## Tech Stack & Tools
The analysis and modeling pipeline rely on standard Python data science libraries:

*   **Pandas & NumPy**: For data manipulation, feature engineering, missing value treatment, and efficient numeric operations.
*   **Matplotlib & Seaborn**: For generating intuitive visualizations, exploratory charts, and correlation plots.
*   **Scikit-Learn / NLP Libraries (NLTK)**: For text preprocessing, feature extraction (TF-IDF), dimensionality reduction (PCA), and unsupervised clustering algorithms (K-Means, Agglomerative Clustering, DBSCAN).
*   **Streamlit**: For building an interactive web application interface.
*   **Gemini API**: Incorporated Large Language Model (LLM) capabilities for natural language queries and automated insights (as mentioned in the project summary, though not explicitly implemented in the provided notebook cells).

## Exploratory Data Analysis (EDA) Insights

1.  **Content Type Distribution**: Movies significantly outnumber TV shows, but there's a noticeable trend of increasing TV show additions in recent years.
2.  **Top Content-Producing Countries**: The United States is the dominant producer, followed by India and the United Kingdom.
3.  **Popular Genres**: 'Dramas' and 'International Movies' are most prevalent, alongside 'Comedies'.
4.  **Maturity Ratings**: Catalog caters to a broad audience, with a skew towards mature ratings (TV-MA, TV-14) and a substantial presence of family-friendly content.
5.  **Content Addition Trends**: Consistent growth in content additions, with peak acquisition in 2017-2019, followed by a sharp decline in 2020-2021 (potentially due to data cutoff).

## Hypothesis Testing Conclusions

1.  **Content Type Shift (Movies vs. TV Shows)**: No statistically significant difference in the *proportion* of TV shows versus movies added across different time periods (before vs. after 2017). This indicates the relative balance remained stable.
2.  **Country vs. Content Type Association**: A significant association exists between the primary content-producing country and the type of content (Movie vs. TV Show), confirming regional production variations.
3.  **Maturity Rating Distribution (Movies vs. TV Shows)**: A statistically significant difference exists in the distribution of maturity ratings between movies and TV shows, implying distinct targeting for each format.

## Clustering Model Outcomes

1.  **Data Preparation**: Textual features (`description`, `title`) were preprocessed and vectorized using TF-IDF. Categorical features (`type`, `rating`, `country`, `listed_in`, `director`, `cast`) were one-hot encoded. Numerical features were scaled. The entire feature set underwent dimensionality reduction using PCA, retaining 263 components explaining 90% of the variance.
2.  **Model Performance Comparison**:
    *   **K-Means (K=4)**: Silhouette Score: 0.112, Calinski-Harabasz Score: 581.788, Davies-Bouldin Score: 2.131. Indicated moderate cluster quality.
    *   **Agglomerative Clustering (K=4, Ward Linkage)**: Performed marginally better across metrics (Silhouette: 0.080, Calinski-Harabasz: 416.634, Davies-Bouldin: 2.371) compared to K-Means, making it the preferred model.
    *   **DBSCAN**: Ineffective with initial parameters, classifying most points as noise and finding only one cluster, highlighting its sensitivity to parameter tuning.

**Conclusion**: Agglomerative Clustering was identified as the best-performing model for content segmentation among those tested, although moderate evaluation scores suggest blending categories in Netflix's content landscape.

## Business Impact and Recommendations

*   **Content Acquisition & Production**: Insights into popular genres and regional specialization can guide strategic investment in content.
*   **Recommendation System Enhancement**: Content clusters can form a foundational layer for improving recommendation algorithms.
*   **Targeted Marketing**: Content distribution insights by country, genre, and rating allow for highly targeted marketing campaigns.

## Future Work

*   Explore more advanced NLP techniques (e.g., word embeddings with deep learning models) for richer textual representation.
*   Investigate alternative clustering algorithms (e.g., spectral clustering).
*   Integrate external metadata (e.g., IMDb ratings, audience reviews) for more distinct and interpretable content clusters.
*   Perform further hyperparameter tuning for all clustering models, especially DBSCAN, using systematic approaches.
```
