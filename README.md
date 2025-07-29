# MovieLens M1 Recommender System & Rating Prediction Models

This project explores the MovieLens M1 dataset through extensive analysis, data preparation, machine learning rating prediction, and movie recommendation system development. It includes:

- Exploratory Data Analysis (EDA)
- Data preparation & feature engineering
- Rating prediction using boosting models
- Content-based recommendation systems
- A deployed Streamlit web app

---

## Exploratory Data Analysis (EDA)

The EDA was conducted in the `eda.ipynb` notebook and focused on:

- Understanding structure and data types of the MovieLens M1 dataset
- Visualizing value distributions, counts, and top features
- Exploring trends and averages (e.g., by genres, years)
- Identifying useful vs. irrelevant features (e.g., ZIP codes were converted to U.S. states using external mappings)
- Auto-generated profiling using [`ydata-profiling`](https://github.com/ydataai/ydata-profiling) (HTML reports included)

---

## Data Preparation

Found in `data-preparation.ipynb`:

- MovieLens data required minimal cleaning (already clean, no missing values or obvious outliers)
- Scaled numerical features and encoded categorical ones
- Created **two prepared datasets**:
  - **Content-based dataset**: `ratings + movies`
  - **Hybrid dataset**: `ratings + movies + users`
- Feature engineering:
  - Removed irrelevant columns
  - Added aggregate features like average ratings, number of ratings per user/movie, and user state

---

## Rating Prediction Models

Found in `hybrid-rating-prediction.ipynb`:

- Used the **hybrid dataset** (ratings + users + movies)
- Two experimental setups:
  1. Traditional train-validation-test split
  2. 5-fold cross-validation

- Algorithms used:
  - XGBoost
  - LightGBM
  - CatBoost (performed best in both setups)

### Best Model

- **CatBoost + 5-Fold Cross-Validation**
- **RMSE: 0.8746**
  
>  **RMSE** (Root Mean Square Error) measures the difference between predicted and actual ratings. Lower RMSE means better performance.

---

## Recommendation Systems

Built using the **content-based dataset**, in `recommendation-models.ipynb`.

### Cosine Similarity Recommender

- Constructed a **movie-item matrix**
- Measured similarity between movies using **cosine similarity**
- Given a movie, recommends 10 most similar ones
- Produced **plausible** and relevant results

### TF-IDF Variant (Tried but worse)

- Experimented with enriching the item matrix using **TF-IDF** (Term Frequency-Inverse Document Frequency), a common text-based weighting technique
- Resulted in **worse** recommendations, so not used in final version — but included for experimentation completeness

>  **TF-IDF** is a statistical method used to evaluate how important a word is to a document relative to a collection (corpus). It was tested here to weigh genre and textual features differently, but didn’t outperform simpler similarity.

---

## Deployed Web App

A simple web app built with **Streamlit**, deployed at:

🔗 [https://movielens-recommender-random-movie-10-similar.streamlit.app/](https://movielens-recommender-random-movie-10-similar.streamlit.app/)

### Features:

- Picks a random movie from the dataset
- Recommends top 10 similar movies using the cosine similarity model
- Basic UI, but functional and fast for demo purposes

---

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `xgboost`, `lightgbm`, `catboost`
- `scikit-learn`
- `streamlit`
- `ydata-profiling`

---

## Summary

This project shows a full pipeline from EDA to deployment for a recommender system. It demonstrates:

- Real-world dataset handling
- Model comparison for regression (rating prediction)
- Content-based filtering techniques
- Deployment of a working prototype
