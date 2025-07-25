# app/app.py

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from recommender import get_random_movie, recommend_movies

# Load data
movie_features = pd.read_pickle('app/data/movie_features.pkl')
movies_og = pd.read_pickle('app/data/movies_og.pkl')

X = movie_features.drop('MovieID', axis=1)
similarity_matrix = cosine_similarity(X)

# UI
st.title("Movie Recommender")

if st.button("Get Recommendations"):
    movie_id, title = get_random_movie(movie_features, movies_og)
    recommendations = recommend_movies(movie_id, movie_features, similarity_matrix, movies_og, top_n=10)
    
    st.subheader(f"Because you liked **{title}**")
    for row in recommendations.itertuples():
        st.markdown(f"- **{row.Title}** (Rating: {row.Avg_Rating:.2f}) — *{row.Genres}*")



