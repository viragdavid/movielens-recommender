import numpy as np

def get_title_from_id(movie_id, movies_og):
    title = movies_og[movies_og['MovieID'] == movie_id]['Title']
    return title.values[0] if not title.empty else None

def get_random_movie(movie_features, movies_og):
    random_row = movie_features.sample(1)
    movie_id = random_row['MovieID'].values[0]
    title = get_title_from_id(movie_id, movies_og)
    return movie_id, title

def recommend_movies(movie_id, movie_features, similarity_matrix, movies_og, top_n=5):
    movie_idx = movie_features[movie_features['MovieID'] == movie_id].index[0]
    similarity_scores = similarity_matrix[movie_idx]
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    similar_movies = movie_features.iloc[similar_indices].copy()
    similar_movies['Title'] = similar_movies['MovieID'].apply(lambda x: get_title_from_id(x, movies_og))
    return similar_movies[['MovieID', 'Title', 'Avg_Rating']]
