from flask import Flask, render_template
import pandas as pd
#import pickle
from recommender import get_random_movie, recommend_movies
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your data
movie_features = pd.read_pickle('data/movie_features.pkl')
movies_og = pd.read_pickle('data/movies_og.pkl')
#with open('data/similarity_matrix.pkl', 'rb') as f:
    #similarity_matrix = pickle.load(f)

X = movie_features.drop('MovieID', axis=1)
similarity_matrix = cosine_similarity(X)

@app.route('/')
def index():
    # Pick random movie
    movie_id, title = get_random_movie(movie_features, movies_og)
    recommendations = recommend_movies(movie_id, movie_features, similarity_matrix, movies_og)

    return render_template('index.html', random_title=title, recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
