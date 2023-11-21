import pandas as pd
import numpy as np
from surprise import SVD, Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import xgboost as xgb

def clean_data(data):
    cleaned_data = data.dropna()
    return cleaned_data

def analyze_data(data):
    
    user_avg_ratings = data.groupby('userId')['rating'].mean()
    movie_avg_ratings = data.groupby('movieId')['rating'].mean()
        
    return user_avg_ratings, movie_avg_ratings

def recommend_movies(user_id, user_avg_ratings, movie_avg_ratings, train_sparse_matrix):
    # movie recommendation algorithm
   
    if user_id not in user_avg_ratings:
        popular_movies = movie_avg_ratings.nlargest(5).index
        return movie_avg_ratings.loc[popular_movies]
    
    # personalized recommendation logic
    
    user_high_ratings = train_sparse_matrix[user_id, :].toarray().ravel() > user_avg_ratings[user_id]
    recommended_movies = movie_avg_ratings[user_high_ratings].nlargest(3)
    
    return recommended_movies

def create_cli_interface():
    while True:
        user_name = input("Enter your name (or 'exit' to quit): ")
        
        if user_name.lower() == 'exit':
            break

        user_id = user_name_to_id_mapping.get(user_name)
        if user_id is not None:
            recommendations = recommend_movies(user_id, user_avg_ratings, movie_avg_ratings, train_sparse_matrix)
            print(f"\nHello {user_name}! Here are your personalized movie recommendations:")
            print(recommendations)
        else:
            print("User not found. Please try again.")

if __name__ == "__main__":
    # Load and clean data
    raw_data = pd.read_csv('data.txt', header=None, names=['User', 'Movie', 'Rating'])
    cleaned_data = clean_data(raw_data)

    # Analyze data
    user_avg_ratings, movie_avg_ratings = analyze_data(cleaned_data)

    # Create a sparse matrix
    train_sparse_matrix = csr_matrix((cleaned_data.rating.values, (cleaned_data.userId.values, cleaned_data.movieId.values)))

    # Model training and shaping

    # User ID mapping for CLI
    user_name_to_id_mapping = dict(zip(cleaned_data['userName'], cleaned_data['userId']))

    # Create CLI interface
    create_cli_interface()
