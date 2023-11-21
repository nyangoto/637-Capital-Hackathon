import pandas as pd
import numpy as np
from surprise import SVD, Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import xgboost as xgb

def clean_data(data):
    # Assuming your data needs cleaning (replace this with actual cleaning logic)
    cleaned_data = data.dropna()  # Placeholder for actual cleaning process
    return cleaned_data

def analyze_data(data):
    # Analyze user preferences, average ratings, etc.
    user_avg_ratings = data.groupby('userId')['rating'].mean()
    movie_avg_ratings = data.groupby('movieId')['rating'].mean()
        
    return user_avg_ratings, movie_avg_ratings

def recommend_movies(user_id, user_avg_ratings, movie_avg_ratings, train_sparse_matrix):
    # Your movie recommendation algorithm goes here
    # Implement robust measures to handle irregular or incomplete data
    
    # Example: Fallback to popular movies if user data is insufficient
    if user_id not in user_avg_ratings:
        popular_movies = movie_avg_ratings.nlargest(5).index
        return movie_avg_ratings.loc[popular_movies]
    
    # Your personalized recommendation logic based on user ratings
    # Example: Consider movies that the user has rated highly
    user_high_ratings = train_sparse_matrix[user_id, :].toarray().ravel() > user_avg_ratings[user_id]
    recommended_movies = movie_avg_ratings[user_high_ratings].nlargest(5)
    
    return recommended_movies

def create_cli_interface():
    # Design a simple Command Line Interface (CLI)
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
    raw_data = pd.read_csv('user_movie_ratings.csv')
    cleaned_data = clean_data(raw_data)

    # Analyze data
    user_avg_ratings, movie_avg_ratings = analyze_data(cleaned_data)

    # Create a sparse matrix
    train_sparse_matrix = csr_matrix((cleaned_data.rating.values, (cleaned_data.userId.values, cleaned_data.movieId.values)))

    # Train your recommendation model (if applicable)

    # User ID mapping for CLI
    user_name_to_id_mapping = dict(zip(cleaned_data['userName'], cleaned_data['userId']))

    # Create CLI interface
    create_cli_interface()
