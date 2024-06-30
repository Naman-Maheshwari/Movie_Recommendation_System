import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load the dataset
file_path = 'IMDB-Movie-Dataset(2023-1951).csv'
movies_df = pd.read_csv(file_path)

# Remove the unnecessary index column
movies_df.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values in the 'year' column by filling with a placeholder
movies_df['year'].fillna('Unknown', inplace=True)

# Function to convert year to numeric, handling 'Unknown'
def convert_to_numeric(year):
    try:
        return int(year)
    except ValueError:
        return np.nan

# Convert 'year' column to numeric values, handling 'Unknown'
movies_df['year'] = movies_df['year'].apply(convert_to_numeric)

# Function to get valid years for slider
def get_valid_years(df):
    valid_years = df[df['year'].notnull()]['year']
    if valid_years.empty:
        return None
    return (int(valid_years.min()), int(valid_years.max()))

# Combine the relevant columns into a single feature
movies_df['features'] = movies_df['movie_name'] + ' ' + movies_df['genre'] + ' ' + movies_df['overview']

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the features
X = count_vectorizer.fit_transform(movies_df['features'])

# Initialize the KNN model
knn = NearestNeighbors(n_neighbors=11, metric='cosine')

# Fit the model to the data
knn.fit(X)

# Function to get movie recommendations using KNN
def get_recommendations(movie_name, year_range, selected_genre, knn=knn, X=X):
    try:
        # Get the index of the movie that matches the title
        idx = movies_df[movies_df['movie_name'] == movie_name].index[0]
        
        # Get the distances and indices of the 10 most similar movies
        distances, indices = knn.kneighbors(X[idx], n_neighbors=11)
        
        # Get the movie indices
        movie_indices = indices.squeeze().tolist()[1:]
        
        # Filter based on selected year range and genre
        recommended_movies = movies_df.iloc[movie_indices]
        if year_range[0] != movies_df['year'].min() or year_range[1] != movies_df['year'].max():
            recommended_movies = recommended_movies[
                (recommended_movies['year'] >= year_range[0]) & 
                (recommended_movies['year'] <= year_range[1])
            ]
        if selected_genre != 'All':
            recommended_movies = recommended_movies[movies_df['genre'].str.contains(selected_genre, case=False)]
        
        if recommended_movies.empty:
            return "No recommendations found for the selected filters."
        
        # Sort the recommended movies based on year
        recommended_movies = recommended_movies.sort_values('year', ascending=False)
        
        # Format 'year' column to remove decimals and convert to integer
        recommended_movies['year'] = recommended_movies['year'].apply(lambda x: str(int(x)) if not pd.isnull(x) else 'Unknown')
        
        # Select only desired columns
        recommended_movies = recommended_movies[['movie_name', 'year', 'genre']]
        
        return recommended_movies
    except IndexError:
        return "Movie not found."

# Streamlit app
def main():
    st.set_page_config(page_title="Movie Recommendation System", layout="wide")
    st.title("Movie Recommendation System")
    
    # Sidebar layout
    st.sidebar.title("Filters")
    
    # Dropdown to select a movie name
    movie_name = st.sidebar.selectbox("Select a Movie", movies_df['movie_name'].sort_values().unique())
    
    # Get valid years for slider
    valid_years = get_valid_years(movies_df)
    if valid_years:
        min_year, max_year = valid_years
        year_range = st.sidebar.slider("Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    else:
        st.sidebar.warning("No valid years found in the dataset.")
        return
    
    # Dropdown to select a genre
    genre_options = ['All'] + sorted(set(genre for sublist in movies_df['genre'].str.split(', ') for genre in sublist))
    selected_genre = st.sidebar.selectbox("Select Genre", genre_options)
    
    if st.sidebar.button("Get Recommendations"):
        recommendations = get_recommendations(movie_name, year_range, selected_genre, knn=knn, X=X)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.header("Top 10 Movie Recommendations:")
            st.table(recommendations.set_index('year'))

if __name__ == "__main__":
    main()