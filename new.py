import pandas as pd
from sqlalchemy import create_engine

# Create an engine and connect to the database
engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')

# Fetch movie details from the database
query = "SELECT * FROM movie_details"
df = pd.read_sql(query, engine)

# Check the first few rows to see the structure of the DataFrame
print(df.head())


from sklearn.feature_extraction.text import TfidfVectorizer

# Apply TF-IDF Vectorizer to the 'detail_story' column (assuming it's the description column)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust 'max_features' as needed
movie_tfidf_matrix = tfidf_vectorizer.fit_transform(df['detail_story'])

# Get the feature names (words from the descriptions)
feature_names = tfidf_vectorizer.get_feature_names_out()


from sklearn.preprocessing import LabelEncoder, StandardScaler

# Apply Label Encoding to categorical columns (e.g., genre and type)
label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre'])
df['type'] = label_encoder.fit_transform(df['type'])

# Select numeric columns and apply scaling
numeric_columns = ['year', 'rating', 'votes', 'meta_score']  # Add any numeric columns here
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_columns])

# Combine the TF-IDF matrix and scaled numeric features into a single matrix
from scipy.sparse import hstack

combined_matrix = hstack([movie_tfidf_matrix, df_scaled])


def recommend_movie(user_input_feature):
    # Preprocess the user input (description)
    user_input_tfidf = tfidf_vectorizer.transform([user_input_feature])
    
    # Get user input metadata (e.g., genre, year, etc.). Assuming you have it as well.
    user_input_metadata = [0, 2022, 7.5, 1200, 1, 80]  # Example, you will need real user input here
    
    # Scale the metadata (make sure it's the same columns as the training data)
    user_input_metadata_scaled = scaler.transform([user_input_metadata])
    
    # Combine the user input TF-IDF and metadata
    user_input_combined = hstack([user_input_tfidf, user_input_metadata_scaled])
    
    # Use a Nearest Neighbors model or other algorithm for recommendation
    from sklearn.neighbors import NearestNeighbors
    model = NearestNeighbors(n_neighbors=5, metric='cosine')
    
    # Fit the model on the combined movie data
    model.fit(combined_matrix)
    
    # Find the nearest neighbors (recommended movies)
    distances, indices = model.kneighbors(user_input_combined)
    
    # Get the names of the recommended movies
    recommended_movies = df.iloc[indices[0]]['movie_name']
    
    return recommended_movies

# Example of user input
user_input_feature = "action thriller"
recommended_movie_names = recommend_movie(user_input_feature)

# Display the recommended movie names
print("Recommended Movies:")
print(recommended_movie_names)
