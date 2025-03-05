import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')
select = "select * from movie_details"
df = pd.read_sql(select, engine)
# df = pd.read_csv('movie_data.csv')
imputer = SimpleImputer(strategy='mean') 
df['rating'] = imputer.fit_transform(df[['rating']])

# One-Hot Encoding for the 'genre' column
df = pd.get_dummies(df, columns=['genre'])


scaler = StandardScaler()
df['rating_scaled'] = scaler.fit_transform(df[['rating']])

# Example if you have a 'detail_story' column
vectorizer = TfidfVectorizer(max_features=5000)
text_features = vectorizer.fit_transform(df['detail_story'])



# Prepare the movie features (e.g., genre, rating, etc.)
movie_features = df[['rating_scaled', 'votes']]  # Add more features if needed

# Train KNN model to find similar movies
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(movie_features)

# Find 5 most similar movies to a given movie (example: movie_id 1)
similar_movies = knn.kneighbors([movie_features.iloc[0].values], n_neighbors=6)
print(similar_movies)
 


# Assuming y_true is the actual ratings and y_pred is the predicted ratings
y_true = df['rating']
y_pred = df['rating']
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)

