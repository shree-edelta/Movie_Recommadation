# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression  # or any other model for classification
# from sqlalchemy import create_engine
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import confusion_matrix

# engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')
# select = "select * from movie_details"
# df = pd.read_sql(select, engine)

# label_encoder = LabelEncoder()
# df['genre'] = label_encoder.fit_transform(df['genre']) 
# df['type'] = label_encoder.fit_transform(df['type'])

# tfidf_vectorizer = TfidfVectorizer(max_features=1000) 
# text_features = tfidf_vectorizer.fit_transform(df['detail_story']).toarray()

# # Add the TF-IDF features to the DataFrame
# df_tfidf = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out())

# # Concatenate the text features with the other features (such as genre, rating, etc.)
# df = pd.concat([df.drop(columns=['detail_story']), df_tfidf], axis=1)


# features = ['genre', 'year', 'rating', 'votes', 'type', 'meta_score']+ list(df_tfidf.columns)


# X = df[features]
# y = df['movie_name']
  
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)


# pca = PCA(n_components=3)  
# X_pca = pca.fit_transform(X_scaled)


# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = model.score(X_test, y_test)
# print(f"Accuracy: {accuracy}")
# print(f"Number of Principal Components: {pca.n_components_}")
# print(f"confusion matrix",confusion_matrix(y_test, y_pred))
# print(f"Explained Variance by Principal Components: {pca.explained_variance_ratio_}")


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine
import numpy as np

# Load data from PostgreSQL database
engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')
select = "SELECT * FROM movie_details"
df = pd.read_sql(select, engine)

# Encode categorical features like 'genre' and 'type'
label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre']) 
df['type'] = label_encoder.fit_transform(df['type'])

# Preprocess the text feature using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
text_features = tfidf_vectorizer.fit_transform(df['detail_story']).toarray()

# Add the TF-IDF features to the DataFrame
df_tfidf = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate the text features with other features
df = pd.concat([df.drop(columns=['detail_story']), df_tfidf], axis=1)

# Define the features for model input
features = ['genre', 'year', 'rating', 'votes', 'type', 'meta_score'] + list(df_tfidf.columns)

X = df[features]
y = df['movie_name']

# Handle missing data by imputing missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale the features for better performance in distance-based models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply PCA for dimensionality reduction (optional step for large datasets)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize the KNN model
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_train)

# Function to process user input and make a recommendation
def recommend_movie(user_input_feature):
    # Preprocess user input: Assume user input is a description or keyword
    user_input_feature = user_input_feature.lower()
    user_input_feature = user_input_feature.split()
    user_input_feature = [word for word in user_input_feature if word.isalpha()]
    user_input_feature = ' '.join(user_input_feature)
    
    # Transform the user input to match the feature space (same as training)
    user_input_tfidf = tfidf_vectorizer.transform([user_input_feature]).toarray()
    user_input_tfidf_df = pd.DataFrame(user_input_tfidf, columns=tfidf_vectorizer.get_feature_names_out())
    
    # Reorder columns to match the training dataset (important for matching feature names)
    user_input_tfidf_df = user_input_tfidf_df.reindex(columns=df.columns[6:], fill_value=0)
    
    # Combine the user input with metadata features (genre, year, etc.)
    user_input_metadata = {
        'genre': 0,  # Default value for genre (set to a value, or ask user)
        'year': 2020,  # Default value for year (set a reasonable default or prompt the user)
        'rating': 5.0,  # Default value for rating
        'votes': 1000,  # Default value for votes
        'type': 0,  # Default value for type
        'meta_score': 50  # Default value for meta_score
    }
    
    # Convert user input metadata into a DataFrame
    user_input_metadata_df = pd.DataFrame([user_input_metadata])
    
    # Concatenate the TF-IDF data and metadata
    user_input_combined = pd.concat([user_input_metadata_df, user_input_tfidf_df], axis=1)

    # Impute missing values and scale the input to match training data format
    user_input_imputed = imputer.transform(user_input_combined)
    user_input_scaled = scaler.transform(user_input_imputed)

    # Apply PCA to reduce dimensionality to match training data
    user_input_pca = pca.transform(user_input_scaled)

    # Use KNN model to find the nearest neighbors
    distances, indices = knn.kneighbors(user_input_pca)

    # Get the recommended movie names based on the indices
    recommended_movies = y_train.iloc[indices[0]].values
    return recommended_movies

# Example usage: ask for user input
user_input_feature = input("Enter a movie description or keywords for recommendation: ")
recommended_movie = recommend_movie(user_input_feature)

# Display the recommended movies
print("Recommended Movies:")
for movie in recommended_movie:
    print(movie)
