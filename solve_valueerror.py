import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD
# from nltk.corpus import stopwords

# Assuming df contains the full dataset for training
engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')
select = "select * from movie_details"
df = pd.read_sql(select, engine)

label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre']) 
df['type'] = label_encoder.fit_transform(df['type'])

x = df[['genre', 'year', 'rating', 'votes', 'type', 'meta_score']]
y = df[['movie_name']]

# Preprocess and transform data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(x)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['detail_story']) 
n_components = 100  # Specify the number of dimensions you want to reduce to
svd = TruncatedSVD(n_components=n_components)
df_tfidf = svd.fit_transform(tfidf_matrix) 
components = svd.components_


feature_names = tfidf_vectorizer.get_feature_names_out()

for i, component in enumerate(components):
    top_features_idx = component.argsort()[-6:][::-1] 
print(feature_names[top_features_idx])# Get indices of top 5 
print("df_tfidf.shape", df_tfidf.shape)


# model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
# model.fit(X_pca, y)


def preprocess_user_input(user_input):
    user_input = user_input.lower()
    user_input = user_input.split() 
    user_input = [word for word in user_input if word.isalpha()]
    user_input = ' '.join(user_input)
    
    return user_input

def recommend_movie(user_input_feature):
    # Step 1: Preprocess user input (convert to lowercase, remove non-alphabetic words, etc.)
    user_input_feature = preprocess_user_input(user_input_feature)
    # print(f"Processed user input: {user_input_feature}")  # Debugging step
    
    # Step 2: Transform the preprocessed input into TF-IDF features
    user_input_tfidf = tfidf_vectorizer.transform([user_input_feature]).toarray()
    # print(f"User input TF-IDF vector: {user_input_tfidf}")  # Debugging step

    # Convert user input to DataFrame to ensure all columns are present
    user_input_tfidf_df = pd.DataFrame(user_input_tfidf, columns=feature_names)
    
    # Step 3: Add missing TF-IDF features if any
    missing_features = set(feature_names) - set(user_input_tfidf_df.columns)
    for feature in missing_features:
        user_input_tfidf_df[feature] = 0  # Add missing features with default value (0)

    # Ensure the order of columns is correct (only TF-IDF features)
    user_input_tfidf_df = user_input_tfidf_df[feature_names]
    
    # Step 4: Add metadata columns if missing
    metadata_columns = ['genre', 'year', 'rating', 'votes', 'type', 'meta_score']
    for feature in metadata_columns:
        if feature not in user_input_tfidf_df.columns:
            user_input_tfidf_df[feature] = 0  # Add missing metadata columns with default value (0)
    print(df[metadata_columns].shape)
    # Step 5: Ensure the correct order of columns: TF-IDF features + metadata columns
    for i, component in enumerate(components):
        top_features_idx = component.argsort()[-6:][::-1] 
        print(feature_names[top_features_idx])# Get indices of top 5 
        print("df_tfidf.shape", df_tfidf.shape)
        user_input_tfidf_df = user_input_tfidf_df[feature_names[top_features_idx] + metadata_columns]

    # Debugging: Print the DataFrame shape and columns to ensure everything is aligned
    print(f"Shape of user_input_tfidf_df: {user_input_tfidf_df.shape}")
    print(f"Columns of user_input_tfidf_df: {user_input_tfidf_df.columns}")

    # Step 6: Impute missing values (if any)
    user_input_feature_imputed = imputer.transform(user_input_tfidf_df)

    # Step 7: Apply scaling and PCA (same as during training)
    user_input_feature_scaled = scaler.transform(user_input_feature_imputed)
    user_input_feature_pca = pca.transform(user_input_feature_scaled)

    # Step 8: Get the indices of the closest neighbors
    distances, indices = model.kneighbors(user_input_feature_pca, n_neighbors=5)

    # Debugging: Print indices and distances
    print(f"indices: {indices}")
    print(f"distances: {distances}")

    # Step 9: Map the indices to movie names
    recommended_movie_names = df.iloc[indices[0]]['movie_name'].values

    return recommended_movie_names

# Example of user input
user_input_feature = input("Enter a movie description: ")
recommended_movie_names = recommend_movie(user_input_feature)

# Print the recommended movie titles
print("Recommended Movies:", recommended_movie_names)






