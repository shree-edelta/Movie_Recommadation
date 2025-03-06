# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sqlalchemy import create_engine
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import confusion_matrix

# # Connect to the PostgreSQL database and fetch data
# engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')
# select = "select * from movie_details"
# df = pd.read_sql(select, engine)

# # Encode categorical columns
# label_encoder = LabelEncoder()
# df['genre'] = label_encoder.fit_transform(df['genre']) 
# df['type'] = label_encoder.fit_transform(df['type'])

# # Apply TF-IDF vectorizer to 'detail_story' column
# tfidf_vectorizer = TfidfVectorizer(max_features=1000) 
# text_features = tfidf_vectorizer.fit_transform(df['detail_story']).toarray()

# # Add the TF-IDF features to the DataFrame
# df_tfidf = pd.DataFrame(text_features, columns=tfidf_vectorizer.get_feature_names_out())

# # Concatenate the text features with the other features (genre, rating, etc.)
# df = pd.concat([df.drop(columns=['detail_story']), df_tfidf], axis=1)

# # Features and target variable
# features = ['genre', 'year', 'rating', 'votes', 'type', 'meta_score'] + list(df_tfidf.columns)
# X = df[features]
# y = df['movie_name']

# # Impute missing values
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)

# # Apply PCA to reduce dimensionality
# pca = PCA(n_components=3)  
# X_pca = pca.fit_transform(X_scaled)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# User input recommendation function
# def recommend_movie(user_input_feature):
#     # Preprocess user input in the same way as training data
#     user_input_feature = user_input_feature.lower()  # Make input lowercase
#     user_input_feature = user_input_feature.split()  # Split into words
#     user_input_feature = [word for word in user_input_feature if word.isalpha()]  # Remove non-alphabetic words
#     user_input_feature = ' '.join(user_input_feature)  # Join back into a string
    
#     # Transform the user input with the same TF-IDF vectorizer
#     user_input_tfidf = tfidf_vectorizer.transform([user_input_feature]).toarray()
    
#     # Convert the user input into a DataFrame with the same column names as training data
#     user_input_tfidf_df = pd.DataFrame(user_input_tfidf, columns=tfidf_vectorizer.get_feature_names_out())
    
#     # Add missing features to match the training data structure
#     missing_features = [col for col in df_tfidf.columns if col not in user_input_tfidf_df.columns]
#     for feature in missing_features:
#         user_input_tfidf_df[feature] = 0  # Add missing features with a default value of 0
    
#     # Reorder the columns to match the training data's order
#     user_input_tfidf_df = user_input_tfidf_df[df_tfidf.columns]
    
#     # Apply imputation and scaling, just as you did with training data
#     user_input_feature_imputed = imputer.transform(user_input_tfidf_df)  # Impute missing values
#     user_input_feature_scaled = scaler.transform(user_input_feature_imputed)  # Scale the input
    
#     # Apply PCA transformation
#     user_input_feature_pca = pca.transform(user_input_feature_scaled)
    
#     # Make a prediction using the trained model
#     prediction = model.predict(user_input_feature_pca)
    
#     return prediction

# # Sample user input
# user_input_feature = input("Search for movie: ")
# recommended_movie = recommend_movie(user_input_feature)
# print("Recommended Movie:", recommended_movie)

# def recommend_movie(user_input_feature):
#     # Preprocess user input in the same way as training data
#     user_input_feature = user_input_feature.lower()  # Make input lowercase
#     user_input_feature = user_input_feature.split()  # Split into words
#     user_input_feature = [word for word in user_input_feature if word.isalpha()]  # Remove non-alphabetic words
#     user_input_feature = ' '.join(user_input_feature)  # Join back into a string
    
#     # Transform the user input with the same TF-IDF vectorizer
#     user_input_tfidf = tfidf_vectorizer.transform([user_input_feature]).toarray()
    
#     # Convert the user input into a DataFrame with the same column names as training data
#     user_input_tfidf_df = pd.DataFrame(user_input_tfidf, columns=tfidf_vectorizer.get_feature_names_out())
    
#     # Add missing features to match the training data structure
#     missing_features = [col for col in df_tfidf.columns if col not in user_input_tfidf_df.columns]
#     for feature in missing_features:
#         user_input_tfidf_df[feature] = 0  # Add missing features with a default value of 0
    
#     # Add other necessary columns (genre, rating, etc.) if they are missing
#     for feature in ['genre', 'year', 'rating', 'votes', 'type', 'meta_score']:
#         if feature not in user_input_tfidf_df.columns:
#             user_input_tfidf_df[feature] = 0  # You can set them to a default value (e.g., 0, or NaN)
    
#     # Reorder the columns to match the training data's order (excluding 'movie_name' column)
#     input_features = ['genre', 'year', 'rating', 'votes', 'type', 'meta_score'] + list(df_tfidf.columns)
#     user_input_tfidf_df = user_input_tfidf_df[input_features]  # Ensures order of columns matches

#     # Apply imputation and scaling, just as you did with training data
#     user_input_feature_imputed = imputer.transform(user_input_tfidf_df)  # Impute missing values
#     user_input_feature_scaled = scaler.transform(user_input_feature_imputed)  # Scale the input
    
#     # Apply PCA transformation
#     user_input_feature_pca = pca.transform(user_input_feature_scaled)
    
#     # Make a prediction using the trained model
#     prediction = model.predict(user_input_feature_pca)
    
#     return prediction

# # Sample user input
# user_input_feature = input("Search for movie: ")
# recommended_movie = recommend_movie(user_input_feature)
# print("Recommended Movie:", recommended_movie)

# ``````````````````````````````````````````````````````````````````````

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Assuming df contains the full dataset for training
engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')
select = "select * from movie_details"
df = pd.read_sql(select, engine)

# Replace this with your actual dataset load
label_encoder = LabelEncoder()
df['genre'] = label_encoder.fit_transform(df['genre']) 
df['type'] = label_encoder.fit_transform(df['type'])

# tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# df_tfidf = tfidf_vectorizer.fit_transform(df['movie_name'])
# df_tfidf = pd.DataFrame(df_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# df_tfidf['genre'] = df['genre']
# df_tfidf['year'] = df['year']
# df_tfidf['rating'] = df['rating']
# df_tfidf['votes'] = df['votes']
# df_tfidf['type'] = df['type']
# df_tfidf['meta_score'] = df['meta_score']

# df_tfidf.to_csv('movie_details_tfidf.csv', index=False)


# Example, adjust with your dataset file

# Assuming you have the necessary trained models and preprocessors loaded

# knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', algorithm='auto')


# These should be saved from your model training

# missing_features = [col for col in df_tfidf.columns if col not in user_input_tfidf_df.columns]
# for feature in missing_features:
#     user_input_tfidf_df[feature] = 0
x = df[['genre', 'year', 'rating', 'votes', 'type', 'meta_score']]
y = df[['movie_name']]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(x)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


# imputer = SimpleImputer(strategy='mean')  # Example strategy
# scaler = StandardScaler()
# pca = PCA(n_components=5)  # Adjust PCA components based on your model training
#  # TF-IDF vectorizer used during training
tfidf_vectorizer = TfidfVectorizer() 
tokenizer = tfidf_vectorizer.build_tokenizer() 
tokens = df['detail_story'].apply(tokenizer)
flat_tokens = [token for sublist in tokens for token in sublist]
model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
# model = NearestNeighbors(n_neighbors=5)  # KNN or other model you trained
# knn.fit(X_pca, y)
# Train on the dataset
df_tfidf = tfidf_vectorizer.fit_transform(df['detail_story']) 
feature_names = tfidf_vectorizer.get_feature_names_out()
# Adjust column name
# imputer.fit(df[['genre', 'year', 'rating', 'votes', 'type', 'meta_score']])  # Fit imputer
# scaler.fit(df[['genre', 'year', 'rating', 'votes', 'type', 'meta_score']])  # Fit scaler
# pca.fit(df[['genre', 'year', 'rating', 'votes', 'type', 'meta_score']])  # Fit PCA
# model.fit(df[['genre', 'year', 'rating', 'votes', 'type', 'meta_score']])  # Fit model
model.fit(X_pca, y)

# Function to handle user input and make predictions
def recommend_movie(user_input_feature):
    # Step 1: Preprocess user input
    
    # Example user input: user_input_feature = "action thriller"
    user_input_feature = user_input_feature.lower()  # Normalize to lowercase
    user_input_feature = user_input_feature.split()  # Split into words
    user_input_feature = [word for word in user_input_feature if word.isalpha()]  # Remove non-alphabetic words
    user_input_feature = ' '.join(user_input_feature)  # Join words back into string
    
    # Step 2: Transform the user input with the same TF-IDF vectorizer used during training
    user_input_tfidf = tfidf_vectorizer.transform([user_input_feature]).toarray()

    # Convert user input to DataFrame to ensure all columns are present
    user_input_tfidf_df = pd.DataFrame(user_input_tfidf, columns=feature_names)
    
    
    # Add missing TF-IDF features to the user input DataFrame
    missing_tfidf_features = [feature for feature in flat_tokens if feature not in user_input_tfidf_df.columns]
    for feature in missing_tfidf_features:
        user_input_tfidf_df[feature] = 0  # Set missing features to 0
    
    # Add missing metadata features (like 'genre', 'rating', etc.)
    for feature in ['genre', 'year', 'rating', 'votes', 'type', 'meta_score']:
        if feature not in user_input_tfidf_df.columns:
            user_input_tfidf_df[feature] = 0  # Add default value (e.g., 0 or NaN)
    
    # Step 3: Ensure column order matches the model's training data (exclude 'movie_name')
    input_features = ['genre', 'year', 'rating', 'votes', 'type', 'meta_score'] + list(flat_tokens)
    user_input_tfidf_df = user_input_tfidf_df[input_features]  # Reorder columns
    x_columns = user_input_tfidf_df.columns
    # Step 4: Apply imputation, scaling, and PCA transformation
   
    
    common_columns = x_columns.intersection(user_input_tfidf_df.columns) #assuming user_input fir imuted has all columns
    user_input_tfidf_df = user_input_tfidf_df[common_columns]
    missing_columns = set(x_columns) - set(user_input_tfidf_df.columns)
    for col in missing_columns:
        user_input_tfidf_df[col] = np.nan
        
    user_input_feature_imputed = imputer.transform(user_input_tfidf_df)
        
    # Impute missing values
    user_input_feature_scaled = scaler.transform(user_input_feature_imputed)  # Scale the input data
    user_input_feature_pca = pca.transform(user_input_feature_scaled)  # Apply PCA
    
    # Step 5: Use the trained KNN model to recommend similar movies
    prediction = model.kneighbors(user_input_feature_pca, n_neighbors=5)  # Find 5 closest neighbors

    return prediction

# Example of user input
user_input_feature = input("Enter a movie description: ")
recommended_movie = recommend_movie(user_input_feature)

# Print the recommended movie titles (or indices, based on your data)
print("Recommended Movies:", recommended_movie)


# prediction = model.kneighbors(user_input_feature_pca, n_neighbors=5)
