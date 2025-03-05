import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Assuming df is your DataFrame, and it's already loaded
connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()

if cursor:
    select = "select * from movie_details"
    df = pd.read_sql(select, connection)
    cursor.execute(select)
    rows = cursor.fetchall()

cursor.close()

connection.commit()
connection.close() 
# Preprocessing the features (X) and labels (y)
x = df[['genre', 'year', 'rating', 'votes', 'detail_story', 'type', 'meta_score']]
y = df['movie_name']

# Label encoding for categorical features
label_encoder = LabelEncoder()

# Ensure that all categorical columns are properly encoded to numerical
x['genre'] = label_encoder.fit_transform(x['genre'])
x['type'] = label_encoder.fit_transform(x['type'])
x['detail_story'] = label_encoder.fit_transform(x['detail_story'])

y = label_encoder.fit_transform(df['movie_name'])

# Impute missing values with the mean (you can also use median or another strategy)
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Initialize and fit the KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='minkowski', algorithm='auto')
# knn.fit(x_train, y_train)

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
 
# pred = knn.predict(x_test)
print(pred)
def recommend_movie(user_input_feature):
    
    recommended_movie = df[df['movie_name'] == user_input_feature]['movie_name'].values[0]
    return recommended_movie

user_input_feature = input("Search for movie : ") 
recommended_movie = recommend_movie(user_input_feature)
print("Recommended Movie:", recommended_movie)
