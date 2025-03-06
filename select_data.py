import psycopg2
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import sklearn.neighbors as neighbors
from sklearn.neighbors import NearestNeighbors

connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()

if cursor:
    select = "select * from movie_details"
    df = pd.read_sql(select, connection)
    cursor.execute(select)
    rows = cursor.fetchall()

cursor.close()

connection.commit()
connection.close()  # Close the connection


x = df[['genre','year','rating','votes','detail_story','type','meta_score']]
y = df['movie_name']


label_encoder = LabelEncoder()

x.loc[:, 'genre'] = label_encoder.fit_transform(x['genre'])
x.loc[:,'type'] = label_encoder.fit_transform(x['type'])
x.loc[:,'detail_story'] = label_encoder.fit_transform(x['detail_story'])

# If you have multiple categorical columns, you can apply One-Hot Encoding
x = pd.get_dummies(x, columns=['genre', 'type','detail_story'], drop_first=True)  # One-Hot Encoding

# Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# print(x_scaled[:5]) 
# x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# knn = neighbors.KNeighborsClassifier(n_neighbors=10,weights='distance',metric='minkowski',algorithm='auto')
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean', algorithm='auto')
# similar_movies = knn.kneighbors([movie_features.iloc[0].values], n_neighbors=6)
# knn = NearestNeighbors(n_neighbors=5, metric='cosine')
# knn.fit(x_scaled)

# Find 5 most similar movies to a given movie (example: movie_id 1)
# similar_movies = knn.kneighbors([x_scaled.iloc[0].values], n_neighbors=6)
# print(similar_movies)
 
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

set = knn.fit(x_train, y_train)
pred = set.predict(x_test)
print(pred)
# print("Accuracy:", accuracy_score(y_test, pred))


# print("confusion matrix",confusion_matrix(y_test, pred)) 

