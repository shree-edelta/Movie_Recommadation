import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Label encoding for categorical features
label_encoder = LabelEncoder()

# Using .loc[] to prevent SettingWithCopyWarning
x.loc[:, 'genre'] = label_encoder.fit_transform(x['genre'])
x.loc[:, 'type'] = label_encoder.fit_transform(x['type'])
x.loc[:, 'detail_story'] = label_encoder.fit_transform(x['detail_story'])

# Impute missing values with the mean (you can also use median or another strategy)
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Convert 'movie_name' to numeric labels (for classification task)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(df['movie_name'])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Initialize and fit the Logistic Regression model (or use any model you prefer)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
pred = model.predict(x_test)
# print("Predictions:", pred)
# print("Accuracy:", accuracy_score(y_test, pred))

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, x_scaled, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())
