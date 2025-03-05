import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer


# df = pd.read_sql('SELECT * FROM movie_details', connection)  # Or load from CSV
engine = create_engine('postgresql://shree:Tech%40123@localhost:5432/movie_db')

# connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

# cursor = connection.cursor()


select = "select * from movie_details"
df = pd.read_sql(select, engine)


x = df[['genre', 'year', 'rating', 'votes', 'detail_story', 'type', 'meta_score']]
y = df['movie_name']


label_encoder = LabelEncoder()

x.loc[:, 'genre'] = label_encoder.fit_transform(x['genre'])
x.loc[:, 'type'] = label_encoder.fit_transform(x['type'])
x.loc[:, 'detail_story'] = label_encoder.fit_transform(x['detail_story'])


# Impute missing values with the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(x['detail_story'] )
print(text_features.toarray())


x['detail_story'] = x['detail_story'].fillna('')

vectorizer = TfidfVectorizer()

# Fit and transform the 'detail_story' column
text_features = vectorizer.fit_transform(df['detail_story'])

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)


def recommend_movie(user_input_feature):
    
    default_values = {
        'genre': ' ',  
        'year': ' ',         
        'rating': 7.5,        
        'votes': 1000,         
        'detail_story': ' ',  
        'type': ' ',      
        'meta_score': 70       
    }
    
    if isinstance(user_input_feature, str):
       
        if user_input_feature.lower() in ['action', 'comedy', 'drama', 'adventure', 'horror']: 
            feature_name = 'genre'
        else:
            feature_name = 'detail_story'  
    elif isinstance(user_input_feature, (int, float)):
        # Assign numeric values to rating, year, or votes
        if user_input_feature > 10:  
            feature_name = 'votes'
        elif 1 <= user_input_feature <= 10: 
            feature_name = 'rating'
        else:  
            feature_name = 'year'

 
    user_input = {key: default_values[key] for key in default_values}
    user_input[feature_name] = user_input_feature  
    
   
    input_data = pd.DataFrame([user_input])  
    

    input_data['genre'] = label_encoder.transform([user_input['genre']]) if user_input['genre'] in label_encoder.classes_ else -1
    input_data['type'] = label_encoder.transform([user_input['type']]) if user_input['type'] in label_encoder.classes_ else -1
    input_data['detail_story'] = label_encoder.transform([user_input['detail_story']]) if user_input['detail_story'] in label_encoder.classes_ else -1
    
    #  impute missing values
    input_data = imputer.transform(input_data)

    input_data_scaled = scaler.transform(input_data)

    predicted_movie = model.predict(input_data_scaled)

    movie_name = label_encoder_y.inverse_transform(predicted_movie)
    
    return movie_name


user_input_feature = input("Search for movie : ") 


recommended_movie = recommend_movie(user_input_feature)
print("Recommended Movie:", recommended_movie)

