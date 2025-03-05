import pandas as pd
import psycopg2

df = pd.read_csv('/Users/bhavik/Desktop/movie_recommded/movie_data.csv',encoding='latin1')

# print(df['Rating'])

for i in  df['Movie Name']:
    p = i.find(' ')
    movie = i[:p]
    df['Movie Name'] =df['Movie Name'].str.replace(movie,'')
for j in  df['Rating']:
    j = str(j)
    p1 = j.find('?')
    rating = j[p1:]
    df['Rating'] =df['Rating'].str.replace(rating,'')

df['Votes'] =df['Votes'].str.replace('Votes','')
df['Votes'] =df['Votes'].str.replace(',','')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

df['Year'] = df['Year'].str.split('?').str[0]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year','Votes'])

df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')



# insert data

connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()
max_length = 255

if cursor:
     sql = '''insert into movie_details(Genre,Movie_Name,Year,Rating,Votes,Type,Detail_Story,Meta_score) values(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (Movie_Name) DO NOTHING;'''
     for i in range(len(df)):
         genre = df.iloc[i]['Genre'][:max_length]
         movie_name = df.iloc[i]['Movie Name'][:max_length]
         year = int(df.iloc[i]['Year'])
         rating = float(df.iloc[i]['Rating'])  
         votes = int(df.iloc[i]['Votes'])
         detail_main_story = df.iloc[i]['Detail/Main Story'][:max_length]
         type = df.iloc[i]['Type']
         metascore = float(df.iloc[i]['Metascore'])  
         
         cursor.execute(sql,(genre,movie_name,year,rating,votes,type,detail_main_story,metascore))
         connection.commit()
print("successfully inserted data")
    
    
    
cursor.close()

connection.commit()
