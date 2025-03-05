import pandas as pd
import psycopg2


df = pd.read_csv('/Users/bhavik/Desktop/Shree/movie_data.csv',encoding='latin1')

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
df['Year'] =df['Year'].str.replace('?','')

print(df['Votes'])
# insert data

connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()

if cursor:
    sql = '''insert into movie_data(Genre,Movie_Name,Year,Rating,Votes,Type,Detail_Story,Meta_score) values(%s,%s,%s,%s,%s,%s,%s,%s);'''
    for i in range(len(df)):
        cursor.execute(sql,(df['Genre'][i],df['Movie Name'][i],df['Year'][i],df['Rating'][i],df['Votes'][i],df['Type'][i],df['Detail/Main Story'][i],df['Metascore'][i]))
        connection.commit()
    print("successfully inserted data")
    
    
    
cursor.close()

connection.commit()
