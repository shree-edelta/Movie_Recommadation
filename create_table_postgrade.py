# shree
# Tech@123
# movie_db
# localhost/127.0.0.1
# 5432
import psycopg2



connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()

if cursor:
    sql = '''CREATE TABLE movie_details(Genre varchar(255),
                                            Movie_Name varchar(255)primary key not null,
                                            Year bigint,
                                            Rating float,
                                            Votes bigint,
                                            Type varchar(255),
                                            Detail_Story varchar(255),
                                            Meta_score float);'''
    cursor.execute(sql)
    print("successfully created table")
    

    
cursor.close()

connection.commit()