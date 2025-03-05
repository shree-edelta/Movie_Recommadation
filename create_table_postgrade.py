# shree
# Tech@123
# movie_db
# localhost/127.0.0.1
# 5432
import psycopg2



connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()

if cursor:
    sql = '''CREATE TABLE movie_data(Genre varchar(30),
                                            Movie_Name varchar(30)primary key not null,
                                            Year int,
                                            Rating float,
                                            Votes int,
                                            Type varchar(30),
                                            Detail_Story varchar(30),
                                            Meta_score int);'''
    cursor.execute(sql)
    print("successfully created table")
    

    
cursor.close()

connection.commit()