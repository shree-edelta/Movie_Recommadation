import psycopg2



connection = psycopg2.connect(database="movie_db", user="shree", password="Tech@123", host="localhost", port=5432)

cursor = connection.cursor()

if cursor:
    select = "select * from movie_data"
    cursor.execute(select)
    rows = cursor.fetchall()
    for row in rows:
        print(row)