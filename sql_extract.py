#import the relevant sql library 
from sqlalchemy import create_engine, text
import pandas as pd
import psycopg2

#def Extract_data():
    #engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    #conn = psycopg2.connect(host="ec2-54-156-8-21.compute-1.amazonaws.com",
        #database="df050o1ta4o7ba",
        #user="eeedbjazwbtsbo",
        #password="39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840")
    #cur = conn.cursor()
    #with engine.connect() as conn:
        #sql_query = pd.read_sql_query(text("SELECT * FROM sp500"), conn)
    #df = pd.DataFrame(sql_query, columns = ['Date', 'Close'])
    #df = df.set_index('Date')
    #return df

def Extract_data():
    #engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    with psycopg2.connect(host="ec2-54-156-8-21.compute-1.amazonaws.com",
        database="df050o1ta4o7ba",
        user="eeedbjazwbtsbo",
        password="39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840") as conn:
    #with conn.cursor() as curs:
        #curs.execute("SELECT * FROM sp500")
        sql_query = pd.read_sql_query("SELECT * FROM sp500", conn)
        #result = curs.fetchall()
    conn.close()
    df = pd.DataFrame(sql_query, columns = ['Date', 'Close'])
    df = df.set_index('Date')
    return df

def Extract_error_data():
    #engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    with psycopg2.connect(host="ec2-54-156-8-21.compute-1.amazonaws.com",
        database="df050o1ta4o7ba",
        user="eeedbjazwbtsbo",
        password="39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840") as conn:
    #with conn.cursor() as curs:
        #curs.execute("SELECT * FROM sp500")
        sql_query = pd.read_sql_query("SELECT * FROM error", conn)
        #result = curs.fetchall()
    conn.close()
    df = pd.DataFrame(sql_query, columns = ['Date', 'Error'])
    df = df.set_index('Date')
    return df
#if __name__ == "__main__":
   # sql_query = Extract_data()
    #df = pd.DataFrame(sql_query, columns = ['Date', 'Close'])
    #df = df.set_index('Date')
    #print(df)
    #df = Extract_error_data()
    #print(df)
