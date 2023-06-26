#import the relevant sql library 
from sqlalchemy import create_engine, text
import pandas as pd
import psycopg2


def Extract_data():
    with psycopg2.connect(host="ec2-54-156-8-21.compute-1.amazonaws.com",
        database="df050o1ta4o7ba",
        user="eeedbjazwbtsbo",
        password="39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840") as conn:

        sql_query = pd.read_sql_query("SELECT * FROM sp500", conn)

    conn.close()
    df = pd.DataFrame(sql_query, columns = ['Date', 'Close'])
    df = df.set_index('Date')
    return df

def Extract_error_data():
    with psycopg2.connect(host="ec2-54-156-8-21.compute-1.amazonaws.com",
        database="df050o1ta4o7ba",
        user="eeedbjazwbtsbo",
        password="39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840") as conn:
        sql_query = pd.read_sql_query("SELECT * FROM error", conn)
    conn.close()
    df = pd.DataFrame(sql_query, columns = ['Date', 'Error'])
    df = df.set_index('Date')
    return df
#if __name__ == "__main__":