#import the relevant sql library 
from sqlalchemy import create_engine, text
from lstm_copy import normal_df
import pandas as pd

def Create_db():
    engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    sql_create_script = "CREATE TABLE sp500(Date DATE PRIMARY KEY,Close NUMERIC);"
    with engine.connect() as conn:
        conn.execute(text(sql_create_script))

def Create_initial_data():
    df = normal_df()
    df=df.loc['2022-01-03':'2023-06-21']
    engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    df.to_sql("sp500", con = engine, if_exists='append')

def Extract_data():
    engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)

    with engine.connect() as conn:
        sql_query = pd.read_sql_query(text("SELECT * FROM sp500"), conn)
    df = pd.DataFrame(sql_query, columns = ['Date', 'Close'])
    df = df.set_index('Date')
    return df

#Create_db()
#Create_initial_data()
#if __name__ == '__main__':
    #df = Extract_data()
    #print(type(df.index))