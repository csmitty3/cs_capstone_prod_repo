#import the relevant sql library 
from sqlalchemy import create_engine, text
from lstm_copy import normal_df
from datetime import date, timedelta

def Add_sql():
    df = normal_df()
    df = df[-1: ]
    index = df.index
    index_date = index.date[0]
    if index_date == date.today():
        engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
# attach the data frame (df) to the database with a name of the 
# table; the name can be whatever you like
        df.to_sql("sp500", con = engine, if_exists='append')
# run a
#with engine.connect() as conn:
    #results = conn.execute(text("SELECT * FROM sp500;")).fetchall()
#for row in results:
    #print(row)
#print('complete')
if __name__ == "__main__":
    Add_sql()