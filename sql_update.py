#import the relevant sql library 
from sqlalchemy import create_engine, text
from lstm_copy import normal_df, df_to_windowed_df, windowed_df_to_date_X_y, recursive_predict, mle_analysis
from sql_extract import Extract_data
from datetime import date, timedelta, datetime
import pickle
import pandas as pd


def Get_data():
    df = normal_df()
    df = df[-1: ]
    return df

def Add_sql(df):
    engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    df.to_sql("sp500", con = engine, if_exists='append')

def Calc_error():
    df_extract = Extract_data()
    today = date.today()
    today = datetime.strftime(today, "%Y-%m-%d")
    windowed_df, scaler = df_to_windowed_df(df_extract, '2022-01-10', today, n=5)
    dates, X, Y = windowed_df_to_date_X_y(windowed_df)
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    recursive_prediction = recursive_predict(1, X, model, scaler)
    recursive_prediction = recursive_prediction['Prediction 1'][0]
    actual_result = df_extract['Close'].values[0]
    mae = abs(actual_result - recursive_prediction)
    upload_dict = {'Date': [today], 'Error': [mae]}
    upload_df = pd.DataFrame(upload_dict)
    engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    upload_df.to_sql("error", con = engine, if_exists='append')

def Retrain_model():
    mle_analysis()

def Run_update():
    df = Get_data()
    index = df.index
    index_date = index.date[0]
    if index_date == date.today():
        Add_sql(df)
        Calc_error()
        Retrain_model()

if __name__ == "__main__":
    Run_update()
