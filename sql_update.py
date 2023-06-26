#import the relevant sql library 
from sqlalchemy import create_engine, text
from lstm_copy import normal_df, df_to_windowed_df, windowed_df_to_date_X_y, recursive_predict, mle_analysis
from sql_extract import Extract_data
from datetime import date, timedelta, datetime
import pickle
import pandas as pd


def Get_data():
    """
    This function calls the 'normal_df function from 'lstm_copy.py' which is used to pull in API data from Financial Modeling Prep
    This then takes and returns just the last row of data from that DataFrame
    """
    df = normal_df()
    df = df[-1: ]
    return df

def Add_sql(df):
    """
    Connects to Postgres Database on Heroku and adds the latest data from the API call.
    """
    engine = create_engine("postgresql://eeedbjazwbtsbo:39d5efb393a525797a40aa4edefbce1680207d94ec798aa650a9ea1e481e8840@ec2-54-156-8-21.compute-1.amazonaws.com:5432/df050o1ta4o7ba", echo = False)
    df.to_sql("sp500", con = engine, if_exists='append')

def Calc_error():
    """
    Finds the difference between the recursive prediction and the actual outcome of the day + uploads to sql.
    """
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
    """
    Calls the 'mle_analysis' function from 'lstm_copy' to retrain the deep learning model with the latest data.
    """
    mle_analysis()

def Run_update():
    """
    Checks to see if today's date is equivalent to the latest date in the API call. If so- this script is run daily on Heroku's scheduler to update the data and retrain the model.
    """
    df = Get_data()
    index = df.index
    index_date = index.date[0]
    if index_date == date.today():
        Add_sql(df)
        Calc_error()
        Retrain_model()

if __name__ == "__main__":
    Run_update()
