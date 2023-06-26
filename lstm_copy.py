#standard imports
import pandas as pd
#import datetime
import matplotlib.pyplot as plt
import numpy as np

#tensorflow imports for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import layers
from keras.constraints import maxnorm
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



#sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

#copy for recursive
from copy import deepcopy

from bayes_opt import BayesianOptimization

import pickle

import unittest

import requests
import json
from sql_extract import Extract_data
from datetime import datetime, date, timedelta


def get_jsonparsed_data(url):
    """
    Take contents of url call and parse as json
    url- website to connect to(str)
    return- json of the api call
    """
    response = requests.get(url)

    return response.json()



#function to change string dates to datetime
def str_to_datetime(s):
    """
    Takes a date in string format and returns in datetime format
    s- a date(str)
    return- a date(datetime)
    """
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime(year=year, month=month, day=day)

def normal_df():
    """
    Calls Financial Modeling Prep API and obtains historical data for S&P500 + cleans up.
    return- Pandas DataFrame with 'Date' as the index and 'Close' as the only column
    """
    api_key= 'dee9e143b1d0b3ce72ab2bf088fbfab9'
    sp500='^GSPC'
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{sp500}?apikey={api_key}")
    dictionary = get_jsonparsed_data(url)

    dates=[]
    history = dictionary['historical']
    for i in range(len(history)):
        dates.insert(0, history[i]['date'])
    
    close=[]
    for i in range(len(history)):
        close.insert(0, history[i]['close'])

    price_dict = {"Date": dates, "Close": close}
    df = pd.DataFrame(price_dict)

    #apply str_to_datetime
    df['Date'] = df['Date'].apply(str_to_datetime)

    #set index
    df.set_index('Date', inplace=True)
    return df

def create_df():
    """
    Extracts data from Postgres Database on Heroku and scales with a MinMaxScaler
    returns- Pandas DataFrame
    """
    scaler=MinMaxScaler(feature_range=(0,1))
    df = Extract_data()
    scaledclose=scaler.fit_transform(np.array(df).reshape(-1,1))
    df['Close'] = scaledclose
    df = df[['Close']]
    return df


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    """
    Takes S&P stock data and creates a windowed DataFrame
    dataframe- Pandas DF with 'Date' index and 'Close' column
    first_date_str- The starting date in the timeframe(str)
    last_date_str- The last dat in the timeframe (str)
    n- number of windows to create(int)(default = 3)
    RETURNS
    ret_df- Pandas DataFrame with windowed columns
    scaler- If scaler was used in this step- will return to be used later to descale
    """
    if dataframe['Close'].values[0] > 1:
        scaler=MinMaxScaler(feature_range=(0,1))
        scaledclose=scaler.fit_transform(np.array(dataframe).reshape(-1,1))
        dataframe['Close'] = scaledclose
    else:
        scaler = ''

    first_date = str_to_datetime(first_date_str)

    last_date  = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)
        next_week = dataframe.loc[target_date:target_date+timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime(day=int(day), month=int(month), year=int(year))
 
        if last_time:
            break
    
        target_date = next_date

        if target_date == last_date:
            last_time = True
 
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
  
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n-i}'] = X[:, i]
  
    ret_df['Target'] = Y

    return ret_df, scaler

def windowed_df_to_date_X_y(windowed_dataframe):
    """
    Takes windowed dataframe and returns the dates, X, and Y values
    windowed_dataframe- A Pandas DF that has been put through the df_to_windowed_df function
    RETURNS
    dates- all dates in DF
    X- All X values in DF(np.float32)
    Y- All y values in DF(np.float32)
    """
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


def split_train_val_test(dates, X, y):
    """
    Splits dates, X, y into train val and test data- preparing for Deep Learning
    dates-dates used for processing
    X- all X values used for processing
    y- all y values used for processing
    RETURNS
    dates_train- All dates within training data
    X_train- All X values within training data
    y_train- All y values within training data
    dates_val- All dates within validation data
    X_val- All X values within validation data
    y_val- All y values within validation data
    dates_test- All dates within test data
    X_test- All X values with test data
    y_test- All y values within test data
    """
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test


def bayesian_optimization(X_train, y_train, X_val, y_val):
    """
    Bayesian Optimization function used to find optimal hyperparameters for LSTM Neural Networks
    X_train- List of X_train data
    y_train- list of y_train data
    X_val- List of X_val data
    y_val- list of y_val data
    RETURN 
    learning_rate- optimal learning rate for model(float)
    neurons- optimal neurons for model(int)
    dropout_rate- optimal dropout_rate for model
    activation- optimal type of activation for model(str)
    """
    def define_model(learning_rate, neurons, dropout_rate, activation):
        """
        Helper function to define the model
        learning_rate- learning rate for model(float)
        neurons- neurons for model(int)
        dropout_rate- dropout_rate for model
        activation- type of activation for model(str)
        RETURNS
        model- model that these various hyperparams created
        """
        activation_mapping = {0: 'relu', 1: 'sigmoid'}
        activation = activation_mapping[int(activation)]
        
        model = Sequential([
            layers.Input((X_train.shape[1], 1)),
            layers.LSTM(neurons),
            layers.Dense(32, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation=activation),
            layers.Dense(1)
        ])

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=['mean_absolute_error'])

        return model

    # Define the parameter search space
    params = {
        'learning_rate': (0.001, 0.15),
        'neurons': (32, 64),
        'dropout_rate': (0.1, 0.4),
        'activation': (0, 1),
        
    }

    # Define the objective function to be maximized
    def objective(learning_rate, neurons, dropout_rate, activation):
        """
        Define the objective function to be maximized(val_loss)
        learning_rate- learning rate for model(float)
        neurons- neurons for model(int)
        dropout_rate- dropout_rate for model
        activation- type of activation for model(str)
        RETURNS
        -val_loss- Result of the objective function
        """
        model = define_model(learning_rate, int(neurons), dropout_rate, activation)
        model.fit(X_train, y_train, epochs=100, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        return -val_loss  # Negative sign because BayesianOptimization minimizes the objective

    # Perform Bayesian optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=params,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=25, n_iter=25)

    best_params = optimizer.max['params']
    learning_rate = best_params['learning_rate']
    neurons = int(best_params['neurons'])
    dropout_rate = best_params['dropout_rate']
    activation = int(best_params['activation'])  # Cast activation to int
    activation_mapping = {0: 'relu', 1: 'sigmoid'}
    activation = activation_mapping[int(activation)]
    return learning_rate, neurons, dropout_rate, activation


#Training model on best params
def train_model(X_train, y_train, X_val, y_val, dates_val, X_test, y_test, dates_test, learning_rate, neurons, activation, dropout_rate):
    """
    Train most optimal model in a thorough fashion
    dates_train- All dates within training data
    X_train- All X values within training data
    y_train- All y values within training data
    dates_val- All dates within validation data
    X_val- All X values within validation data
    y_val- All y values within validation data
    dates_test- All dates within test data
    X_test- All X values with test data
    y_test- All y values within test data
    learning_rate- learning rate for model(float)
    neurons- neurons for model(int)
    dropout_rate- dropout_rate for model
    activation- type of activation for model(str)
    RETURNS
    train_predictions- list of predictions the model makes on the training data
    val_predictions- list of predictions the model makes on the validation data
    test_predictions- list of predictions the model makes on the test data
    recursive_predictions- list of predictions the model makes on data recursively
    train_predictions- list of predictions the model makes on the training data
    model- The model used for these predictions
    """
    model = Sequential([layers.Input((X_train.shape[1], 1)),
                        layers.LSTM(neurons),
                        layers.Dense(32, activation=activation),
                        layers.Dropout(dropout_rate),
                        layers.Dense(32, activation=activation),
                        layers.Dense(1)])
    
    model.compile(loss='mse', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    
    recursive_predictions = []
    recursive_dates = np.concatenate([dates_val, dates_test])
    last_window = deepcopy(X_train[-1])
    for target_date in recursive_dates:
        next_prediction = model.predict(np.array([last_window])).flatten()
        recursive_predictions.append(next_prediction)
        for i in range(len(last_window)):
            if i == (len(last_window)-1):
                last_window[i] = next_prediction
            else:
                last_window[i] = last_window[i+1]
    return train_predictions, val_predictions, test_predictions, recursive_predictions, model

class TestCalc(unittest.TestCase):
    def test_recursive(self):
        """
        Unit Testing on recursive data- making sure length of result is equal to val+test predictions
        """
        windowed_df = df_to_windowed_df(df, '2021-03-25', '2022-03-23', n=5)
        dates, X, y = windowed_df_to_date_X_y(windowed_df)
        dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = split_train_val_test(dates, X, y)
        learning_rate, neurons, dropout_rate, activation = bayesian_optimization(X_train, y_train, X_val, y_val)
        train_predictions, val_predictions, test_predictions, recursive_predictions, model = train_model(X_train, y_train, X_val, y_val, dates_val, X_test, y_test, dates_test, learning_rate, neurons, activation, dropout_rate)
        result = len(recursive_predictions)
        self.assertEqual(result, len(val_predictions) + len(test_predictions))
    
    def test_split_train_val(self):
        """
        Unit Testing on splt_train_val data
        """
        windowed_df = df_to_windowed_df(df, '2021-03-25', '2022-03-23', n=5)
        dates, X, y = windowed_df_to_date_X_y(windowed_df)
        dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = split_train_val_test(dates, X, y)
        self.assertEqual(len(X_train), len(y_train))

def recursive_predict(num, data, model, scaler):
    """
    Creates predictions recursively that can be used to predict the next values of S&P500 data
    num- number of predictions
    data- current historical data
    model- model to be used
    scaler- Need to be used to convert data back to correct values
    RETURN
    recursive_dictionary- dictionary of recursive predictions
    """
    recursive_predictions = []
    last_window = deepcopy(data[-1])
    for i in range(num):
        next_prediction = model.predict(np.array([last_window])).flatten()
        recursive_predictions.append(next_prediction)
        for i in range(len(last_window)):
            if i == (len(last_window)-1):
                last_window[i] = next_prediction
            else:
                last_window[i] = last_window[i+1]
    recursive_predictions = scaler.inverse_transform(recursive_predictions)
    recursive_dictionary = {}
    for i in range(num):
        recursive_dictionary[f'Prediction {i+1}'] = list(recursive_predictions[i])
    return recursive_dictionary

#function to run the whole script
def mle_analysis():
    """
    Function used to run multiple of the functions above and retrain the deep learning model
    Takes final results and pickleizes the model and saves it to be used for future predictions
    """
    df= create_df()
    today = date.today()-timedelta(days=1)
    today = datetime.strftime(today, "%Y-%m-%d")
    windowed_df, scaler = df_to_windowed_df(df, '2022-01-10', today, n=5)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = split_train_val_test(dates, X, y)
    learning_rate, neurons, dropout_rate, activation = bayesian_optimization(X_train, y_train, X_val, y_val)
    train_predictions, val_predictions, test_predictions, recursive_predictions, model = train_model(X_train, y_train, X_val, y_val, dates_val, X_test, y_test, dates_test, learning_rate, neurons, activation, dropout_rate)
    #unittest.main()
    pickle.dump(model, open('model.pkl', 'wb'))
if __name__ == "__main__":
    mle_analysis()