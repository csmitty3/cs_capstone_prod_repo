from flask import Flask, render_template, request, send_file, redirect, url_for
from flask_restful import Api, Resource
import pickle
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import requests
from sql_extract import Extract_data
from lstm import df_to_windowed_df, windowed_df_to_date_X_y, recursive_predict


app = Flask(__name__)
api = Api(app)

fig, ax = plt.subplots(figsize=(7,2.5), facecolor='lightblue')

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predictions')
def predictions(result):
    return render_template('result.html', result=result)

@app.route('/visualize')
def visualize():
    """
    Creates graph to be displayed on inital page of application
    Sends matplotlib graph to html page to display line chart
    """
    df = Extract_data()
    ax.plot(df, color='#A45EE5')
    ax.set_facecolor("lightblue")
    ax.set_title('S&P 500 Stock Prices for 2022 and 2023', fontsize= 10)
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('Price',fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', labelsize=5, length=0)
    img=io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype="img/png")



@app.route("/predict", methods=['POST', "GET"])
def predict():
    """
    When a number of predictions is give- Runs functions need to recrusively predict
    """
    if request.method=='POST':
        num = int(request.form['preds'])
        df = Extract_data()
        df_index = df[-1: ]
        index = df_index.index
        index_date = index.date[0]
        index_date = datetime.strftime(index_date, "%Y-%m-%d")
        windowed_df, scaler = df_to_windowed_df(df, '2022-01-10', index_date, n=5)
        dates, X, Y = windowed_df_to_date_X_y(windowed_df)
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        recursive_predictions = recursive_predict(num, X, model, scaler)
        return render_template('result.html', result=recursive_predictions)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)