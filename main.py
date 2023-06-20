from flask import Flask, render_template, redirect, url_for, request
from flask_restful import Api, Resource
import pickle
import requests
from lstm import df_to_windowed_df, windowed_df_to_date_X_y, recursive_predict, create_df


app = Flask(__name__)
api = Api(app)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predictions')
def predictions(result):
    return render_template('result.html', result=result)


#class PredictFive(Resource):
    #def get(self):
        #if request.method=='POST':
            #num = int(requests.form['preds'])
            #df = create_df()
            #windowed_df = df_to_windowed_df(df, '2021-03-25', '2022-03-23', n=5)
            #dates, X, Y = windowed_df_to_date_X_y(windowed_df)
            #with open('model.pkl', 'rb') as file:
                #model = pickle.load(file)
            #recursive_predictions = recursive_predict(num, X, model)
            #return redirect(url_for('predictions', result = recursive_predictions))

@app.route("/predict", methods=['POST', "GET"])
def predict():
    if request.method=='POST':
        num = int(request.form['preds'])
        df = create_df()
        windowed_df = df_to_windowed_df(df, '2021-03-25', '2022-03-23', n=5)
        dates, X, Y = windowed_df_to_date_X_y(windowed_df)
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        recursive_predictions = recursive_predict(num, X, model)
        return render_template('result.html', result=recursive_predictions)
#api.add_resource(PredictFive, "/predict")

if __name__ == "__main__":
    app.run(debug=True)