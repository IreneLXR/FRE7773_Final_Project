"""
    This script runs a small Flask app that displays a simple web form for users to insert some input value
    and retrieve predictions.

    Inspired by: https://medium.com/shapeai/deploying-flask-application-with-ml-models-on-aws-ec2-instance-3b9a1cec5e13
"""

from flask import Flask, render_template, request
import numpy as np
from metaflow import Flow
from metaflow import get_metadata, metadata
from io import StringIO
import pandas as pd
from matplotlib import pyplot as plt
from flask import jsonify
import os
import pickle
from datetime import datetime
#### THIS IS GLOBAL, SO OBJECTS LIKE THE MODEL CAN BE RE-USED ACROSS REQUESTS ####

FLOW_NAME = 'MyFlow' # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('./')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())

# get latest successful run from metadata
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r
# calculate errors      
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# generate future volatility trend graph and the specific volatilty at a point based on stock id and time id
def process_data(_x: int, time: int):
    # group data by stock id to make prediction
    df_stock_id_x = latest_df_test.get_group(_x).apply(lambda x: x) 
    X_test_stock_id_x = df_stock_id_x[[col for col in df_stock_id_x.columns if col not in ['target','time_id','stock_id','level_0','index']]]
    y_test_stock_id_x = df_stock_id_x['target']
    y_pred = latest_model.predict(X_test_stock_id_x)
    rmspe_x = round(rmspe(y_test_stock_id_x, y_pred), 5)
    
    # draw graph
    y_pred = pd.Series(y_pred, y_test_stock_id_x.index)
    y_pred_series = pd.Series(y_pred, y_test_stock_id_x.index)
    y_at_time = round(y_pred_series.iloc[time], 5)
    df_stock_id_x['time_id'] = df_stock_id_x['time_id'] - df_stock_id_x['time_id'].values[0]
    plt.plot(df_stock_id_x['time_id'], y_test_stock_id_x, 'b', zorder=1)
    plt.plot(df_stock_id_x['time_id'], y_pred_series, color='orange', zorder=2)
    plt.scatter(df_stock_id_x['time_id'][y_pred_series.index[time]], y_at_time, marker='X', c='red', s=100, zorder=3)
    classes = ['volatility_true', 'volatility_predict', 'volatility_at_time_' + str(time)]
    plt.legend(labels=classes)
    # if static dir does not exist, mkdir
    if os.path.isdir('./static') == False:
        os.mkdir('./static')
    plt.savefig('./static/stock_id_' + str(_x) + '_time_id_' + str(time) + '.png')
    plt.close()
    img_path = '/static/stock_id_' + str(_x) + '_time_id_' + str(time) + '.png'
    return img_path, rmspe_x, y_pred, y_at_time
    
# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.best_lgb_model
latest_df_test = latest_run.data.df_test_group_by_stock

# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/',methods=['POST','GET'])
def main():

  # on GET we display the page  
  if request.method=='GET':
    return render_template('index.html', project=FLOW_NAME)
  # on POST we make a prediction over the input text supplied by the user
  if request.method=='POST':
    data = request.get_json()
    print(data)
    _idx = data["id"]
    _idx = int(_idx)
    _id = latest_run.data.total_stock_ids[_idx]
    _time = data["time"]
    _time = int(_time)
    img_path, rmspe_x, _, y_at_time = process_data(_id, _time)
    return jsonify({'img_path':img_path, 'rmspe': 'rmspe = ' + str(rmspe_x), 'y_at_time': 'volatility at time ' + str(_time) + ' is ' + str(y_at_time)})
    
@app.route('/predict', methods=['GET'])
def predict():
    if request.method=='GET':
        stock_id_idx = int(request.args.get('stock_id'))
        stock_id = latest_run.data.total_stock_ids[stock_id_idx]
        time_id = int(request.args.get('time_id'))
        img_path, rmspe_x, y_pred, y_at_time = process_data(stock_id, time_id)
        time = datetime.now()
        predict_dict = {
            "data": {
                "prediction_of_volatility_at_time=" + str(time_id):y_at_time,
                "predictions_of_volatility_for_whole_period":[y_pred.tolist()]
            },
            "metadata": {
                "serverTimeStamp": time.timestamp(),
                "time": time
                }
        }
        response = jsonify(predict_dict)
        response.status_code = 200
        return response

    
if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)
