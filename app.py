from flask import Flask,request,render_template
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            grid=int(request.form.get('grid')),
            position=int(request.form.get('position')),
            laps=int(request.form.get('laps')),
            fastestLap=int(request.form.get('fastestLap')),
            rank=int(request.form.get('rank')),
            fastestLapSpeed=float(request.form.get('fastestLapSpeed')),
            drivername=request.form.get('drivername'),
            year=int(request.form.get('year')),
            round=int(request.form.get('round')),
            location=request.form.get('location'),
            constructorname=request.form.get('constructorname'),
            status=request.form.get('status'),
            race_age=int(request.form.get('race_age')),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0") 