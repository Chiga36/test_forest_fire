from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scalar pickle
import os

# Specify the absolute path to your files
base_path = r'C:\Users\Dell\Desktop\data\science\ds\steps_to_implement\mm'
ridge_path = os.path.join(base_path, 'ridge.pkl')
scalar_path = os.path.join(base_path, 'scalar.pkl')

ridge_model = pickle.load(open(ridge_path, 'rb'))
standard_scalar = pickle.load(open(scalar_path, 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled = standard_scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
    # bhjghjgjh