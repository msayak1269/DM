from unittest import result
from flask import (
    Flask, render_template,request
)
import os
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__, static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "thisisasecretkey"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def predictWithEnsemble(to_predict_list):
    print(to_predict_list)
    loaded_model = pickle.load(open("DT_model.pkl", "rb"))
    result = loaded_model.predict(to_predict_list)
    return result[0]

@app.route("/")
def home():
    attributes = [0,0,0,0,0,0,0,0]
    return render_template("form.html", result = '', attributes = attributes)

@app.route("/getting/result",methods = ["POST"])
def getResult():
    input = request.form.to_dict()
    attributes = list(input.values())
    print(attributes)
    attributes = list(map(float, attributes))
    attr = [attributes]
    col = ['Pregnancies', 'Glucose',  'BloodPressure',  'SkinThickness',  'Insulin',  'BMI',  'DiabetesPedigreeFunction', 'Age']
    dataFrame = pd.DataFrame(attr, columns = col)
    result = predictWithEnsemble(dataFrame)
    print(result)
    if(result == 1):
        return render_template("form.html", attributes = attributes,result = 'Yes')
    else:
        return render_template("form.html", attributes = attributes, result = 'No')


if __name__=="__main__":
    app.run(port=5001, debug=True, host='0.0.0.0')