from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        pred_args2 = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}
        pred_args2=pd.Series(pred_args2)
        new_args1=standartScalar(pred_args2)
        print(new_args1)
        rf_model=joblib.load('rf_classifier')
        model_predcition=rf_model.predict([new_args1])
        if model_predcition[0]== 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        print(type(model_predcition))
    return render_template('predict.html', prediction = res)
def standartScalar(args):
    scaled_args=np.array(args[["age","trestbps","chol","thalach","oldpeak"]])
    scaled_args=scaled_args.reshape(-1,1)
    print(scaled_args)
    if args["sex"]==0: # male
        sex_0=1 # set male
        sex_1=0
        scaled_args=np.append(scaled_args,[sex_0,sex_1])
    elif args["sex"]==1: # female
        sex_0=0
        sex_1=1 # set female
        scaled_args=np.append(scaled_args,[sex_0,sex_1])
    if args["cp"]==0:
        cp_0=1
        cp_1=0
        cp_2=0
        cp_3=0
        scaled_args=np.append(scaled_args,[cp_0,cp_1,cp_2,cp_3])
    elif args["cp"]==1:
        cp_0=0
        cp_1=1
        cp_2=0
        cp_3=0
        scaled_args=np.append(scaled_args,[cp_0,cp_1,cp_2,cp_3])
    elif args["cp"]==2:
        cp_0=0
        cp_1=0
        cp_2=1
        cp_3=0
        scaled_args=np.append(scaled_args,[cp_0,cp_1,cp_2,cp_3])
    elif args["cp"]==3:
        cp_0=0
        cp_1=0
        cp_2=0
        cp_3=1
        scaled_args=np.append(scaled_args,[cp_0,cp_1,cp_2,cp_3])
    if args["fbs"]==0:
        fbs_0=1
        fbs_1=0
        scaled_args=np.append(scaled_args,[fbs_0,fbs_1])
    elif args["fbs"]==1:
        fbs_0=0
        fbs_1=1
        scaled_args=np.append(scaled_args,[fbs_0,fbs_1])
    if args['restecg']==0:
        restecg_0=1
        restecg_1=0
        restecg_2=0
        scaled_args=np.append(scaled_args,[restecg_0,restecg_1,restecg_2])
    elif args['restecg']==1:
        restecg_0=0
        restecg_1=1
        restecg_2=0
        scaled_args=np.append(scaled_args,[restecg_0,restecg_1,restecg_2])
    elif args['restecg']==2:
        restecg_0=0
        restecg_1=0
        restecg_2=1
        scaled_args=np.append(scaled_args,[restecg_0,restecg_1,restecg_2])
    if args['exang']==0:
        exang_0=1
        exang_1=0
        scaled_args=np.append(scaled_args,[exang_0,exang_1])
    elif args['exang']==1:
        exang_0=0
        exang_1=1
        scaled_args=np.append(scaled_args,[exang_0,exang_1])
    if args['slope']==0:
        slope_0=1
        slope_1=0
        slope_2=0
        scaled_args=np.append(scaled_args,[slope_0,slope_1,slope_2])
    elif args['slope']==1:
        slope_0=0
        slope_1=1
        slope_2=0
        scaled_args=np.append(scaled_args,[slope_0,slope_1,slope_2])
    elif args['slope']==2:
        slope_0=0
        slope_1=0
        slope_2=1
        scaled_args=np.append(scaled_args,[slope_0,slope_1,slope_2])
    if args['ca']==0:
        ca_0=1
        ca_1=0
        ca_2=0
        ca_3=0
        ca_4=0
        scaled_args=np.append(scaled_args,[ca_0,ca_1,ca_2,ca_3,ca_4])
    elif args['ca']==1:
        ca_0=0
        ca_1=1
        ca_2=0
        ca_3=0
        ca_4=0
        scaled_args=np.append(scaled_args,[ca_0,ca_1,ca_2,ca_3,ca_4])
    elif args['ca']==2:
        ca_0=0
        ca_1=0
        ca_2=1
        ca_3=0
        ca_4=0
        scaled_args=np.append(scaled_args,[ca_0,ca_1,ca_2,ca_3,ca_4])
    elif args['ca']==3:
        ca_0=0
        ca_1=0
        ca_2=0
        ca_3=3
        ca_4=0
        scaled_args=np.append(scaled_args,[ca_0,ca_1,ca_2,ca_3,ca_4])
    elif args['ca']==4:
        ca_0=0
        ca_1=0
        ca_2=0
        ca_3=0
        ca_4=1
        scaled_args=np.append(scaled_args,[ca_0,ca_1,ca_2,ca_3,ca_4])
    if args['thal']==0:
        thal_0=1
        thal_1=0
        thal_2=0
        thal_3=0
        scaled_args=np.append(scaled_args,[thal_0,thal_1,thal_2,thal_3])
    elif args['thal']==1:
        thal_0=0
        thal_1=1
        thal_2=0
        thal_3=0
        scaled_args=np.append(scaled_args,[thal_0,thal_1,thal_2,thal_3])
    elif args['thal']==2:
        thal_0=0
        thal_1=0
        thal_2=1
        thal_3=0
        scaled_args=np.append(scaled_args,[thal_0,thal_1,thal_2,thal_3])
    elif args['thal']==3:
        thal_0=0
        thal_1=0
        thal_2=0
        thal_3=1
        scaled_args=np.append(scaled_args,[thal_0,thal_1,thal_2,thal_3])
    pred_args=[]
    for i in scaled_args:
        pred_args.append(i)
    return pred_args



if __name__ == '__main__':
    app.run()