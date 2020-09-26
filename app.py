# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:11:36 2020

@author: team3
"""
import traceback
import numpy as np
from flask import Flask,request,render_template
import pickle
try:
    app = Flask(__name__,template_folder='template')
    model = pickle.load(open('model.pkl', 'rb'))
    @app.route("/")
    def home():
     return render_template("index.html")
    @app.route("/go")
    def go():
     return render_template("go.html")
    @app.route("/form")
    def form():
     return render_template("form.html")
    @app.route("/output")
    def output(prediction):
     return render_template("output.html",prediction)
    @app.route("/risk")
    def risk():
     return render_template("risk.html")
    @app.route("/about")
    def about():
     return render_template('about.html')
    @app.route('/predict',methods=['POST','GET'])
    def predict():
     features = []
     tsh=request.form['tsh']
     features.append(float(tsh))
     final_features = [np.array(features)]
     print(final_features)
     prediction = model.predict(final_features)
     print(prediction)
     if prediction[0]==1:
         return render_template('output.html', prediction='HypoThyroid!')
     elif prediction[0]==0:
         return render_template('output.html', prediction='Hypothyroid Negative!')
    if __name__ == "__main__":
     app.run(debug=True,use_reloader=False,port=5555)
except :
    traceback.print_exc()

 
