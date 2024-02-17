from flask import Flask, render_template, request, app, jsonify, url_for, redirect
import pickle
import numpy as np 
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data'] 
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_features = scalar.transform(np.array(input_features).reshape(1,-1))
    print(final_features)
    output = model.predict(final_features)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)