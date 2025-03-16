# from flask import Flask,request, url_for, redirect, render_template
# import pickle
# import numpy as np

# app = Flask(__name__, template_folder='template')


# model=pickle.load(open('model.pkl','rb'))


# @app.route('/')
# def hello_world():
#     return render_template("index.html")


# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     int_features=[int(x) for x in request.form.values()]
#     final=[np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction=model.predict_proba(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)

#     if output>str(0.5):
#         return render_template('index.html',pred='You need a treatment.\nProbability of mental illness is {}'.format(output))
#     else:
#         return render_template('index.html',pred='You do not need treatment.\n Probability of mental illness is {}'.format(output))


# if __name__ == '__main__':
#     app.run(debug=True)


import os
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='template')

# Get the absolute path to the model in the root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get current script directory
MODEL_PATH = os.path.join(BASE_DIR, '..', 'mental_health_model.keras')  # Locate the model

# Load the Keras model
model = load_model(MODEL_PATH)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]  # Convert input to float
    final = np.array([int_features])  # Convert list to NumPy array
    print("Input Features:", int_features)
    print("Final Array:", final)

    prediction_prob = model.predict(final)[0][0]  # Get prediction probability

    output = '{0:.2f}'.format(prediction_prob)  # Format probability

    if float(output) > 0.5:
        return render_template('index.html', pred=f'You need treatment. Probability: {output}')
    else:
        return render_template('index.html', pred=f'You do not need treatment. Probability: {output}')

if __name__ == '__main__':
    app.run(debug=True)
