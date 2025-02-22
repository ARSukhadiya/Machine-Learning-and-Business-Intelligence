from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
# Load models
model = joblib.load('voting_regressor_model.pkl')

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    carat = float(request.form['carat'])
    cut = request.form['cut']
    color = request.form['color']
    clarity = request.form['clarity']
    depth = float(request.form['depth'])
    table = float(request.form['table'])
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])
    volume=x*y*z

    # Prepare input for prediction
    input_features = pd.DataFrame([{'carat':carat,'cut':cut,'color':color,'clarity':clarity,'depth':depth,'table':table,'volume':volume}])

    # Make prediction
    prediction = model.predict(input_features)
    prediction = round(np.expm1(prediction[0]), 2)

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
