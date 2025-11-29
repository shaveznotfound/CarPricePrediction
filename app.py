from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Ensure these paths are correct relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'LinearRegressionModel.pkl')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Cleaned_data.csv')

# Load model and data
model = pickle.load(open(MODEL_PATH, 'rb'))
car = pd.read_csv(CSV_PATH)

@app.route("/")
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].dropna().unique().tolist())

    models_by_company = (
        car.groupby('company')['name']
        .unique()
        .apply(lambda arr: sorted(arr.tolist()))
        .to_dict()
    )

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        models_by_company=models_by_company
    )

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # get form values
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # basic validation/fallbacks
    try:
        year = int(year)
    except Exception:
        return jsonify({'error': 'Invalid year'}), 400

    try:
        driven = float(driven)
    except Exception:
        return jsonify({'error': 'Invalid kilometres driven'}), 400

    # Build DataFrame expected by your trained model
    X = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                     data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))

    # predict
    try:
        prediction = model.predict(X)
        predicted_price = float(np.round(prediction[0], 2))
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'prediction': predicted_price})

if __name__ == '__main__':
    app.run(debug=True, port=5000)