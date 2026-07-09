from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Columns after one-hot encoding (must match train_model.py)
feature_columns = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS',
    'MaxHR', 'Oldpeak',
    'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP',
    'ChestPainType_TA', 'RestingECG_Normal',
    'RestingECG_ST', 'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Age': int(request.form['Age']),
        'RestingBP': int(request.form['RestingBP']),
        'Cholesterol': int(request.form['Cholesterol']),
        'FastingBS': int(request.form['FastingBS']),
        'MaxHR': int(request.form['MaxHR']),
        'Oldpeak': float(request.form['Oldpeak']),
        'Sex_M': 1 if request.form['Sex'] == 'M' else 0,
        'ChestPainType_ATA': 1 if request.form['ChestPainType'] == 'ATA' else 0,
        'ChestPainType_NAP': 1 if request.form['ChestPainType'] == 'NAP' else 0,
        'ChestPainType_TA': 1 if request.form['ChestPainType'] == 'TA' else 0,
        'RestingECG_Normal': 1 if request.form['RestingECG'] == 'Normal' else 0,
        'RestingECG_ST': 1 if request.form['RestingECG'] == 'ST' else 0,
        'ExerciseAngina_Y': 1 if request.form['ExerciseAngina'] == 'Y' else 0,
        'ST_Slope_Flat': 1 if request.form['ST_Slope'] == 'Flat' else 0,
        'ST_Slope_Up': 1 if request.form['ST_Slope'] == 'Up' else 0
    }

    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        result = "High Risk of Heart Disease"
        color = "red"
    else:
        result = "Low Risk of Heart Disease"
        color = "green"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability, 2),
        color=color
    )

if __name__ == '__main__':
    app.run(debug=True)
