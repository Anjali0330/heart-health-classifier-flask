from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

with open('model_results.json') as f:
    meta = json.load(f)

FEATURE_ORDER = meta['feature_order']
BEST_MODEL_NAME = meta['best_model']
BEST_MODEL_ACC = meta['results'][BEST_MODEL_NAME]['test_accuracy']

FIELDS = [
    ("age", "Age (years)", "e.g. 54"),
    ("sex", "Sex (1 = Male, 0 = Female)", "0 or 1"),
    ("cp", "Chest Pain Type (0-3)", "0,1,2, or 3"),
    ("trestbps", "Resting Blood Pressure (mm Hg)", "e.g. 130"),
    ("chol", "Serum Cholesterol (mg/dl)", "e.g. 246"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", "0 or 1"),
    ("restecg", "Resting ECG Result (0-2)", "0,1, or 2"),
    ("thalach", "Max Heart Rate Achieved", "e.g. 150"),
    ("exang", "Exercise Induced Angina (1=Yes, 0=No)", "0 or 1"),
    ("oldpeak", "ST Depression (exercise vs rest)", "e.g. 1.0"),
    ("slope", "Slope of Peak Exercise ST Segment (0-2)", "0,1, or 2"),
    ("ca", "Number of Major Vessels Colored (0-3)", "0,1,2, or 3"),
    ("thal", "Thalassemia (1=Normal,2=Fixed Defect,3=Reversible Defect)", "1,2, or 3"),
]

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    probability = None
    error = None
    form_values = {}

    if request.method == 'POST':
        try:
            for key, _, _ in FIELDS:
                form_values[key] = request.form.get(key, '')

            input_vector = [float(form_values[key]) for key, _, _ in FIELDS]
            input_df = pd.DataFrame([input_vector], columns=FEATURE_ORDER)
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            result = "At Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
            probability = round(proba * 100, 1)

        except (ValueError, KeyError):
            error = "Please fill in all fields with valid numeric values."

    return render_template(
        'index.html',
        fields=FIELDS,
        result=result,
        probability=probability,
        error=error,
        form_values=form_values,
        model_name=BEST_MODEL_NAME,
        model_acc=round(BEST_MODEL_ACC * 100, 1)
    )

if __name__ == '__main__':
    app.run(debug=True)
