# Cardiac Risk Screener — End-to-End ML Web App

🔗 **Live Demo:** https://heart-health-classifier-flask.onrender.com

A machine learning web app that predicts heart disease risk from 13 clinical measurements. I built this to go beyond a Jupyter Notebook project — training a model is one thing, but I wanted to actually ship something a real user could open in a browser and use.

## What it does
You enter basic clinical values — age, blood pressure, cholesterol, chest pain type, and so on — and the app runs them through a trained Random Forest model to estimate whether the patient is at risk of heart disease, along with a confidence score.

## How I built it
I used the UCI Cleveland Heart Disease dataset (303 patient records, 13 features). Instead of picking an algorithm upfront, I trained and compared five different classifiers — Logistic Regression, Random Forest, SVM, KNN, and Decision Tree — using 5-fold cross-validation, and picked the best-performing one based on accuracy, AUC, and consistency across folds.

| Model | Test Accuracy | AUC | 5-Fold CV Mean |
|---|---|---|---|
| **Random Forest (selected)** | **82.0%** | **0.912** | 83.5% (± 4.3%) |
| SVM | 82.0% | 0.883 | 80.2% (± 3.9%) |
| KNN | 82.0% | 0.884 | 81.0% (± 3.4%) |
| Logistic Regression | 80.3% | 0.869 | 83.1% (± 4.1%) |
| Decision Tree | 78.7% | 0.818 | 73.2% (± 3.3%) |

Random Forest won out — best AUC and stable performance across folds.

Once I had a model I trusted, I wrapped it in a Flask app with a simple form-based interface, added input validation so it doesn't break on bad input, and saved the trained model and scaler with joblib so the app doesn't need to retrain every time it runs. Then I deployed it on Render so it's actually usable at a live link, not just something that runs on my machine.

## Tech Stack
Python · scikit-learn · Pandas · NumPy · Flask · joblib · HTML/CSS

## Project Structure
```
├── app.py                     # Flask app — routes and prediction logic
├── train_model.py             # Trains and compares all 5 models, saves the best one
├── heart.csv                  # UCI Cleveland Heart Disease dataset
├── model.pkl                  # Saved Random Forest model
├── scaler.pkl                 # Saved StandardScaler
├── model_results.json         # Comparison metrics across all models
├── templates/index.html       # Web UI
├── requirements.txt
├── Procfile                   # For Render deployment
└── README.md
```

## Running it locally
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

## Data Source
UCI Machine Learning Repository — Cleveland Heart Disease dataset.
Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64, 304-310.

## Disclaimer
This is a student project built for learning and portfolio purposes. It's not a medical device and shouldn't be used for actual diagnosis — the app itself displays this disclaimer too.

## What I'd add next
- Explainability (SHAP) so the app can show which inputs pushed a prediction toward "at risk"
- A larger dataset (the combined Cleveland + Hungarian + Switzerland + VA data, ~920 records) for more robust training
- Stricter input validation to catch biologically implausible values, not just missing ones

---


