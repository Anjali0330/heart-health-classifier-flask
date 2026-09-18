# Cardiac Risk Screener — End-to-End ML Web App

🔗 **Live Demo:** https://heart-health-classifier-flask.onrender.com

A machine learning web application that estimates heart disease risk from 13 clinical measurements, built end-to-end: data → model comparison → Flask web app → deployable.

## Project Summary
Rather than stopping at a Jupyter Notebook model, this project takes a trained classifier all the way to a usable web interface — the same workflow a real data/ML team follows when shipping a model, not just building one.

## What This Project Demonstrates
- **Model selection through comparison, not assumption:** 5 classification algorithms (Logistic Regression, Random Forest, SVM, KNN, Decision Tree) were trained and evaluated with 5-fold cross-validation before picking a winner.
- **Honest evaluation:** reported accuracy is the actual holdout test-set result, validated with cross-validation to check it isn't a fluke of one particular train/test split.
- **Full-stack delivery:** the trained model is wrapped in a Flask web application with a real form-based UI — not just a script that prints a prediction.
- **Production readiness basics:** input validation, error handling for incomplete forms, and a saved/reusable model + scaler pipeline (via joblib) rather than retraining on every request.

## Results
| Model | Test Accuracy | AUC | 5-Fold CV Mean |
|---|---|---|---|
| **Random Forest (selected)** | **82.0%** | **0.912** | 83.5% (± 4.3%) |
| SVM | 82.0% | 0.883 | 80.2% (± 3.9%) |
| KNN | 82.0% | 0.884 | 81.0% (± 3.4%) |
| Logistic Regression | 80.3% | 0.869 | 83.1% (± 4.1%) |
| Decision Tree | 78.7% | 0.818 | 73.2% (± 3.3%) |

Random Forest was selected for its combination of accuracy, the highest AUC (0.912), and stable cross-validation performance.

## Data Source
UCI Machine Learning Repository — Cleveland Heart Disease dataset (303 patient records, 13 clinical features + target). One of the most widely used and well-validated benchmark datasets in medical ML research.
Citation: Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64, 304-310.

## Tech Stack
`Python` · `scikit-learn` · `Pandas` · `NumPy` · `Flask` · `joblib` · `HTML/CSS`

## Repository Structure
```
├── app.py                  # Flask application (routes, prediction logic)
├── train_model.py          # Model training, comparison, and evaluation script
├── heart.csv                # UCI Cleveland Heart Disease dataset
├── model.pkl                 # Saved best-performing model (Random Forest)
├── scaler.pkl                 # Saved StandardScaler for consistent preprocessing
├── model_results.json          # Comparison metrics for all 5 models
├── templates/index.html          # Web UI (form + result display)
├── requirements.txt
├── Procfile                       # For Render/Heroku-style deployment
└── README.md
```

## How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py     # retrains and re-saves model.pkl / scaler.pkl (optional — already included)
python app.py              # starts the Flask dev server
```
Then open http://127.0.0.1:5000 in your browser.

## How to Deploy (Render — free tier)
1. Push this repository to GitHub.
2. Go to https://render.com, sign in with GitHub, and click "New Web Service."
3. Select this repository.
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Deploy. Render will give you a public URL (e.g. `your-app.onrender.com`) — this is the link to put on your resume.

**Note:** Render's free tier spins the app down after inactivity, so the first request after idle time takes ~30-50 seconds to wake up. This is normal and worth mentioning if a recruiter tries the link and it's slow to load the first time.

## ⚠️ Disclaimer
This is a student machine learning project for educational and portfolio purposes only. It is not a certified medical device and must never be used for actual clinical diagnosis or treatment decisions. The UI includes this disclaimer directly for anyone using the live app.

## Future Improvements
- Add SHAP-based explainability so the app shows *which* input features drove a specific prediction
- Expand to the larger multi-source UCI heart disease dataset (Cleveland + Hungarian + Switzerland + VA, ~920 records) for more robust training
- Add input range validation (e.g. flag biologically implausible values) beyond basic type-checking

---

