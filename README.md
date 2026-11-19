# Telco Customer Churn Prediction

An end-to-end Machine Learning web application that predicts whether a telecom customer is likely to churn based on tenure, billing, contract type, internet service, and payment method.

## Tech Stack
- **Language:** Python
- **Data Processing & ML:** Pandas, NumPy, Scikit-Learn (`RandomForestClassifier`, `StandardScaler`)
- **Model Persistence:** Joblib
- **Web Interface:** Streamlit

## Project Structure
- `train_model.py`: Loads and cleans the Telco dataset, encodes features, scales data, trains/evaluates the model, and saves artifacts (`.pkl`).
- `app.py`: Interactive Streamlit web app for real-time churn prediction.
- `requirements.txt`: Python dependencies required to run the project.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt