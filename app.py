import joblib
import pandas as pd
import streamlit as st

# Load saved artifacts
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊")
st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn:")

col1, col2 = st.columns(2)

with col1:
  tenure = st.slider("Tenure (Months)", 0, 72, 12)
  monthly_charges = st.number_input(
      "Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0
  )
  total_charges = st.number_input(
      "Total Charges ($)", min_value=10.0, max_value=9000.0, value=840.0
  )

with col2:
  contract = st.selectbox(
      "Contract", ["Month-to-month", "One year", "Two year"]
  )
  internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
  payment = st.selectbox(
      "Payment Method",
      [
          "Electronic check",
          "Mailed check",
          "Bank transfer (automatic)",
          "Credit card (automatic)",
      ],
  )

if st.button("Predict Churn", type="primary"):
  input_dict = {
      "tenure": tenure,
      "MonthlyCharges": monthly_charges,
      "TotalCharges": total_charges,
      f"Contract_{contract}": 1,
      f"InternetService_{internet}": 1,
      f"PaymentMethod_{payment}": 1,
  }

  input_df = pd.DataFrame([input_dict])
  input_df = input_df.reindex(columns=model_columns, fill_value=0)
  input_scaled = scaler.transform(input_df)

  prediction = model.predict(input_scaled)[0]
  prob = model.predict_proba(input_scaled)[0][1] * 100

  st.divider()
  if prediction == 1:
    st.error(f"⚠️ High Churn Risk ({prob:.1f}% probability)")
  else:
    st.success(f"✅ Customer Likely to Stay (Churn probability: {prob:.1f}%)")