import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- 1. SETUP PAGE ---
st.set_page_config(
    page_title="Telco Retention Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("cleaned_data.csv")
    model = joblib.load("models/churn_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, model, model_columns, scaler


df, model, model_columns, scaler = load_data_and_model()

# --- 3. SIDEBAR: INPUT USER ---
st.sidebar.header("Customer Inspector")
st.sidebar.markdown(
    "**Instructions:** Adjust customer parameters below to simulate different risk profiles and test intervention outcomes."
)


def user_input_features(df):
    # Choose Internet Service first
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["Fiber optic", "DSL", "No"]
    )

    # Filter Dataframe based on user choice
    subset = df[df["InternetService"] == internet_service]
    real_min = float(subset["MonthlyCharges"].min())
    real_max = float(subset["MonthlyCharges"].max())
    real_mean = float(subset["MonthlyCharges"].median())
    monthly_charges = st.sidebar.number_input(
        f"Monthly Charges ($) - (Data Range: {real_min} - {real_max})",
        min_value=real_min,
        max_value=real_max,
        value=real_mean,
    )

    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    total_charges = tenure * monthly_charges

    contract = st.sidebar.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    online_security = st.sidebar.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method,
        "OnlineSecurity": online_security,
        "PaperlessBilling": paperless_billing,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "No",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features(df)


# --- 4. PREPROCESSING INPUT ---
def preprocess_input(input_df, model_columns, scaler):
    # 1. Numeric Columns
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # 2. Scaling
    input_df_scaled = input_df.copy()
    input_df_scaled[num_cols] = scaler.transform(input_df[num_cols])

    # 3. Encoding (Get Dummies)
    input_processed = pd.get_dummies(input_df_scaled)

    # 4. Align Columns
    input_processed = input_processed.reindex(columns=model_columns, fill_value=0)

    return input_processed


input_ready = preprocess_input(input_df, model_columns, scaler)

# --- 5. MAIN PAGE ---
# Header
st.title("Telco Customer Retention Engine")
st.markdown(
    """
This application helps the marketing team identify high-risk customers 
and simulate the most profitable retention strategies.
"""
)
st.markdown(
    """
**Objective:** Reduce revenue leakage by proactively identifying high-risk customers and optimizing retention spend.
This dashboard simulates intervention strategies to maximize return on retention investment.
"""
)
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Risk Assessment")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")
    threshold = st.sidebar.slider("Churn Threshold", 0.0, 1.0, 0.5, step=0.05)
    st.sidebar.caption("Adjust the threshold for classifying churn risk.")

    prediction_proba = model.predict_proba(input_ready)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0

    if prediction == 1:
        st.error(f"HIGH RISK CHURN (Probability: {prediction_proba:.1%})")
        st.markdown(
            "**Recommendation:** This customer is very likely to churn. Offer incentives immediately."
        )
    else:
        st.success(f"LOYAL CUSTOMER (Probability: {prediction_proba:.1%})")
        st.markdown(
            "**Recommendation:** Customer is safe. Avoid aggressive retention offers to prevent unnecessary subsidy expenses."
        )
    fig_prob, ax_prob = plt.subplots(figsize=(10, 2))
    ax_prob.barh(
        ["Churn Risk"],
        [prediction_proba],
        color="#D9534F" if prediction == 1 else "#5CB85C",
    )
    ax_prob.set_xlim(0, 1)
    st.pyplot(fig_prob)

    # --- BUSINESS SIMULATION ---
    if prediction == 1:
        st.subheader("Retention Simulator")
        st.markdown("If we offer a retention discount, is our profit safe?")

        # Retention Cost Slider
        cost = st.slider("Promotion Cost/Discount per User ($)", 0, 100, 50)
        success_rate = st.slider(
            "Retention Success Estimate (%)", 0.0, 1.0, 0.5, step=0.05
        )
        annual_revenue = input_df["MonthlyCharges"].values[0] * 12
        expected_value = (annual_revenue * success_rate) - cost

        st.write(f"Annual Revenue at Risk: **${annual_revenue:.2f}**")
        st.write(f"Intervention Cost: **${cost:.2f}**")

        if expected_value > 0:
            st.success(f"**Projected Net Benefit: +${expected_value:.2f}**")
            st.caption(
                f"With assumption {success_rate:.0%} customers accepting this offer."
            )
        else:
            st.error(f"**Projected Net Loss: -${abs(expected_value):.2f}**")
            st.caption(
                "Promotion cost is too high compared to the success probability."
            )

with col2:
    st.subheader("Key Risk Drivers")
    st.markdown("Observed risk factors based on historical patterns:")

    drivers = []
    if input_df["Contract"].values[0] == "Month-to-month":
        drivers.append(
            "❌ Contract: Month-to-month contracts have historically high churn rates."
        )
    if input_df["InternetService"].values[0] == "Fiber optic":
        drivers.append("❌ Service: Fiber Optic users show higher attrition rates.")
    if input_df["tenure"].values[0] < 12:
        drivers.append("❌ Tenure: New User (< 1 Year)")
    if input_df["PaymentMethod"].values[0] == "Electronic check":
        drivers.append(
            "❌ Payment: Electronic check users are statistically more likely to leave."
        )

    if len(drivers) > 0:
        for d in drivers:
            st.markdown(d)
    else:
        st.markdown("No major risk factors detected.")

    st.markdown("---")
    st.info("ℹ️ **Model Info:** Random Forest Classifier (Recall: ~62%)")

# --- 6. RECOMMENDATION ---
st.markdown("---")
st.subheader("Retention Strategy Recommendations")
st.markdown("Suggested actions to reduce churn risk for the customer:")


def simulate_intervention(input_data, intervention_col, intervention_val):
    input_sim = input_data.copy()
    input_sim[intervention_col] = intervention_val
    input_sim_processed = preprocess_input(input_sim, model_columns, scaler)
    prob_new = model.predict_proba(input_sim_processed)[0][1]
    return prob_new


# Scenario Interventions
interventions = [
    ("Change to 1-Year Contract", "Contract", "One year"),
    ("Change to 2-Year Contract", "Contract", "Two year"),
    ("Add Tech Support", "TechSupport", "Yes"),
    ("Add Online Security", "OnlineSecurity", "Yes"),
]

# Calculate Delta for each scenario
results = []
current_prob = prediction_proba

for label, col, val in interventions:
    if input_df[col].values[0] != val:
        new_prob = simulate_intervention(input_df, col, val)
        delta = current_prob - new_prob
        if delta > 0.01:  # Only show if risk reduction > 1%
            results.append((label, new_prob, delta))

# Show Best Recommendation
if len(results) > 0:
    # Sort by largest risk reduction (Delta)
    results.sort(key=lambda x: x[2], reverse=True)

    best_action = results[0]

    st.success(f"**Main Recommendation:** {best_action[0]}")
    st.metric(
        label="Potential Risk Reduction",
        value=f"{best_action[1]:.1%}",
        delta=f"-{best_action[2]:.1%}",
        delta_color="inverse",
    )

    with st.expander("View Alternative Strategies"):
        for res in results[1:]:
            st.write(
                f"• **{res[0]}**: Risk becomes {res[1]:.1%} (Impact: -{res[2]:.1%})"
            )
else:
    st.info(
        "This customer already has an optimal configuration (or the risk is already very low)."
    )
