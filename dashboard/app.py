import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils import (
    load_model_and_data,
    make_prediction,
    get_model_metrics,
    detect_data_drift
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY
# ─────────────────────────────────────────────────────────────────────────────
pio.templates.default = "plotly_white"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* MAIN BACKGROUND */
.stApp {
    background-color: #f3f6fb;
}

/* REMOVE STREAMLIT TOP SPACE */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* TITLE */
.main-title {
    font-size: 56px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 35px;
}

/* SECTION TITLE */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* DASHBOARD CARD */
.dashboard-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 22px;
    min-height: 130px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
}

.dashboard-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.06);
}

/* CARD LABEL */
.card-title {
    font-size: 17px;
    color: #334155;
    margin-bottom: 16px;
    font-weight: 500;
}

/* CARD VALUE */
.card-value {
    font-size: 42px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}

/* CARD SUBTEXT */
.card-subtext {
    margin-top: 10px;
    font-size: 17px;
    color: #475569;
}

/* GREEN TEXT */
.green-text {
    color: #16a34a;
    font-weight: 600;
}

/* FEATURE LIST */
.feature-box {
    margin-top: 10px;
}

.feature-box ul {
    padding-left: 24px;
}

.feature-box li {
    margin-bottom: 14px;
    font-size: 22px;
    color: #0f172a;
}

/* BUTTON */
.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
}

/* INPUTS */
.stTextInput input,
.stNumberInput input,
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid #d1d5db !important;
}

/* METRICS */
[data-testid="metric-container"] {
    border-radius: 14px;
    padding: 14px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Churn Prediction System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predictions", "Analytics", "Monitoring"]
)

st.sidebar.divider()

st.sidebar.info("Contact: ml-team@company.com")

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown(
        '<div class="main-title">Welcome to Churn Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────────────────
    # TOP ROW
    # ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Model Type</div>
            <div class="card-value">XGBoost Classifier</div>
            <div class="card-subtext">Production</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Expected Performance</div>
            <div class="card-value">0.89</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Status</div>
            <div class="card-value">Active</div>
            <div class="card-subtext green-text">Healthy</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")

    # ─────────────────────────────────────────────────────────
    # QUICK STATS
    # ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Quick Stats</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Total Customers</div>
            <div class="card-value">7,043</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Churn Rate</div>
            <div class="card-value">16.3%</div>
            <div class="card-subtext green-text">↓ -2.1%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Avg Tenure</div>
            <div class="card-value">32.4 months</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">Revenue at Risk</div>
            <div class="card-value">$234.5K</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")

    # ─────────────────────────────────────────────────────────
    # FEATURES
    # ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Features</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="feature-box">
    <ul>
        <li><b>Single Prediction:</b> Predict churn for individual customers</li>
        <li><b>Batch Predictions:</b> Process multiple customers at once</li>
        <li><b>Analytics:</b> Dashboard with key metrics and visualizations</li>
        <li><b>Monitoring:</b> Real-time drift detection and alerts</li>
        <li><b>Export:</b> Download predictions and reports</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictions":

    st.title("Customer Churn Prediction")

    try:
        model, preprocessor, df = load_model_and_data()

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            0,
            100,
            24
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            0.0,
            200.0,
            65.5
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            1570.0
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col3:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

    if st.button("Predict Churn"):

        customer_data = {
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': contract,
            'PhoneService': phone_service,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'TechSupport': tech_support
        }

        try:

            result = make_prediction(
                customer_data,
                model,
                preprocessor
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                if result['prediction'] == "CHURN":
                    st.error("🚨 Customer Likely to Churn")
                else:
                    st.success("✅ Customer Retained")

            with col2:
                st.metric(
                    "Churn Probability",
                    f"{result['churn_prob']:.2%}"
                )

            with col3:
                st.metric(
                    "Confidence",
                    f"{max(result['churn_prob'], result['no_churn_prob']):.2%}"
                )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
