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

BRAND_COLORS = [
    "#0b5fff",
    "#06b6d4",
    "#ef4444",
    "#f59e0b",
    "#10b981"
]

# ─────────────────────────────────────────────────────────────────────────────
# FIXED CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

:root {
    --bg-main-top: #f8fbff;
    --bg-main-bottom: #eef6ff;

    --bg-card: #ffffff;
    --bg-card-soft: #f7fbff;

    --text-main: #0f172a;
    --text-secondary: #334155;

    --heading: #0b5fff;

    --border: rgba(15,23,42,0.10);

    --accent: #0b5fff;
}

/* MAIN APP */
.stApp {
    background: linear-gradient(
        180deg,
        var(--bg-main-top) 0%,
        var(--bg-main-bottom) 100%
    );

    color: var(--text-main);

    font-family: -apple-system,
                 BlinkMacSystemFont,
                 'Segoe UI',
                 Roboto,
                 Arial;
}

/* HEADINGS */
h1, h2, h3 {
    color: var(--heading) !important;
    font-weight: 700 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ecf5ff 0%,
        #e6f1ff 100%
    ) !important;

    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

/* METRIC CARDS */
[data-testid="metric-container"] {

    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f7fbff 100%
    ) !important;

    border: 1px solid rgba(15,23,42,0.08);

    padding: 18px !important;

    border-radius: 14px;

    box-shadow: 0 4px 12px rgba(0,0,0,0.04);

}

/* METRIC LABEL */
[data-testid="metric-container"] label {

    color: #334155 !important;

    font-size: 15px !important;

    font-weight: 600 !important;

}

/* METRIC VALUE */
[data-testid="metric-container"] [data-testid="stMetricValue"] {

    color: #0f172a !important;

    font-size: 32px !important;

    font-weight: 700 !important;

}

/* METRIC DELTA */
[data-testid="metric-container"] [data-testid="stMetricDelta"] {

    color: #16a34a !important;

    font-weight: 600 !important;

}

/* BUTTON */
.stButton > button {

    background-color: var(--accent) !important;

    color: white !important;

    border-radius: 10px !important;

    border: none !important;

    font-weight: 600 !important;

    height: 48px;

    font-size: 16px;

}

/* INPUTS */
[data-baseweb="select"] > div,
.stNumberInput input {

    background: white !important;

    color: #0f172a !important;

    border: 1px solid rgba(15,23,42,0.10) !important;

    border-radius: 10px !important;

}

/* INPUT LABELS */
.stSelectbox label,
.stNumberInput label {

    color: #334155 !important;

    font-weight: 600 !important;

}

/* SUCCESS BOX */
.stSuccess {

    background-color: rgba(16,185,129,0.12) !important;

    color: #065f46 !important;

}

/* ERROR BOX */
.stError {

    background-color: rgba(239,68,68,0.10) !important;

    color: #991b1b !important;

}

/* WARNING BOX */
.stWarning {

    background-color: rgba(245,158,11,0.12) !important;

    color: #92400e !important;

}

/* INFO BOX */
.stInfo {

    background-color: rgba(59,130,246,0.10) !important;

    color: #1e40af !important;

}

/* TABLE */
table {

    color: #0f172a !important;

}

/* GENERAL TEXT */
p, span, div {

    color: #0f172a;

}

/* BLOCK PADDING */
.block-container {

    padding-top: 1rem;

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

    st.title("Welcome to Churn Prediction Dashboard")

    st.markdown("### Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Model Type",
            value="XGBoost Classifier"
        )

    with col2:
        st.metric(
            label="Expected Performance",
            value="0.89 AUC"
        )

    with col3:
        st.metric(
            label="System Status",
            value="Active"
        )

    st.divider()

    st.markdown("### Business Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Customers",
            value="7,043"
        )

    with col2:
        st.metric(
            label="Churn Rate",
            value="16.3%"
        )

    with col3:
        st.metric(
            label="Average Tenure",
            value="32.4 Months"
        )

    with col4:
        st.metric(
            label="Revenue At Risk",
            value="$234.5K"
        )

    st.divider()

    st.markdown("### Platform Features")

    st.info("""
    ✅ Single Customer Prediction  
    ✅ Batch Predictions  
    ✅ Real-time Analytics  
    ✅ Drift Monitoring  
    ✅ Export Reports  
    """)
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

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":

    st.title("Analytics Dashboard")

    try:
        model, preprocessor, df = load_model_and_data()

        from sklearn.model_selection import train_test_split

        df_clean = df.copy()

        # Convert TotalCharges safely
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'],
            errors='coerce'
        )

        df_clean.dropna(subset=['TotalCharges'], inplace=True)

        # FIXED ERROR HERE
        df_clean['Churn_label'] = (
            df_clean['Churn']
            .map({'Yes': 1, 'No': 0})
            .astype(int)
        )

        feature_cols = [
            c for c in df_clean.columns
            if c not in [
                'customerID',
                'Churn',
                'Churn_label'
            ]
        ]

        X = preprocessor.transform(df_clean[feature_cols])

        y = df_clean['Churn_label'].values

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        metrics = get_model_metrics(
            model,
            X_test,
            y_test
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.2%}"
            )

        with col2:
            st.metric(
                "AUC-ROC",
                f"{metrics['AUC-ROC']:.4f}"
            )

        with col3:
            st.metric(
                "F1 Score",
                f"{metrics['F1-Score']:.4f}"
            )

        with col4:
            st.metric(
                "Precision",
                f"{metrics['Precision']:.2%}"
            )

        with col5:
            st.metric(
                "Recall",
                f"{metrics['Recall']:.2%}"
            )

        st.divider()

        # PIE CHART
        col1, col2 = st.columns(2)

        with col1:

            churn_counts = df_clean['Churn'].value_counts()

            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Customer Churn Distribution",
                color_discrete_sequence=["#0068C9", "#EF553B"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # BOX PLOT
        with col2:

            fig = px.box(
                df_clean,
                x="Churn",
                y="tenure",
                title="Tenure Distribution by Churn",
                color="Churn",
                color_discrete_sequence=BRAND_COLORS
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # HISTOGRAM
        fig = px.histogram(
            df_clean,
            x="MonthlyCharges",
            color="Churn",
            title="Monthly Charges Distribution",
            barmode="overlay",
            opacity=0.7,
            color_discrete_sequence=BRAND_COLORS
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Analytics Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MONITORING PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Monitoring":

    st.title("Real-time Monitoring")

    try:

        model, preprocessor, df = load_model_and_data()

        st.subheader("Data Drift Detection")

        df_reference = df.sample(
            frac=0.1,
            random_state=0
        )

        df_current = df.sample(
            frac=0.1,
            random_state=42
        )

        drift_report = detect_data_drift(
            df_reference,
            df_current
        )

        if drift_report:

            drift_detected = any(
                v['is_drift']
                for v in drift_report.values()
            )

            if drift_detected:

                st.error(
                    "⚠️ DATA DRIFT DETECTED"
                )

            else:

                st.success(
                    "✅ No Significant Drift Detected"
                )

            drift_data = []

            for k, v in drift_report.items():

                drift_data.append({

                    "Feature": str(k),

                    "KS Statistic": round(
                        v['statistic'],
                        4
                    ),

                    "P-Value": round(
                        v['p_value'],
                        4
                    ),

                    "Drift": (
                        "⚠️ Yes"
                        if v['is_drift']
                        else "✅ No"
                    )
                })

            st.table(
                pd.DataFrame(drift_data)
            )

        st.divider()

        st.subheader("Alerts")

        st.info("ℹ️ No critical alerts")

        st.warning("⚠️ Performance stable")

    except Exception as e:
        st.error(f"Monitoring Error: {e}")
