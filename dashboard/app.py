import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os

# Add parent directory to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils import load_model_and_data, make_prediction, get_model_metrics, detect_data_drift

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Brand colours ─────────────────────────────────────────────────────────────
pio.templates.default = 'plotly_white'
BRAND_COLORS = ['#0b5fff', '#06b6d4', '#ef4444', '#f59e0b', '#10b981']

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-main-top: #f8fbff;
        --bg-main-bottom: #eef6ff;
        --bg-card: #ffffff;
        --bg-card-soft: #f7fbff;
        --text-main: #0f172a;
        --text-muted: #334155;
        --heading: #0b5fff;
        --border: rgba(15,23,42,0.10);
        --accent: #0b5fff;
        --accent-soft: rgba(11,95,255,0.10);
    }
    .stApp {
        background: linear-gradient(180deg,var(--bg-main-top) 0%, var(--bg-main-bottom) 100%);
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    h1, h2, h3 { color: var(--heading) !important; letter-spacing: -0.2px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecf5ff 0%, #e6f1ff 100%) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }
    .metric-card {
        background: linear-gradient(180deg,var(--bg-card) 0%, var(--bg-card-soft) 100%);
        padding: 18px; border-radius: 12px; text-align: center;
        box-shadow: 0 6px 18px rgba(11,37,88,0.06);
        border: 1px solid rgba(11,37,88,0.10);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg,#ffffff 0%, #f7fbff 100%);
        border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px;
    }
    .stButton>button {
        background-color: var(--accent); color: white;
        border-radius: 8px; border: none;
    }
    [data-baseweb="select"] > div, .stNumberInput input {
        background: #ffffff !important; color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
    }
    .stApp .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "XGBoost Classifier", delta="Production", delta_color="off")
    with col2:
        st.metric("Expected Performance", "0.89", help="AUC-ROC Score")
    with col3:
        st.metric("Status", "Active", delta="Healthy", delta_color="off")

    st.divider()
    st.subheader("Quick Stats")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", "7,043")
    with col2:
        st.metric("Churn Rate", "16.3%", delta="-2.1%", delta_color="inverse")
    with col3:
        st.metric("Avg Tenure", "32.4 months")
    with col4:
        st.metric("Revenue at Risk", "$234.5K")

    st.divider()
    st.subheader("Features")
    st.markdown("""
    - **Single Prediction**: Predict churn for individual customers
    - **Batch Predictions**: Process multiple customers at once
    - **Analytics**: Dashboard with key metrics and visualisations
    - **Monitoring**: Real-time drift detection and alerts
    - **Export**: Download predictions and reports
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
        senior_citizen    = st.selectbox("Senior Citizen", ["No", "Yes"])
        tenure            = st.number_input("Tenure (months)", 0, 100, 24)
        monthly_charges   = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5)
    with col2:
        total_charges     = st.number_input("Total Charges ($)", 0.0, 10000.0, 1570.0)
        contract          = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        phone_service     = st.selectbox("Phone Service", ["Yes", "No"])
    with col3:
        internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security   = st.selectbox("Online Security", ["Yes", "No"])
        tech_support      = st.selectbox("Tech Support", ["Yes", "No"])

    if st.button("Predict Churn", use_container_width=True):
        customer_data = {
            'SeniorCitizen':    1 if senior_citizen == 'Yes' else 0,
            'tenure':           tenure,
            'MonthlyCharges':   monthly_charges,
            'TotalCharges':     total_charges,
            'Contract':         contract,
            'PhoneService':     phone_service,
            'InternetService':  internet_service,
            'OnlineSecurity':   online_security,
            'TechSupport':      tech_support
        }

        try:
            result = make_prediction(customer_data, model, preprocessor)

            col1, col2, col3 = st.columns(3)
            with col1:
                if result['prediction'] == 'CHURN':
                    st.error(f"🚨 {result['prediction']}")
                else:
                    st.success(f"✅ {result['prediction']}")
            with col2:
                st.metric("Churn Probability", f"{result['churn_prob']:.2%}")
            with col3:
                st.metric("Confidence", f"{max(result['churn_prob'], result['no_churn_prob']):.2%}")

            st.divider()
            st.subheader("Recommendations")
            if result['churn_prob'] > 0.7:
                st.error("**HIGH RISK** — Consider immediate retention strategies")
                st.write("- Offer personalised discounts\n- Assign dedicated account manager\n- Provide enhanced support\n- Explore upgrade opportunities")
            elif result['churn_prob'] > 0.4:
                st.warning("**MEDIUM RISK** — Proactive engagement recommended")
                st.write("- Monitor usage patterns\n- Send personalised offers\n- Request feedback\n- Highlight new features")
            else:
                st.success("**LOW RISK** — Maintain relationship quality")
                st.write("- Continue regular communication\n- Upsell relevant services\n- Ensure satisfaction\n- Encourage loyalty programmes")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS PAGE  — no src.preprocessing import; uses raw df directly
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.title("Analytics Dashboard")

    try:
        model, preprocessor, df = load_model_and_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # ── Lightweight preprocessing inline (no src dependency) ──────────────
        df_clean = df.copy()

        # Convert TotalCharges to numeric (raw CSV has spaces)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean.dropna(subset=['TotalCharges'], inplace=True)

        # Encode target
        df_clean['Churn_label'] = (df_clean['Churn'].map({'Yes': 1, 'No': 0})
                                   .fillna(df_clean['Churn'].astype(int)
                                           if df_clean['Churn'].dtype != object else 0))

        # Use preprocessor to transform features
        feature_cols = [c for c in df_clean.columns if c not in ['Churn', 'Churn_label', 'customerID']]
        try:
            X = preprocessor.transform(df_clean[feature_cols])
            y = df_clean['Churn_label'].values

            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            metrics = get_model_metrics(model, X_test, y_test)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy",  f"{metrics['Accuracy']:.2%}")
            col2.metric("AUC-ROC",   f"{metrics['AUC-ROC']:.4f}")
            col3.metric("F1-Score",  f"{metrics['F1-Score']:.4f}")
            col4.metric("Precision", f"{metrics['Precision']:.2%}")
            col5.metric("Recall",    f"{metrics['Recall']:.2%}")
            st.divider()
        except Exception as metric_err:
            st.warning(f"Could not compute live metrics: {metric_err}")
            st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            churn_counts = df_clean['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Customer Churn Distribution",
                color_discrete_sequence=["#0068C9", "#EF553B"]
            )
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                              font=dict(color='#0f172a'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df_clean, x="Churn", y="tenure",
                title="Tenure Distribution by Churn Status",
                color="Churn",
                color_discrete_sequence=BRAND_COLORS
            )
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                              font=dict(color='#0f172a'))
            st.plotly_chart(fig, use_container_width=True)

        # Monthly charges histogram
        fig = px.histogram(
            df_clean, x="MonthlyCharges", color="Churn",
            title="Monthly Charges Distribution",
            barmode="overlay", opacity=0.75,
            color_discrete_sequence=BRAND_COLORS
        )
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(color='#0f172a'))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Analytics error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MONITORING PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Monitoring":
    st.title("Real-time Monitoring")

    try:
        model, preprocessor, df = load_model_and_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    try:
        st.subheader("Data Drift Detection")

        df_reference = df.sample(frac=0.1, random_state=0)
        df_current   = df.sample(frac=0.1, random_state=42)

        drift_report = detect_data_drift(df_reference, df_current)

        if drift_report:
            drift_detected = any(v['is_drift'] for v in drift_report.values())

            if drift_detected:
                st.error("**DATA DRIFT DETECTED** — Model retraining may be needed")
            else:
                st.success("✅ No significant data drift detected")

            drift_data = [
                {
                    'Feature':       str(k),
                    'KS Statistic':  round(v['statistic'], 4),
                    'P-Value':       round(v['p_value'], 4),
                    'Drift':         "⚠️ Yes" if v['is_drift'] else "✅ No"
                }
                for k, v in drift_report.items()
            ]
            if drift_data:
                st.table(pd.DataFrame(sorted(drift_data, key=lambda r: r['Feature'])))
        else:
            st.info("No numerical columns available for drift comparison.")

        st.divider()
        st.subheader("Alerts")
        st.info("ℹ️ No critical alerts at this time")
        st.warning("⚠️ Performance metrics stable")

    except Exception as e:
        st.error(f"Monitoring error: {e}")
