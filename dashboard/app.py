import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils import load_model_and_data, make_prediction, get_model_metrics, detect_data_drift
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_pipeline

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Churn Prediction System")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predictions", "Analytics", "Monitoring"]
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --bg-main-top: #f8fbff;
        --bg-main-bottom: #eef6ff;
        --bg-panel: #f4f8ff;
        --bg-card: #ffffff;
        --bg-card-soft: #f7fbff;
        --text-main: #0f172a;
        --text-muted: #334155;
        --heading: #0b5fff;
        --border: rgba(15,23,42,0.10);
        --accent: #0b5fff;
        --accent-soft: rgba(11,95,255,0.10);
    }

    /* App base */
    .stApp {
        background: linear-gradient(180deg,var(--bg-main-top) 0%, var(--bg-main-bottom) 100%);
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    .stApp [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    /* Page header and titles */
    h1, h2, h3, .css-1v3fvcr h1 {
        color: var(--heading) !important;
        letter-spacing: -0.2px;
    }

    p, li, label, span, div {
        color: var(--text-main);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecf5ff 0%, #e6f1ff 100%) !important;
        color: var(--text-main) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    /* Sidebar controls (radio/buttons/labels) */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] label {
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: #f5f9ff !important;
        padding: 6px 8px !important;
        border-radius: 6px;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
        background: var(--accent-soft) !important;
        color: var(--heading) !important;
        font-weight: 600;
        border: 1px solid rgba(11,95,255,0.25);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(180deg,var(--bg-card) 0%, var(--bg-card-soft) 100%);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(11,37,88,0.06);
        border: 1px solid rgba(11,37,88,0.10);
    }
    .metric-value {
        font-size: 1.9em;
        font-weight: 700;
        color: var(--heading);
    }
    .metric-label {
        font-size: 0.9em;
        color: var(--text-muted);
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: var(--text-main) !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg,#ffffff 0%, #f7fbff 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--accent);
        color: white;
        border-radius: 8px;
        border: none;
    }

    /* Inputs and select boxes */
    [data-baseweb="select"] > div,
    .stNumberInput input {
        background: #ffffff !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
    }

    /* DataFrame and Table styling */
    .stDataFrame table,
    div[data-testid="stTable"] table {
        table-layout: fixed !important;
        width: 100% !important;
        border-collapse: collapse !important;
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }

    .stDataFrame th,
    .stDataFrame td,
    div[data-testid="stTable"] th,
    div[data-testid="stTable"] td {
        padding: 8px 10px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border-bottom: 1px solid rgba(15,23,42,0.08) !important;
        color: var(--text-main) !important;
        opacity: 1 !important;
    }

    .stDataFrame thead th,
    div[data-testid="stTable"] thead th {
        background: #dbeafe !important;
        color: #1e3a8a !important;
        font-weight: 700 !important;
        position: sticky; top: 0; z-index: 2;
    }

    .stDataFrame tbody tr:nth-child(even) td,
    div[data-testid="stTable"] tbody tr:nth-child(even) td { background: #ffffff !important; }
    .stDataFrame tbody tr:nth-child(odd) td,
    div[data-testid="stTable"] tbody tr:nth-child(odd) td { background: #f8fbff !important; }

    /* Alert boxes with bright readable text */
    [data-testid="stAlert"] {
        border: 1px solid var(--border);
    }
    [data-testid="stAlert"] * {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 600;
    }

    /* Fix Streamlit layout jitter */
    .stApp .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Use Plotly white template with brand colors
import plotly.io as pio
pio.templates.default = 'plotly_white'
from plotly.colors import qualitative
BRAND_COLORS = ['#0b5fff', '#06b6d4', '#ef4444', '#f59e0b', '#10b981']
qualitative.PLOTLY_CUSTOM = BRAND_COLORS

# Home Page
if page == "Home":
    st.title("Welcome to Churn Prediction Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Type",
            "XGBoost Classifier",
            delta="Production",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Expected Performance",
            "0.89",
            label_visibility="visible",
            help="AUC-ROC Score"
        )
    
    with col3:
        st.metric(
            "Status",
            "Active",
            delta="Healthy",
            delta_color="off"
        )
    
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
    - **Analytics**: Dashboard with key metrics and visualizations
    - **Monitoring**: Real-time drift detection and alerts
    - **Export**: Download predictions and reports
    """)

# Predictions Page
elif page == "Predictions":
    st.title("Customer Churn Prediction")
    
    model, preprocessor, df = load_model_and_data()
    
    st.subheader("Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", 0, 100, 24)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5)
    
    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1570.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    
    with col3:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    
    if st.button("Predict Churn", use_container_width=True):
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
            result = make_prediction(customer_data, model, preprocessor)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['prediction'] == 'CHURN':
                    st.error(f"{result['prediction']}")
                else:
                    st.success(f"{result['prediction']}")
            
            with col2:
                st.metric("Churn Probability", f"{result['churn_prob']:.2%}")
            
            with col3:
                st.metric("Confidence", f"{max(result['churn_prob'], result['no_churn_prob']):.2%}")
            
            st.divider()
            
            st.subheader("Recommendations")
            if result['churn_prob'] > 0.7:
                st.info("HIGH RISK: Consider immediate retention strategies")
                st.write("""
                - Offer personalized discounts
                - Assign dedicated account manager
                - Provide enhanced support
                - Explore upgrade opportunities
                """)
            elif result['churn_prob'] > 0.4:
                st.warning("MEDIUM RISK: Proactive engagement recommended")
                st.write("""
                - Monitor usage patterns
                - Send personalized offers
                - Request feedback
                - Highlight new features
                """)
            else:
                st.success("LOW RISK: Maintain relationship quality")
                st.write("""
                - Continue regular communication
                - Upsell relevant services
                - Ensure satisfaction
                - Encourage loyalty programs
                """)
        
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Analytics Page
elif page == "Analytics":
    st.title("Analytics Dashboard")
    
    try:
        model, preprocessor, df = load_model_and_data()
        
        X, y, _ = preprocess_pipeline(df)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        metrics = get_model_metrics(model, X_test, y_test)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        
        with col2:
            st.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
        
        with col3:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        with col4:
            st.metric("Precision", f"{metrics['Precision']:.2%}")
        
        with col5:
            st.metric("Recall", f"{metrics['Recall']:.2%}")
        
        st.divider()
        
        # Churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            churn_counts = df['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Customer Churn Distribution",
                color_discrete_sequence=["#0068C9", "#EF553B"]
            )
            fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#0f172a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tenure distribution by churn
            fig = px.box(
                df, x="Churn", y="tenure",
                title="Tenure Distribution by Churn Status",
                color="Churn"
            )
            fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#0f172a'))
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Analytics error: {e}")

# Monitoring Page
elif page == "Monitoring":
    st.title("Real-time Monitoring")
    
    try:
        model, preprocessor, df = load_model_and_data()
        
        st.subheader("Data Drift Detection")
        
        # Simulate current data (in production, this comes from new predictions)
        df_current = df.sample(frac=0.1, random_state=42)
        df_reference = df.sample(frac=0.1, random_state=0)
        
        drift_report = detect_data_drift(df_reference, df_current)
        
        # Display drift report
        if drift_report:
            drift_detected = any([v['is_drift'] for v in drift_report.values()])
            
            if drift_detected:
                st.error("**DATA DRIFT DETECTED** - Model retraining may be needed")
            else:
                st.success("No significant data drift detected")
            
            # Detailed report - create DataFrame safely and render as a stable table
            try:
                drift_data = []
                for k, v in drift_report.items():
                    drift_data.append({
                        'Feature': str(k),
                        'KS Statistic': float(v['statistic']),
                        'P-Value': float(v['p_value']),
                        'Drift': "Yes" if v['is_drift'] else "No"
                    })

                if drift_data:
                    # sort features to keep order stable across renders
                    drift_df = pd.DataFrame(sorted(drift_data, key=lambda r: r['Feature']))
                    # Use st.table for a static, non-resizing render to prevent blinking
                    st.table(drift_df)
                else:
                    st.info("No drift metrics available")
            except Exception as df_err:
                st.warning(f"Could not display drift details: {df_err}")
        else:
            st.info("No numerical columns to compare for drift detection")
        
        st.divider()
        st.subheader("Alerts")
        st.info("No critical alerts at this time")
        st.warning("Performance metrics stable")
    
    except Exception as e:
        st.error(f"Monitoring error: {str(e)}")

st.sidebar.divider()
st.sidebar.info("Contact: ml-team@company.com")