# Customer Churn Prediction + MLOps Pipeline

A production-ready ML project predicting customer churn with full MLOps implementation.

## 📋 Project Overview

| Aspect | Details |
|--------|---------|
| **Target** | Binary classification (Churn: Yes/No) |
| **Dataset** | Telco Customer Churn (7,043 rows, 21 cols) |
| **Timeline** | 8-10 weeks |
| **Key Skills** | Data Engineering, ML, MLOps, DevOps |
| **Target AUC-ROC** | > 0.85 |

## 🚀 Getting Started (Complete Roadmap)

### Phase 1: Environment Setup (Day 1)

#### Step 1: Install PostgreSQL
- **Windows**: Download from https://www.postgresql.org/download/windows/
- **Version**: 15+ recommended
- **During Installation**: Remember password for postgres user
- **Verify**: Open PowerShell and run:
  ```powershell
  psql --version
  psql -U postgres -c "SELECT version();"
  ```

#### Step 2: Clone & Setup Python
```powershell
cd v:\churn prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Step 3: Create PostgreSQL Database
```powershell
# Connect to PostgreSQL
psql -U postgres

# Inside psql prompt, run:
CREATE DATABASE churn_db;
CREATE SCHEMA raw_data;
CREATE SCHEMA processed_data;

# Verify
\l
\dn
```

#### Step 4: Setup MLflow Server (Optional for now)
```powershell
mlflow ui  # Runs on http://localhost:5000
```

---

## 📝 Phase 2: Data Ingestion & Exploration (Days 2-3)

### Download Datasets:
1. **Telco Customer Churn** (Primary): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - Save to: `v:/churn prediction/data/raw/telco_churn.csv`
   
2. (Optional) IBM HR Analytics: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
3. (Optional) Bank Churn: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

### Files You'll Create:
- `notebooks/01_eda_exploration.ipynb` - Initial data exploration
- `src/data_ingestion.py` - Load CSV → PostgreSQL
- `src/config.py` - Configuration & database connection

**→ See PHASE_2_DATA_INGESTION.md for complete code**

---

## 🔧 Phase 3: Feature Engineering (Days 4-6)

### Tasks:
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Engineer new features (tenure groups, risk scores)
- Handle class imbalance with SMOTE

### Files You'll Create:
- `notebooks/02_feature_engineering.ipynb`
- `src/feature_engineering.py`
- `src/preprocessing.py`

**→ See PHASE_3_FEATURE_ENGINEERING.md for complete code**

---

## 🤖 Phase 4: Model Training (Days 7-9)

### Models to Train:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost (primary)
4. LightGBM
5. Voting Ensemble

### Hyperparameter Tuning:
- Use Optuna (300 trials)
- Track experiments with MLflow
- Target: AUC-ROC > 0.85, F1 > 0.78

### Files You'll Create:
- `notebooks/03_model_training.ipynb`
- `src/model_training.py`
- `src/hyperparameter_tuning.py`

**→ See PHASE_4_MODEL_TRAINING.md for complete code**

---

## 📊 Phase 5: Model Explainability (Days 10-11)

### Techniques:
- SHAP waterfall plots (feature importance)
- LIME local explanations
- Confusion matrix & classification report
- ROC-AUC curve
- Precision-Recall curve
- Calibration curve

### Files You'll Create:
- `notebooks/04_model_explainability.ipynb`
- `src/model_evaluation.py`
- `src/shap_explanations.py`

**→ See PHASE_5_EXPLAINABILITY.md for complete code**

---

## 🐳 Phase 6: API Deployment (Days 12-14)

### Build FastAPI Application:
```bash
# Structure:
api/
  ├── main.py           # FastAPI app
  ├── schemas.py        # Pydantic models
  └── models_loader.py  # Load trained model

# Create Dockerfile
# Test with: docker build -t churn-api . && docker run -p 8000:8000 churn-api
# API will be at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Files You'll Create:
- `api/main.py`
- `api/schemas.py`
- `Dockerfile`
- `docker-compose.yml`
- `api/models_loader.py`

**→ See PHASE_6_API_DEPLOYMENT.md for complete code**

---

## 📈 Phase 7: Dashboard & Monitoring (Days 15-17)

### Option A: Streamlit Dashboard (Quick, Local)
```bash
streamlit run dashboard/app.py
```
- Real-time predictions
- Model performance metrics
- Data drift detection

### Option B: Power BI (Advanced, BI Analyst skill)
- Connect to PostgreSQL
- Build KPIs: Churn Rate, Revenue at Risk
- Schedule daily auto-refresh
- Email distribution

### Files You'll Create:
- `dashboard/app.py`
- `dashboard/pages/` (predictions, analytics, monitoring)

**→ See PHASE_7_DASHBOARD.md for complete code**

---

## 🔄 Phase 8: MLOps & CI/CD (Days 18-20)

### GitHub Setup:
1. Create GitHub account: https://github.com
2. Create repository: `churn-prediction`
3. Clone to local: `git clone ...`
4. Initialize DVC: `dvc init`

### MLOps Components:
- **DVC**: Version datasets (never commit CSVs)
- **MLflow**: Model registry & experiments
- **GitHub Actions**: Automated testing & deployment
- **Great Expectations**: Data validation
- **Evidently AI**: Drift monitoring

### Files You'll Create:
- `.github/workflows/ci-cd.yml` (GitHub Actions)
- `dvc.yaml` (DVC pipeline)
- `great_expectations/` (data validation)
- `tests/` (unit tests)

**→ See PHASE_8_MLOPS.md for complete code**

---

## 📦 Directory Structure

```
churn prediction/
├── data/
│   ├── raw/                    # Original CSVs
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 01_eda_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
├── src/
│   ├── config.py               # DB & settings
│   ├── data_ingestion.py       # CSV → PostgreSQL
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # Model training
│   ├── model_evaluation.py     # Evaluation & SHAP
│   └── preprocessing.py        # Data cleaning
├── api/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py              # Pydantic models
│   └── models_loader.py        # Model serialization
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   └── pages/
├── models/
│   ├── best_model.pkl          # Trained model
│   └── preprocessor.pkl        # Scaler & encoders
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_feature_eng.py
│   └── test_api.py
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions
├── config/
│   └── settings.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🌐 External Resources & URLs

| Resource | URL | Purpose |
|----------|-----|---------|
| **PostgreSQL** | https://www.postgresql.org | Database setup |
| **Kaggle Datasets** | https://www.kaggle.com | Download churn data |
| **FastAPI Docs** | https://fastapi.tiangolo.com | API framework |
| **MLflow Docs** | https://mlflow.org | Experiment tracking |
| **Docker Docs** | https://docs.docker.com | Containerization |
| **GitHub** | https://github.com | Version control |
| **Streamlit Docs** | https://docs.streamlit.io | Dashboard |
| **DVC Docs** | https://dvc.org | Data versioning |
| **SHAP Docs** | https://shap.readthedocs.io | Model explainability |

---

## 🎯 Key Milestones

- ✅ **Week 1**: Data in PostgreSQL, EDA complete
- ✅ **Week 2-3**: Features engineered, SMOTE applied
- ✅ **Week 4-5**: Models trained, AUC-ROC > 0.85
- ✅ **Week 6**: SHAP explanations, calibration
- ✅ **Week 7**: FastAPI running in Docker
- ✅ **Week 8**: Streamlit dashboard live
- ✅ **Week 9-10**: GitHub Actions + MLflow registry + monitoring

---

## 🚀 Quick Start Commands

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run Jupyter
jupyter notebook

# Run API
uvicorn api.main:app --reload

# Run Dashboard
streamlit run dashboard/app.py

# Run Tests
pytest tests/ -v --cov=src

# Start MLflow UI
mlflow ui
```

---

## ✅ Next Steps

1. **Start with Phase 1**: Complete environment setup
2. **Download dataset**: Telco Customer Churn from Kaggle
3. **Follow PHASE_2**: Data ingestion instructions
4. **Work sequentially**: Don't skip phases

Each phase has detailed code files (PHASE_X_*.md) with copy-paste ready code.

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| PostgreSQL not found | Add to PATH or reinstall |
| Import errors | `pip install -r requirements.txt` again |
| Port 8000 in use | `netstat -ano \| findstr :8000` + kill process |
| Database connection fail | Check credentials in `.env` |

---

## 📚 Key Concepts Covered

- ✅ SQL + PostgreSQL (data storage)
- ✅ Pandas + NumPy (data manipulation)
- ✅ Scikit-learn + XGBoost (ML models)
- ✅ SHAP + LIME (explainability)
- ✅ FastAPI + Docker (deployment)
- ✅ MLflow + DVC (MLOps)
- ✅ GitHub Actions (CI/CD)
- ✅ Streamlit (dashboarding)

---

**Project Created**: May 2026  
**Target Role**: Data Engineer / ML Engineer / MLOps Engineer  
**Difficulty**: Intermediate → Advanced
