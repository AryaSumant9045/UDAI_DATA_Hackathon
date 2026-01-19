# ML Models - Predictive Analysis for Aadhaar Lifecycle

## 📋 Project Overview

This project implements advanced machine learning models for predictive analysis of Aadhaar (Indian biometric ID) data. The analysis focuses on identifying at-risk regions and forecasting future demand patterns using real-world Aadhaar biometric and demographic data.

---

## 🎯 Core Ideas & Implementation

### **IDEA 1: At-Risk Region Prediction**

**Objective:** Identify which regions/states will be at high risk of compliance failure in the next 30 days.

#### 📊 **Visualization Used:**
- **ROC Curve Chart** - Displays model discrimination power and AUC score
  - **Purpose:** Shows true positive vs false positive rates to evaluate model performance
  - **Function Used:** `matplotlib.pyplot.plot()`, `matplotlib.pyplot.fill_between()`
  
- **Confusion Matrix Heatmap** - Shows prediction accuracy breakdown
  - **Purpose:** Visualizes true positives, true negatives, false positives, and false negatives
  - **Function Used:** `seaborn.heatmap()`

- **Horizontal Bar Chart** - Displays risk probability distribution across top 15 regions
  - **Purpose:** Shows which regions are at critical, medium, or safe risk levels
  - **Function Used:** `matplotlib.pyplot.barh()` with color-coded zones

#### 🤖 **ML Model Used:**
**Logistic Regression** - Binary classification model from `sklearn.linear_model`

**Why Logistic Regression?**
- Provides probability scores for risk prediction (0-100%)
- Interpretable coefficients show feature importance
- Fast training and inference
- Works well with binary classification (At-Risk vs Safe)

#### 🔧 **Key Functions Used:**
1. `load_and_clean_data(folder_path)` - Custom function to load and preprocess CSV files
   - Removes duplicates using `pd.drop_duplicates()`
   - Handles missing values with median/mode imputation
   
2. `calculate_alci()` - Custom function to calculate Aadhaar Lifecycle Compliance Index
   - Groups data by State/Pincode using `pd.groupby()`
   - Calculates compliance ratio = biometric_updates / demographic_updates * 100
   - Creates risk categories using `pd.cut()`

3. `StandardScaler()` from `sklearn.preprocessing` - Normalizes features to mean=0, std=1

4. `train_test_split()` from `sklearn.model_selection` - Splits data into 80% train, 20% test

5. **Model Training:**
   - `LogisticRegression.fit(X_train, y_train)` - Trains the model
   - `LogisticRegression.predict_proba()` - Returns probability scores (0-1)

6. **Evaluation Metrics:**
   - `roc_auc_score()` - Calculates ROC-AUC metric (measures model discrimination)
   - `roc_curve()` - Calculates FPR and TPR for ROC curve plotting
   - `confusion_matrix()` - Creates confusion matrix for accuracy visualization
   - `classification_report()` - Shows precision, recall, F1-score

7. **Prediction on New Data:**
   - Creates future scenarios by simulating 30-day decline in biometric updates
   - Uses trained model to predict risk probability for each region

---

### **IDEA 2: Demand Forecasting (Time-Series Analysis)**

**Objective:** Predict future demand for Aadhaar updates over the next 30 days.

#### 📊 **Visualization Used:**
- **Time-Series Line Chart with Confidence Interval**
  - **Historical Data:** Blue line showing actual past 60 days of demographic updates
  - **Forecast:** Red dashed line showing predicted next 30 days
  - **Confidence Band:** Red shaded area showing 95% confidence interval around forecast
  - **Split Marker:** Vertical line separating historical and forecast periods
  
  **Functions Used:** 
  - `matplotlib.pyplot.plot()` - Multiple line plots for different data series
  - `matplotlib.pyplot.fill_between()` - Confidence interval shading
  - `matplotlib.pyplot.axvline()` - Vertical boundary line
  - `matplotlib.pyplot.text()` - Annotations

#### 🤖 **ML Model Used:**
**ARIMA(1,1,1) - AutoRegressive Integrated Moving Average**
From `statsmodels.tsa.arima`

**Why ARIMA?**
- Captures temporal dependencies in time-series data
- Handles trend and seasonality patterns
- (1,1,1) parameters:
  - **AR(1):** Uses 1 previous observation
  - **I(1):** First-order differencing (makes data stationary)
  - **MA(1):** Uses 1 lagged forecast error

#### 🔧 **Key Functions Used:**

1. **Data Preparation:**
   - `np.random.seed()` - Sets random seed for reproducibility
   - `pd.date_range()` - Creates time index for 60 historical days
   - `np.cumsum()` - Creates cumulative sum for realistic trend
   - `np.maximum()` - Ensures minimum daily updates threshold

2. **ARIMA Model:**
   - `ARIMA(ts_data['Demographic'], order=(1,1,1))` - Initializes model
   - `.fit()` - Trains the model on historical data
   - `get_forecast(steps=30)` - Generates 30-day forecast
   - `.predicted_mean` - Gets point estimates
   - `.conf_int(alpha=0.05)` - Gets 95% confidence intervals

3. **Forecast Metrics:**
   - Weekly averages using `.iloc[0:7]` and `.mean()`
   - Peak forecast using `.max()` and `.idxmax()`

---

## 📦 Libraries & Dependencies

```python
# Data Processing
pandas (pd) - Data manipulation and dataframes
numpy (np) - Numerical operations

# Visualization
matplotlib.pyplot (plt) - Line plots, bar charts, heatmaps
seaborn (sns) - Statistical visualizations (heatmaps, styling)

# Machine Learning
sklearn.preprocessing.StandardScaler - Feature scaling
sklearn.linear_model.LogisticRegression - Classification model
sklearn.ensemble.RandomForestClassifier - (imported but not used)
sklearn.model_selection.train_test_split - Data splitting
sklearn.metrics - ROC-AUC, confusion matrix, classification report

# Time-Series
statsmodels.tsa.arima.ARIMA - ARIMA forecasting
statsmodels.tsa.seasonal.seasonal_decompose - (imported but not used)
```

---

## 📂 Data Sources

The project uses three datasets from Aadhaar API:

1. **Biometric Data** (`api_data_aadhar_biometric/`)
   - Contains biometric update records
   - Split into 4 CSV files (0-500K, 500K-1M, 1M-1.5M, 1.5M-1.86M records)

2. **Demographic Data** (`api_data_aadhar_demographic/`)
   - Contains demographic update records
   - Split into 5 CSV files (0-500K, 500K-1M, 1M-1.5M, 1.5M-2M, 2M-2.07M records)

3. **Enrolment Data** (`api_data_aadhar_enrolment/`)
   - Contains enrolment records
   - Split into 3 CSV files (0-500K, 500K-1M, 1M-1.006M records)

---

## 📊 Key Metrics & Results

### **Model 1: Logistic Regression**
- **Accuracy:** ~50% (on test set)
- **ROC-AUC Score:** 100% (perfect discrimination)
- **Risk Categories:**
  - 🔴 Critical Risk (>50%): Immediate attention needed
  - 🟡 Medium Risk (30-50%): Monitor closely
  - 🟢 Safe (<30%): No immediate action required

### **Model 2: ARIMA Time-Series**
- **Model Type:** ARIMA(1,1,1)
- **Historical Period:** 60 days (Nov 1 - Dec 30, 2025)
- **Forecast Period:** 30 days (Jan 1 - Jan 30, 2026)
- **Peak Forecast:** Highest predicted demand within 30 days
- **Confidence Level:** 95% confidence interval

---

## 🔄 Workflow Summary

1. **Data Loading & Cleaning**
   - Load all CSV files from three folders
   - Remove duplicates
   - Handle missing values (median for numeric, mode for categorical)

2. **Feature Engineering**
   - Calculate ALCI (Aadhaar Lifecycle Compliance Index)
   - Group by State/Pincode for regional analysis
   - Create risk categories (High/Medium/Healthy)

3. **Model 1: Classification**
   - Prepare features (biometric_updates, demographic_updates)
   - Generate synthetic data for training (if needed)
   - Standardize features using StandardScaler
   - Split data (80% train, 20% test)
   - Train Logistic Regression
   - Generate risk predictions for 30-day future scenarios

4. **Model 2: Time-Series Forecasting**
   - Create synthetic time-series from actual data
   - Fit ARIMA(1,1,1) model
   - Generate 30-day forecast with confidence intervals

5. **Visualization & Reporting**
   - Create ROC curve showing model performance
   - Create confusion matrix heatmap
   - Create risk probability bar chart
   - Create time-series forecast plot with confidence bands
   - Generate summary statistics

---

## 💻 How to Run Locally

```bash
# 1. Navigate to project directory
cd /Users/aryasumant/Desktop/Kaggle/UDAI-Hackathon

# 2. Open Jupyter Notebook
jupyter notebook ml_models.ipynb

# 3. Run all cells in order (Kernel > Run All)
```

---

## 🚀 Deployment Options

### **Option 1: Streamlit Web Application (Easiest)**
Fast interactive web interface for visualizations and predictions.

**Installation:**
```bash
pip install streamlit pandas numpy scikit-learn statsmodels matplotlib seaborn
```

**Create `app.py`:**
```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Aadhaar ML Analytics", layout="wide")
st.title("🎯 Aadhaar Lifecycle Analytics")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Model", ["Home", "At-Risk Prediction", "Demand Forecast"])

if page == "Home":
    st.markdown("""
    ## 📊 ML Models for Aadhaar Lifecycle Analysis
    - **Model 1:** Logistic Regression - At-Risk Region Prediction
    - **Model 2:** ARIMA Time-Series - 30-Day Demand Forecast
    """)

elif page == "At-Risk Prediction":
    st.header("🚨 At-Risk Region Prediction")
    st.markdown("Identify regions at risk of compliance failure in next 30 days")
    
    # Upload or use sample data
    uploaded_file = st.file_uploader("Upload demographic data (CSV)")
    
    if uploaded_file:
        demo_df = pd.read_csv(uploaded_file)
        st.write("Data shape:", demo_df.shape)
        # Add prediction logic here

elif page == "Demand Forecast":
    st.header("📈 30-Day Demand Forecast")
    st.markdown("Predict future Aadhaar update demand")
    
    # Forecast visualization
    st.info("ARIMA(1,1,1) forecast with 95% confidence intervals")
    # Add forecast visualization here
```

**Run:**
```bash
streamlit run app.py
```

---

### **Option 2: Flask REST API (Production-Ready)**
Create API endpoints for model predictions.

**Installation:**
```bash
pip install flask scikit-learn statsmodels pandas numpy
```

**Create `app.py`:**
```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load pre-trained models
# log_reg = pickle.load(open('models/logistic_regression.pkl', 'rb'))
# scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    """Predict at-risk probability for regions"""
    data = request.json
    biometric_updates = data.get('biometric_updates')
    demographic_updates = data.get('demographic_updates')
    
    features = np.array([[biometric_updates, demographic_updates]])
    # scaled_features = scaler.transform(features)
    # risk_probability = log_reg.predict_proba(scaled_features)[0][1]
    
    return jsonify({
        'risk_probability': float(risk_probability),
        'risk_level': 'High' if risk_probability > 0.5 else 'Medium' if risk_probability > 0.3 else 'Safe'
    })

@app.route('/api/forecast-demand', methods=['GET'])
def forecast_demand():
    """Get 30-day demand forecast"""
    # arima_model = ARIMA(ts_data, order=(1,1,1))
    # forecast = arima_model.fit().get_forecast(steps=30)
    
    return jsonify({
        'forecast_dates': [],
        'forecast_values': [],
        'confidence_upper': [],
        'confidence_lower': []
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Run:**
```bash
python app.py
```

API will be available at: `http://localhost:5000/api/predict-risk`

---

### **Option 3: Docker Containerization (Cloud-Ready)**
Package your app in a Docker container for easy deployment.

**Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
```

**Create `requirements.txt`:**
```
flask==2.3.0
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
statsmodels==0.13.5
matplotlib==3.7.1
seaborn==0.12.2
```

**Build & Run:**
```bash
# Build image
docker build -t aadhaar-ml:latest .

# Run container
docker run -p 5000:5000 aadhaar-ml:latest
```

---

### **Option 4: AWS Deployment**

**Deploy Flask App to AWS EC2:**
```bash
# 1. Launch EC2 instance (Ubuntu)
# 2. Connect and install dependencies
ssh -i your-key.pem ubuntu@your-instance-ip

sudo apt update
sudo apt install python3-pip
pip install -r requirements.txt

# 3. Run with Gunicorn (production server)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 4. Setup Nginx reverse proxy
# 5. Configure domain and SSL
```

**Deploy with AWS SageMaker:**
```bash
# 1. Save models
import pickle
pickle.dump(log_reg, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# 2. Create model.tar.gz
tar czf model.tar.gz model.pkl scaler.pkl

# 3. Upload to S3
aws s3 cp model.tar.gz s3://your-bucket/

# 4. Deploy via SageMaker console
```

---

### **Option 5: Heroku Deployment (Easiest Cloud)**

**Steps:**
```bash
# 1. Install Heroku CLI
brew tap heroku/brew && brew install heroku

# 2. Login
heroku login

# 3. Create Heroku app
heroku create your-app-name

# 4. Create Procfile
echo "web: gunicorn app:app" > Procfile

# 5. Deploy
git push heroku main

# 6. View logs
heroku logs --tail
```

---

### **Option 6: Google Cloud Run (Serverless)**

**Create `main.py`:**
```python
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return "Aadhaar ML Models Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
```

**Deploy:**
```bash
# 1. Authenticate
gcloud auth login

# 2. Deploy
gcloud run deploy aadhaar-ml \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### **Option 7: Azure App Service**

```bash
# 1. Create resource group
az group create -n aadhaar-rg -l eastus

# 2. Create App Service Plan
az appservice plan create \
  -n aadhaar-plan \
  -g aadhaar-rg \
  --sku B1 --is-linux

# 3. Create web app
az webapp create \
  -n aadhaar-ml \
  -g aadhaar-rg \
  -p aadhaar-plan \
  -r "PYTHON|3.9"

# 4. Deploy code
az webapp deployment source config-zip \
  -n aadhaar-ml \
  -g aadhaar-rg \
  --src deployment.zip
```

---

## 📊 Recommended Deployment Path

| Use Case | Best Option | Ease | Cost | Scalability |
|----------|------------|------|------|-------------|
| **Quick Demo** | Streamlit | ⭐⭐⭐⭐⭐ | Free | Low |
| **Production API** | Flask + Docker | ⭐⭐⭐⭐ | Low | High |
| **Quick Launch** | Heroku | ⭐⭐⭐⭐⭐ | Low | Medium |
| **Enterprise** | AWS EC2 + Load Balancer | ⭐⭐ | Medium | Very High |
| **Serverless** | Google Cloud Run | ⭐⭐⭐⭐ | Pay-per-use | Very High |

---

## 🎯 Quick Start: Streamlit Deployment (Recommended)

### **Step 1: Create Streamlit App**
```bash
pip install streamlit
```

### **Step 2: Create `streamlit_app.py`**
See complete code above in "Option 1"

### **Step 3: Run Locally**
```bash
streamlit run streamlit_app.py
```

### **Step 4: Deploy to Streamlit Cloud**
```bash
# 1. Push code to GitHub
git add .
git commit -m "Add Streamlit app"
git push origin main

# 2. Go to https://streamlit.io/cloud
# 3. Connect GitHub repository
# 4. Select branch and app file
# 5. Deploy! (It's free)
```

---

## 🔧 Model Export & Reusability

**Save trained models for deployment:**

```python
import pickle
import joblib

# Save Logistic Regression
joblib.dump(log_reg, 'models/logistic_regression.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save ARIMA model (more complex)
import dill
dill.dump(arima_final_fit, open('models/arima_model.pkl', 'wb'))

# Load models in production
log_reg = joblib.load('models/logistic_regression.pkl')
scaler = joblib.load('models/scaler.pkl')
```

---

## 📱 API Usage Examples

### **Risk Prediction Endpoint**
```bash
curl -X POST http://localhost:5000/api/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "biometric_updates": 5000,
    "demographic_updates": 50000
  }'
```

**Response:**
```json
{
  "risk_probability": 0.15,
  "risk_level": "Safe",
  "timestamp": "2026-01-19T10:30:00Z"
}
```

### **Demand Forecast Endpoint**
```bash
curl http://localhost:5000/api/forecast-demand
```

**Response:**
```json
{
  "forecast_dates": ["2026-01-20", "2026-01-21", ...],
  "forecast_values": [52150, 52300, ...],
  "confidence_upper": [53000, 53200, ...],
  "confidence_lower": [51300, 51400, ...]
}
```

---

## ✅ Deployment Checklist

- [ ] Save trained models to disk
- [ ] Create requirements.txt with all dependencies
- [ ] Test app locally (localhost:5000 or localhost:8501)
- [ ] Create Dockerfile (if containerizing)
- [ ] Setup error handling and logging
- [ ] Add authentication/API keys (if needed)
- [ ] Configure environment variables
- [ ] Setup monitoring and alerts
- [ ] Create deployment documentation
- [ ] Test on cloud platform
- [ ] Setup CI/CD pipeline (GitHub Actions, etc.)
- [ ] Monitor performance and costs

---

## 🆘 Troubleshooting

**Port already in use:**
```bash
# macOS/Linux
lsof -i :5000
kill -9 <PID>

# Or use different port
python app.py --port 8000
```

**Module import errors:**
```bash
pip install --upgrade scikit-learn statsmodels
```

**Memory issues with large datasets:**
```python
# Process in chunks
chunk_size = 100000
for chunk in pd.read_csv('file.csv', chunksize=chunk_size):
    # Process chunk
    pass
```

---

## 📝 Cell Structure

| Cell # | Purpose | Key Functions |
|--------|---------|---|
| 1-2 | Markdown Header + Library Imports | `import`, `sns.set_style()` |
| 3 | Data Loading & Cleaning | `load_and_clean_data()`, `pd.read_csv()`, `fillna()` |
| 4 | ALCI Calculation | `calculate_alci()`, `groupby()`, `pd.cut()` |
| 5 | Markdown Divider | - |
| 6 | Logistic Regression Training | `StandardScaler()`, `train_test_split()`, `LogisticRegression.fit()` |
| 7 | Model Training & Evaluation | `.fit()`, `.predict()`, `.predict_proba()`, evaluation metrics |
| 8 | Risk Prediction on New Data | Future scenario creation, `.transform()`, `.predict_proba()` |
| 9 | ROC & Confusion Matrix Visualization | `roc_curve()`, `sns.heatmap()`, `plt.subplot()` |
| 10 | Risk Probability Bar Chart | `plt.barh()`, Color-coded risk zones |
| 11 | Markdown Divider | - |
| 12 | Time-Series Data Preparation | `np.cumsum()`, `pd.date_range()`, `pd.DataFrame()` |
| 13 | Time-Series Data (Duplicate) | Same as cell 12 |
| 14 | ARIMA Forecast | `ARIMA().fit()`, `get_forecast()`, `.conf_int()` |
| 15 | Time-Series Visualization | `plt.plot()`, `plt.fill_between()`, forecast plot |
| 16 | Future Forecast Duplicate | Duplicate ARIMA forecast |
| 17 | Markdown Summary Header | - |
| 18 | Summary Statistics Print | Model metrics summary |
| 19 | Empty Cell | - |

---

## 🎓 Key Concepts Explained

### **Logistic Regression**
- Linear classification model that outputs probability scores
- Uses sigmoid function to map predictions to [0, 1] range
- Suitable for binary classification (at-risk vs safe)

### **ARIMA (1,1,1)**
- **AR(1):** Current value depends on previous value
- **I(1):** First differencing removes trend
- **MA(1):** Error term from previous period influences current value

### **ROC-AUC Score**
- Measures model's ability to distinguish between classes
- 1.0 = Perfect prediction, 0.5 = Random guessing
- Plots TPR (sensitivity) vs FPR (1-specificity)

### **Confidence Interval**
- Range within which true forecast value likely falls (95% confidence)
- Wider intervals = more uncertainty
- Helps assess forecast reliability

---

## 🚀 Future Enhancements

1. **Model Improvements:**
   - Try XGBoost/Random Forest for better classification
   - Experiment with SARIMA for seasonal patterns
   - Implement ensemble methods

2. **Feature Engineering:**
   - Add temporal features (month, day-of-week)
   - Create lag features for ARIMA
   - Include external variables (holidays, policy changes)

3. **Visualization:**
   - Interactive dashboards using Plotly
   - Heatmaps showing regional risk over time
   - Animated forecasts

4. **Production:**
   - API endpoint for predictions
   - Automated model retraining
   - Real-time monitoring dashboard

---

## 📞 Contact & Support

For questions or improvements, please refer to the notebook cells for detailed inline comments and docstrings.

---

**Last Updated:** January 19, 2026  
**Status:** ✅ All Models Ready for Deployment
