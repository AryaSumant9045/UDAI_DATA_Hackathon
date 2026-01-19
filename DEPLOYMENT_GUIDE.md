# 🚀 Quick Start Guide - Deployment

## 📦 Files Created

```
/Users/aryasumant/Desktop/Kaggle/UDAI-Hackathon/
├── streamlit_app.py        # 🎨 Interactive web dashboard
├── flask_app.py            # 🔌 REST API for production
├── ml_models.ipynb         # 📓 Original Jupyter notebook
├── requirements.txt        # 📚 Python dependencies
├── README.md               # 📖 Complete documentation
└── api_data_aadhar_*/      # 📊 Data folders
```

---

## 🎯 Option 1: Run Streamlit App Locally (EASIEST)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the app
```bash
streamlit run streamlit_app.py
```

### Step 3: Access in browser
```
http://localhost:8501
```

**Features:**
- 🏠 Home dashboard
- 🚨 At-Risk Prediction with visualizations
- 📈 30-Day Demand Forecast
- 📊 Model metrics and summaries
- 📤 Batch prediction interface

---

## 🔌 Option 2: Run Flask REST API

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the API
```bash
python flask_app.py
```

### Step 3: API available at
```
http://localhost:5000
```

---

## 📡 API Endpoints

### 1. Health Check
```bash
curl http://localhost:5000/health
```

### 2. Model Information
```bash
curl http://localhost:5000/api/model-info
```

### 3. Predict Single Risk
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
  "status": "success",
  "prediction": {
    "biometric_updates": 5000,
    "demographic_updates": 50000,
    "risk_probability": 0.15,
    "risk_level": "Safe",
    "alci_score": 10.0
  },
  "timestamp": "2026-01-19T10:30:00"
}
```

### 4. Batch Predictions
```bash
curl -X POST http://localhost:5000/api/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "regions": [
      {"name": "Region1", "biometric_updates": 5000, "demographic_updates": 50000},
      {"name": "Region2", "biometric_updates": 3000, "demographic_updates": 30000}
    ]
  }'
```

### 5. Demand Forecast
```bash
curl "http://localhost:5000/api/forecast-demand?days=30"
```

### 6. Regions Analysis
```bash
curl http://localhost:5000/api/regions-analysis
```

---

## ☁️ Option 3: Deploy to Streamlit Cloud (FREE)

### Step 1: Create GitHub Repository
```bash
cd /Users/aryasumant/Desktop/Kaggle/UDAI-Hackathon
git init
git add .
git commit -m "Add Streamlit ML app"
git remote add origin https://github.com/YOUR_USERNAME/aadhaar-ml.git
git push -u origin main
```

### Step 2: Go to Streamlit Cloud
1. Visit https://streamlit.io/cloud
2. Click "New app"
3. Connect GitHub repository
4. Select branch: `main`
5. Select file: `streamlit_app.py`
6. Click "Deploy"

**Your app is now live!** 🎉

---

## 🐳 Option 4: Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 5000

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py"]
```

### Build & Run
```bash
# Build image
docker build -t aadhaar-ml:latest .

# Run container
docker run -p 8501:8501 aadhaar-ml:latest
```

---

## 🚀 Option 5: Deploy to Heroku

### Step 1: Install Heroku CLI
```bash
brew tap heroku/brew && brew install heroku
```

### Step 2: Create Procfile
```bash
echo "web: streamlit run streamlit_app.py --server.port=\$PORT" > Procfile
```

### Step 3: Deploy
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Step 4: View app
```bash
heroku open
```

---

## 📊 Testing the Application

### Test 1: Check if Streamlit runs
```bash
streamlit run streamlit_app.py
```
Visit: http://localhost:8501

### Test 2: Check if Flask API runs
```bash
python flask_app.py
```
Visit: http://localhost:5000/health

### Test 3: Make API prediction
```bash
curl -X POST http://localhost:5000/api/predict-risk \
  -H "Content-Type: application/json" \
  -d '{"biometric_updates": 5000, "demographic_updates": 50000}'
```

---

## 🔐 Production Checklist

- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Test locally: `streamlit run streamlit_app.py`
- [ ] Test API: `python flask_app.py`
- [ ] Check data folders exist (api_data_aadhar_*)
- [ ] Verify models train successfully
- [ ] Test predictions with sample data
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud or Heroku
- [ ] Test deployed app
- [ ] Monitor logs

---

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit
```

### Issue: "No module named 'sklearn'"
**Solution:**
```bash
pip install scikit-learn
```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
streamlit run streamlit_app.py --server.port=8502
```

### Issue: "Data folder not found"
**Solution:**
Make sure data folders are in the same directory:
```
UDAI-Hackathon/
├── api_data_aadhar_biometric/
├── api_data_aadhar_demographic/
├── api_data_aadhar_enrolment/
├── streamlit_app.py
├── flask_app.py
└── ...
```

---

## 📝 Environment Variables (Optional)

Create `.env` file:
```
FLASK_DEBUG=False
PORT=5000
LOG_LEVEL=INFO
```

Load in Flask app:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 🎓 Next Steps

1. **Customize the app:**
   - Modify colors in `streamlit_app.py`
   - Add more visualizations
   - Add user authentication

2. **Add more models:**
   - Random Forest for classification
   - Prophet for forecasting
   - Deep learning models

3. **Database integration:**
   - Store predictions in database
   - Track performance over time
   - Create admin dashboard

4. **Monitor in production:**
   - Setup error logging
   - Monitor API response times
   - Track prediction accuracy

---

## 📞 Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review code comments in streamlit_app.py
3. Check API documentation at `/api/model-info`
4. Monitor logs for errors

---

**Status:** ✅ Ready for Deployment  
**Last Updated:** January 19, 2026
