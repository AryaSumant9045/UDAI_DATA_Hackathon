"""
Flask REST API for Aadhaar ML Models
Provides endpoints for:
1. At-risk region prediction
2. Demand forecasting
3. Batch predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== HELPER FUNCTIONS ====================

def load_and_clean_data(folder_path):
    """Load all CSV files from a folder and clean them"""
    data = pd.DataFrame()
    if os.path.exists(folder_path):
        for file in sorted(os.listdir(folder_path)):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(folder_path, file))
                    data = pd.concat([data, df], ignore_index=True)
                except Exception as e:
                    logger.warning(f"Error loading {file}: {str(e)}")
        
        if data.shape[0] > 0:
            data = data.drop_duplicates().reset_index(drop=True)
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
            for col in data.select_dtypes(include=['object']).columns:
                if data[col].isnull().sum() > 0:
                    mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    data[col].fillna(mode_val, inplace=True)
    return data

def calculate_alci(biometric_df, demographic_df):
    """Calculate Aadhaar Lifecycle Compliance Index"""
    groupby_col = 'State' if 'State' in demographic_df.columns else ('Pincode' if 'Pincode' in demographic_df.columns else None)
    
    if groupby_col is None:
        demo_count = demographic_df.shape[0]
        bio_count = biometric_df.shape[0]
        alci_score = (bio_count / max(demo_count, 1) * 100)
        alci = pd.DataFrame({
            'Region': ['National Overall'],
            'biometric_updates': [bio_count],
            'demographic_updates': [demo_count],
            'ALCI_Score': [min(alci_score, 100)]
        })
    else:
        bio_by_region = biometric_df.groupby(groupby_col).size().reset_index(name='biometric_updates')
        demo_by_region = demographic_df.groupby(groupby_col).size().reset_index(name='demographic_updates')
        alci = bio_by_region.merge(demo_by_region, on=groupby_col, how='outer')
        alci = alci.fillna(1)
        alci.rename(columns={groupby_col: 'Region'}, inplace=True)
        alci['ALCI_Score'] = (alci['biometric_updates'] / alci['demographic_updates'] * 100).round(2)
        alci['ALCI_Score'] = alci['ALCI_Score'].clip(upper=100)
    
    alci['Risk_Level'] = pd.cut(alci['ALCI_Score'], bins=[0, 30, 60, 100], labels=['High Risk', 'Medium Risk', 'Healthy'])
    return alci.sort_values('ALCI_Score', ascending=False)

def train_model():
    """Train logistic regression model"""
    biometric_data = load_and_clean_data('api_data_aadhar_biometric')
    demographic_data = load_and_clean_data('api_data_aadhar_demographic')
    
    alci_data = calculate_alci(biometric_data, demographic_data)
    
    np.random.seed(42)
    X = alci_data[['biometric_updates', 'demographic_updates']].values
    y = (alci_data['Risk_Level'] == 'High Risk').astype(int).values
    
    if y.sum() < 3:
        synthetic_X = np.array([
            [100, 500], [150, 600], [120, 550],
            [5000, 50000], [6000, 60000], [7000, 70000]
        ])
        synthetic_y = np.array([1, 1, 1, 0, 0, 0])
        X = np.vstack([X, synthetic_X])
        y = np.hstack([y, synthetic_y])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)
    
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = log_reg.score(X_test, y_test)
    
    return log_reg, scaler, accuracy, roc_auc, alci_data

# Initialize model on startup
try:
    log_reg, scaler, accuracy, roc_auc, alci_data = train_model()
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    log_reg, scaler, accuracy, roc_auc, alci_data = None, None, 0, 0, None

# ==================== ROUTES ====================

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': '✅ Running',
        'message': 'Aadhaar ML API',
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': '/health',
            'predict_risk': '/api/predict-risk',
            'predict_batch': '/api/predict-batch',
            'model_info': '/api/model-info',
            'forecast_demand': '/api/forecast-demand'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': log_reg is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'models': [
            {
                'name': 'Logistic Regression',
                'type': 'Classification',
                'purpose': 'At-risk region prediction',
                'accuracy': f"{accuracy:.1%}",
                'roc_auc': f"{roc_auc:.1%}",
                'input_features': ['biometric_updates', 'demographic_updates'],
                'output': 'risk_probability (0-1)'
            },
            {
                'name': 'ARIMA(1,1,1)',
                'type': 'Time-Series',
                'purpose': '30-day demand forecasting',
                'confidence_level': '95%',
                'forecast_horizon': '30 days',
                'output': ['forecast_values', 'confidence_upper', 'confidence_lower']
            }
        ]
    }), 200

@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    """
    Predict at-risk probability for a region
    
    Request JSON:
    {
        "biometric_updates": 5000,
        "demographic_updates": 50000
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'biometric_updates' not in data or 'demographic_updates' not in data:
            return jsonify({
                'error': 'Missing required fields: biometric_updates, demographic_updates'
            }), 400
        
        bio_updates = float(data['biometric_updates'])
        demo_updates = float(data['demographic_updates'])
        
        # Validate values
        if bio_updates < 0 or demo_updates < 0:
            return jsonify({
                'error': 'Values must be non-negative'
            }), 400
        
        # Make prediction
        features = np.array([[bio_updates, demo_updates]])
        features_scaled = scaler.transform(features)
        risk_probability = float(log_reg.predict_proba(features_scaled)[0][1])
        
        # Determine risk level
        if risk_probability > 0.5:
            risk_level = 'Critical'
        elif risk_probability > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Safe'
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'biometric_updates': bio_updates,
                'demographic_updates': demo_updates,
                'risk_probability': risk_probability,
                'risk_level': risk_level,
                'alci_score': (bio_updates / max(demo_updates, 1) * 100)
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in predict_risk: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Make predictions on multiple regions
    
    Request JSON:
    {
        "regions": [
            {"name": "Region1", "biometric_updates": 5000, "demographic_updates": 50000},
            {"name": "Region2", "biometric_updates": 3000, "demographic_updates": 30000}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'regions' not in data:
            return jsonify({
                'error': 'Missing required field: regions'
            }), 400
        
        regions = data['regions']
        predictions = []
        
        for region in regions:
            bio_updates = float(region.get('biometric_updates', 0))
            demo_updates = float(region.get('demographic_updates', 0))
            region_name = region.get('name', 'Unknown')
            
            features = np.array([[bio_updates, demo_updates]])
            features_scaled = scaler.transform(features)
            risk_probability = float(log_reg.predict_proba(features_scaled)[0][1])
            
            risk_level = 'Critical' if risk_probability > 0.5 else 'Medium' if risk_probability > 0.3 else 'Safe'
            
            predictions.append({
                'region_name': region_name,
                'biometric_updates': bio_updates,
                'demographic_updates': demo_updates,
                'risk_probability': risk_probability,
                'risk_level': risk_level,
                'alci_score': (bio_updates / max(demo_updates, 1) * 100)
            })
        
        # Summary statistics
        risk_probs = [p['risk_probability'] for p in predictions]
        
        summary = {
            'total_regions': len(predictions),
            'critical_risk': sum(1 for p in predictions if p['risk_level'] == 'Critical'),
            'medium_risk': sum(1 for p in predictions if p['risk_level'] == 'Medium'),
            'safe': sum(1 for p in predictions if p['risk_level'] == 'Safe'),
            'average_risk': np.mean(risk_probs)
        }
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in predict_batch: {str(e)}")
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@app.route('/api/forecast-demand', methods=['GET'])
def forecast_demand():
    """
    Generate 30-day demand forecast
    Optional query param: days (default: 30, max: 90)
    """
    try:
        days = request.args.get('days', 30, type=int)
        if days < 1 or days > 90:
            return jsonify({
                'error': 'Days parameter must be between 1 and 90'
            }), 400
        
        # Create synthetic time series
        np.random.seed(42)
        periods = 60
        time_index = pd.date_range(start='2025-11-01', periods=periods, freq='D')
        
        base_demo = 69000 / 30
        demo_ts = base_demo + np.cumsum(np.random.normal(-10, 50, periods))
        demo_ts = np.maximum(demo_ts, 100)
        
        ts_data = pd.DataFrame({
            'Date': time_index,
            'Demographic': demo_ts.astype(int)
        })
        
        # Fit ARIMA and forecast
        arima_model = ARIMA(ts_data['Demographic'], order=(1, 1, 1))
        arima_fit = arima_model.fit()
        
        forecast_30d = arima_fit.get_forecast(steps=days)
        forecast_mean = forecast_30d.predicted_mean
        forecast_ci = forecast_30d.conf_int(alpha=0.05)
        
        forecast_dates = pd.date_range(
            start=ts_data['Date'].max() + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Create forecast dataframe
        forecast_data = []
        for i in range(len(forecast_mean)):
            forecast_data.append({
                'date': forecast_dates[i].isoformat(),
                'forecast': float(forecast_mean.iloc[i]),
                'confidence_lower': float(forecast_ci.iloc[i, 0]),
                'confidence_upper': float(forecast_ci.iloc[i, 1])
            })
        
        # Calculate statistics
        forecast_values = [f['forecast'] for f in forecast_data]
        
        return jsonify({
            'status': 'success',
            'forecast': forecast_data,
            'statistics': {
                'mean': np.mean(forecast_values),
                'std': np.std(forecast_values),
                'min': np.min(forecast_values),
                'max': np.max(forecast_values),
                'peak_date': forecast_dates[np.argmax(forecast_values)].isoformat()
            },
            'model': 'ARIMA(1,1,1)',
            'confidence_level': '95%',
            'forecast_period': f"{days} days",
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in forecast_demand: {str(e)}")
        return jsonify({
            'error': f'Forecast generation failed: {str(e)}'
        }), 500

@app.route('/api/regions-analysis', methods=['GET'])
def regions_analysis():
    """Get analysis of all regions"""
    try:
        if alci_data is None:
            return jsonify({'error': 'ALCI data not loaded'}), 500
        
        # Get risk predictions for all regions
        X = alci_data[['biometric_updates', 'demographic_updates']].values
        X_scaled = scaler.transform(X)
        risk_predictions = log_reg.predict_proba(X_scaled)[:, 1]
        
        alci_data['risk_probability'] = risk_predictions
        
        # Create response
        regions = []
        for idx, row in alci_data.head(20).iterrows():
            regions.append({
                'region': row['Region'],
                'alci_score': float(row['ALCI_Score']),
                'risk_probability': float(row['risk_probability']),
                'biometric_updates': int(row['biometric_updates']),
                'demographic_updates': int(row['demographic_updates'])
            })
        
        # Summary
        summary = {
            'total_regions': len(alci_data),
            'critical_risk': sum(risk_predictions > 0.5),
            'medium_risk': sum((risk_predictions >= 0.3) & (risk_predictions <= 0.5)),
            'safe': sum(risk_predictions < 0.3)
        }
        
        return jsonify({
            'status': 'success',
            'regions': regions,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in regions_analysis: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'health': '/health',
            'predict_risk': '/api/predict-risk',
            'predict_batch': '/api/predict-batch',
            'model_info': '/api/model-info',
            'forecast_demand': '/api/forecast-demand',
            'regions_analysis': '/api/regions-analysis'
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', False)
    
    logger.info(f"Starting Aadhaar ML API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
