from flask import Flask, request, jsonify, render_template
from datetime import datetime
import joblib
import numpy as np
import logging
from typing import Dict, Any, List
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Current working directory: %s", os.getcwd())
logger.info("Environment variables loaded: %s", os.environ.get('MONGODB_URI') is not None)

app = Flask(__name__)

# MongoDB connection
try:
    mongodb_uri = os.getenv('MONGODB_URI')
    logger.info("MongoDB URI found: %s", mongodb_uri is not None)
    
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
        
    client = MongoClient(mongodb_uri)
    db = client['weather_prediction']
    collection = db['sensor_data']
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    client = None
    collection = None

# Load the trained AI model
try:
    model = joblib.load('model.pkl')
    # Validate model
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded model does not have predict method")
    
    # Log model details
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model parameters: {model.get_params() if hasattr(model, 'get_params') else 'No parameters available'}")
    logger.info(f"Model features: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'No feature names'}")
    
    # Test prediction with sample data
    test_data = np.array([[25.0, 60.0, 1013.0]])
    test_df = pd.DataFrame(test_data, columns=['Temperature', 'Humidity', 'Pressure'])
    test_pred = model.predict(test_df)[0]
    logger.info(f"Test prediction result: {test_pred}")
    
    logger.info("Model loaded and validated successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(f"Error type: {type(e)}")
    model = None

def validate_sensor_data(temp: float, humid: float, press: float) -> bool:
    """Validate sensor data ranges"""
    return (
        -50 <= temp <= 50 and  # Temperature range in Celsius
        0 <= humid <= 100 and  # Humidity range in percentage
        800 <= press <= 1200   # Pressure range in hPa
    )

def make_prediction(temp: float, humid: float, press: float) -> tuple[int, str]:
    """Make prediction using the model"""
    try:
        if model is None:
            logger.error("Model is not loaded")
            raise ValueError("Model not loaded")
            
        # Log model information
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model features: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'No feature names'}")
        
        # Create feature names to match the model's expectations
        feature_names = ['Temperature', 'Humidity', 'Pressure']
        input_features = np.array([[temp, humid, press]])
        
        # Log input features
        logger.info(f"Input features shape: {input_features.shape}")
        logger.info(f"Input features: {input_features}")
        
        # Convert to DataFrame with feature names
        input_df = pd.DataFrame(input_features, columns=feature_names)
        logger.info(f"Input DataFrame:\n{input_df}")
        
        # Get prediction
        try:
            prediction = model.predict(input_df)[0]
            logger.info(f"Raw prediction value: {prediction}")
            logger.info(f"Prediction type: {type(prediction)}")
            
            # Ensure prediction is an integer
            prediction_int = int(prediction)
            logger.info(f"Converted prediction to integer: {prediction_int}")
            
            # Map prediction to label
            prediction_label = "Rain" if prediction_int == 1 else "Sunny"
            logger.info(f"Final prediction label: {prediction_label}")
            
            return prediction_int, prediction_label
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return 0, "Error"
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return 0, "Error"

@app.route('/data', methods=['POST'])
def receive_data():
    """
    Endpoint untuk menerima data JSON dari ESP32.
    JSON:
    {
        "value1": <float> (Temperature),
        "value2": <float> (Humidity),
        "value3": <float> (Pressure)
    }
    """
    logger.info("Received POST request to /data endpoint")
    try:
        # Log raw request data
        logger.info(f"Request data: {request.get_data()}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        payload = request.get_json()
        logger.info(f"Parsed JSON payload: {payload}")
        
        if not payload:
            logger.error("Invalid JSON payload received")
            return jsonify({"error": "Invalid JSON"}), 400

        # Extract and validate data
        try:
            temp = float(payload.get('value1', 0))
            humid = float(payload.get('value2', 0))
            press = float(payload.get('value3', 0))
            logger.info(f"Parsed sensor values - Temp: {temp}, Humid: {humid}, Press: {press}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing sensor values: {str(e)}")
            return jsonify({"error": "Invalid sensor values"}), 400

        # Validate sensor data ranges
        if not validate_sensor_data(temp, humid, press):
            logger.error(f"Sensor values out of range - Temp: {temp}, Humid: {humid}, Press: {press}")
            return jsonify({"error": "Sensor values out of valid range"}), 400

        # Make prediction
        logger.info("Making prediction with model...")
        prediction_int, prediction_label = make_prediction(temp, humid, press)
        logger.info(f"Prediction result - Value: {prediction_int}, Label: {prediction_label}")

        # Build entry
        entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "value1": round(temp, 2),
            "value2": round(humid, 2),
            "value3": round(press, 2),
            "prediction": prediction_int,
            "prediction_label": prediction_label
        }
        logger.info(f"Created database entry: {entry}")

        # Save to MongoDB
        if collection is not None:
            try:
                result = collection.insert_one(entry)
                logger.info(f"Data saved to MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {str(e)}")
        else:
            logger.error("MongoDB collection not available")

        response_data = {
            "status": "success",
            "prediction": prediction_int,
            "label": prediction_label
        }
        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def index():
    """Dashboard halaman utama"""
    # Get last 100 records from MongoDB
    if collection is not None:  # Fixed collection check
        data_history = list(collection.find().sort('timestamp', -1).limit(100))
        # Convert ObjectId to string for JSON serialization
        for item in data_history:
            item['_id'] = str(item['_id'])
    else:
        data_history = []
    
    return render_template('index.html', data_history=data_history)

if __name__ == '__main__':
    # Jalankan di semua IP, cocok untuk ESP32 akses lokal
    app.run(host='0.0.0.0', port=5000, debug=True)
