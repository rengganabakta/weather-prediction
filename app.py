from flask import Flask, request, jsonify, render_template
from datetime import datetime
import joblib
import numpy as np
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained AI model
try:
    model = joblib.load('model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# In-memory storage for received data with maximum size
MAX_HISTORY_SIZE = 100
data_history: List[Dict[str, Any]] = []

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
            raise ValueError("Model not loaded")
            
        input_features = np.array([[temp, humid, press]])
        prediction = model.predict(input_features)[0]
        prediction_int = int(prediction)
        prediction_label = "Rain" if prediction_int == 1 else "Sunny"
        return prediction_int, prediction_label
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
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
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid JSON"}), 400

        # Extract and validate data
        try:
            temp = float(payload.get('value1', 0))
            humid = float(payload.get('value2', 0))
            press = float(payload.get('value3', 0))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid sensor values"}), 400

        # Validate sensor data ranges
        if not validate_sensor_data(temp, humid, press):
            return jsonify({"error": "Sensor values out of valid range"}), 400

        # Make prediction
        prediction_int, prediction_label = make_prediction(temp, humid, press)

        # Build entry
        entry = {
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "value1": round(temp, 2),
            "value2": round(humid, 2),
            "value3": round(press, 2),
            "prediction": prediction_int,
            "prediction_label": prediction_label
        }

        # Add to history with size limit
        data_history.insert(0, entry)
        if len(data_history) > MAX_HISTORY_SIZE:
            data_history.pop()

        return jsonify({
            "status": "success",
            "prediction": prediction_int,
            "label": prediction_label
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def index():
    """Dashboard halaman utama"""
    return render_template('index.html', data_history=data_history)

if __name__ == '__main__':
    # Jalankan di semua IP, cocok untuk ESP32 akses lokal
    app.run(host='0.0.0.0', port=5000, debug=True)
