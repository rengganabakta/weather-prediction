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
import paho.mqtt.client as mqtt
import json
from flask_socketio import SocketIO
import time
import threading
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Current working directory: %s", os.getcwd())
logger.info("Environment variables loaded: %s", os.environ.get('MONGODB_URI') is not None)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# MQTT settings
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "weather_station/data"
MQTT_KEEPALIVE = 60
MQTT_RECONNECT_DELAY = 5

# MongoDB connection
try:
    mongodb_uri = os.getenv('MONGODB_URI')
    logger.info("MongoDB URI found: %s", mongodb_uri is not None)
    
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
        
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    db = client['weather_prediction']
    collection = db['sensor_data']
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    client = None
    collection = None

# MQTT Client setup
mqtt_client = None
mqtt_connected = False

def setup_mqtt():
    global mqtt_client, mqtt_connected
    try:
        mqtt_client = mqtt.Client()
        
        @mqtt_client.on_connect
        def on_connect(client, userdata, flags, rc):
            global mqtt_connected
            if rc == 0:
                logger.info("Connected to MQTT broker")
                mqtt_connected = True
                client.subscribe(MQTT_TOPIC)
                logger.info(f"Subscribed to topic: {MQTT_TOPIC}")
            else:
                logger.error(f"Failed to connect to MQTT broker with code: {rc}")
                mqtt_connected = False

        @mqtt_client.on_disconnect
        def on_disconnect(client, userdata, rc):
            global mqtt_connected
            logger.warning(f"Disconnected from MQTT broker with code: {rc}")
            mqtt_connected = False

        @mqtt_client.on_message
        def on_message(client, userdata, msg):
            try:
                logger.info(f"Received message on topic {msg.topic}")
                logger.info(f"Message payload: {msg.payload.decode()}")
                
                payload = json.loads(msg.payload.decode())
                logger.info(f"Parsed payload: {payload}")
                
                # Extract sensor data
                temp = float(payload.get('value1', 0))
                humid = float(payload.get('value2', 0))
                press = float(payload.get('value3', 0))
                
                logger.info(f"Sensor values - Temp: {temp}, Humid: {humid}, Press: {press}")
                
                # Make prediction
                prediction_int, prediction_label = make_prediction(temp, humid, press)
                logger.info(f"Prediction - Value: {prediction_int}, Label: {prediction_label}")
                
                # Save to database
                entry = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "value1": round(temp, 2),
                    "value2": round(humid, 2),
                    "value3": round(press, 2),
                    "prediction": prediction_int,
                    "prediction_label": prediction_label
                }
                
                if collection is not None:
                    result = collection.insert_one(entry)
                    logger.info(f"Data saved to MongoDB with ID: {result.inserted_id}")
                
                # Emit to websocket
                socketio.emit('sensor_update', entry)
                logger.info("Data emitted to websocket")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing MQTT message: {str(e)}")

        # Start MQTT client
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        mqtt_client.loop_start()
        logger.info("MQTT client started")
        
    except Exception as e:
        logger.error(f"Error setting up MQTT client: {str(e)}")
        mqtt_client = None

# Load the trained AI model
try:
    model = joblib.load('model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def make_prediction(temp: float, humid: float, press: float) -> tuple[int, str]:
    """Make prediction using the model"""
    try:
        if model is None:
            logger.error("Model is not loaded")
            return 0, "Error"
            
        # Create feature names to match the model's expectations
        feature_names = ['Temperature', 'Humidity', 'Pressure']
        input_features = np.array([[temp, humid, press]])
        
        # Convert to DataFrame with feature names
        input_df = pd.DataFrame(input_features, columns=feature_names)
        
        # Get prediction
        prediction = model.predict(input_df)[0]
        prediction_int = int(prediction)
        prediction_label = "Rain" if prediction_int == 1 else "Sunny"
        
        return prediction_int, prediction_label
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return 0, "Error"

def pull_data_from_database():
    """Function to pull data from database and publish to MQTT"""
    while True:
        try:
            if collection is not None and mqtt_connected:
                # Get the latest record from database
                latest_record = collection.find_one(sort=[('timestamp', -1)])
                if latest_record:
                    # Remove MongoDB _id
                    latest_record.pop('_id', None)
                    
                    # Convert to JSON
                    json_data = json.dumps(latest_record)
                    
                    # Publish to MQTT
                    mqtt_client.publish(MQTT_TOPIC, json_data)
                    logger.info(f"Published latest data to MQTT: {json_data}")
            
            # Wait for 5 seconds before next pull
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error pulling data from database: {str(e)}")
            time.sleep(5)

@app.route('/data', methods=['POST'])
def receive_data():
    """Endpoint untuk menerima data JSON dari ESP32"""
    try:
        payload = request.get_json()
        logger.info(f"Received POST request with payload: {payload}")
        
        if not payload:
            return jsonify({"error": "Invalid JSON"}), 400

        # Extract and validate data
        temp = float(payload.get('value1', 0))
        humid = float(payload.get('value2', 0))
        press = float(payload.get('value3', 0))

        # Make prediction
        prediction_int, prediction_label = make_prediction(temp, humid, press)

        # Save to database
        entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "value1": round(temp, 2),
            "value2": round(humid, 2),
            "value3": round(press, 2),
            "prediction": prediction_int,
            "prediction_label": prediction_label
        }

        if collection is not None:
            result = collection.insert_one(entry)
            logger.info(f"Data saved to MongoDB with ID: {result.inserted_id}")

        # Emit to websocket
        socketio.emit('sensor_update', entry)

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
    try:
        # Get last 100 records from MongoDB
        if collection is not None:
            data_history = list(collection.find().sort('timestamp', -1).limit(100))
            # Convert ObjectId to string for JSON serialization
            for item in data_history:
                item['_id'] = str(item['_id'])
                # Ensure prediction_label exists
                if 'prediction_label' not in item:
                    item['prediction_label'] = "Sunny" if item.get('prediction', 0) == 0 else "Rain"
        else:
            data_history = []
        
        # Force garbage collection
        gc.collect()
        
        return render_template('index.html', data_history=data_history)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return render_template('index.html', data_history=[])

if __name__ == '__main__':
    # Setup MQTT
    setup_mqtt()
    
    # Start database pull thread
    db_pull_thread = threading.Thread(target=pull_data_from_database)
    db_pull_thread.daemon = True
    db_pull_thread.start()
    
    # Run Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
