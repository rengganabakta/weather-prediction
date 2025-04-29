from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained AI model
model = joblib.load('models/model.pkl')

# In-memory storage for received data
data_history = []

# Simple HTML template to display the data
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>ESP32 Data Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        th { background: #f4f4f4; }
    </style>
</head>
<body>
    <h1>ESP32 Data Dashboard</h1>
    <p>Received {{ data_history | length }} entries.</p>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Temperature (°C)</th>
                <th>Humidity (%)</th>
                <th>Pressure (hPa)</th>
                <th>Prediction</th>
            </tr>
        </thead>
        <tbody>
            {% for entry in data_history %}
            <tr>
                <td>{{ entry.timestamp }}</td>
                <td>{{ entry.value1 }}</td>
                <td>{{ entry.value2 }}</td>
                <td>{{ entry.value3 }}</td>
                <td>{{ entry.prediction }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
'''

@app.route('/data', methods=['POST'])
def receive_data():
    """
    Endpoint for ESP32 to POST JSON data:
    {
        "value1": <float> (Temperature),
        "value2": <float> (Humidity),
        "value3": <float> (Pressure)
    }
    """
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Invalid JSON"}), 400

    temp = payload.get('value1')
    humid = payload.get('value2')
    press = payload.get('value3')

    if temp is None or humid is None or press is None:
        return jsonify({"error": "Missing values"}), 400

    # Prepare input for the model
    input_features = np.array([[temp, humid, press]])

    # Make a prediction
    prediction = model.predict(input_features)[0]

    # Add timestamp and store
    entry = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        "value1": temp,
        "value2": humid,
        "value3": press,
        "prediction": prediction
    }
    data_history.insert(0, entry)

    return jsonify({"status": "success", "prediction": prediction}), 200

@app.route('/')
def index():
    """Render the dashboard with the latest data entries"""
    return render_template_string(HTML_TEMPLATE, data_history=data_history)

if __name__ == '__main__':
    # Run the app on all IPs, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
