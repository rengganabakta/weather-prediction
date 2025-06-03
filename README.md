# 🌦️ AIoT Weather Predictor Dashboard

**ESP32 + Flask + Machine Learning + Railway**

Proyek ini adalah sistem **AIoT (Artificial Intelligence of Things)** yang memanfaatkan sensor dari ESP32 untuk mengirim data lingkungan seperti **suhu (temperature)**, **kelembaban (humidity)**, dan **tekanan udara (pressure)** ke sebuah server. Server ini menggunakan **model machine learning** untuk memprediksi apakah akan terjadi **hujan di hari berikutnya** (`Rain Tomorrow`). Hasil prediksi ditampilkan secara real-time di dashboard web yang otomatis **refresh setiap 5 detik**.

---

## 📡 Alur Sistem

1. **ESP32** mengirimkan data sensor melalui HTTP POST (dalam format JSON).
2. **Server Flask** menerima data dan menyimpannya secara in-memory.
3. Server melakukan **prediksi cuaca** menggunakan model ML yang telah dilatih (`Random Forest Classifier`).
4. Dashboard web menampilkan histori data & hasil prediksi secara otomatis.

---

## 🧠 Model Machine Learning

- Algoritma: `Random Forest Classifier`
- Dataset: CSV berisi kolom `Temperature`, `Humidity`, `Pressure`, dan `Rain Tomorrow`
- Akurasi: ~77%
- Parameter terbaik:
  ```python
  {
    "n_estimators": 150,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1
  }
  ```

````

# Weather Prediction System

## Prerequisites
- Python 3.8 or higher
- MongoDB Atlas account
- MQTT Broker (using HiveMQ public broker)

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd app
````

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:
   Create a `.env` file in the app directory with the following content:

```
MONGODB_URI=your_mongodb_connection_string
FLASK_ENV=production
FLASK_APP=app.py
```

5. Run the application:

```bash
python app.py
```

## Deployment Checklist

1. Environment Variables:

   - [ ] Set MONGODB_URI in .env file
   - [ ] Verify FLASK_ENV is set to production
   - [ ] Ensure FLASK_APP is set to app.py

2. Dependencies:

   - [ ] All required packages are in requirements.txt
   - [ ] Python version is compatible (3.8+)
   - [ ] Virtual environment is activated

3. Files Required:

   - [ ] app.py
   - [ ] requirements.txt
   - [ ] .env
   - [ ] model.pkl
   - [ ] templates/index.html

4. Database:

   - [ ] MongoDB Atlas account is active
   - [ ] Connection string is valid
   - [ ] Database and collection are created

5. MQTT:
   - [ ] Using HiveMQ public broker (broker.hivemq.com)
   - [ ] Port 1883 is accessible
   - [ ] Topic "weather_station/data" is available

## Troubleshooting

1. Database Connection Issues:

   - Check MongoDB URI in .env file
   - Verify network connectivity
   - Check MongoDB Atlas IP whitelist

2. MQTT Connection Issues:

   - Verify internet connection
   - Check if broker.hivemq.com is accessible
   - Ensure port 1883 is not blocked

3. Model Loading Issues:

   - Verify model.pkl exists in the correct directory
   - Check model compatibility with scikit-learn version

4. WebSocket Issues:
   - Ensure eventlet is properly installed
   - Check if port 8080 is available
   - Verify CORS settings

## Production Deployment

For production deployment, it's recommended to use Gunicorn:

```bash
gunicorn --worker-class eventlet -w 1 app:app -b 0.0.0.0:8080
```

## Security Considerations

1. Change the default MQTT broker to a private one in production
2. Use environment variables for sensitive information
3. Implement proper authentication for MongoDB
4. Use HTTPS in production
5. Implement rate limiting for API endpoints
