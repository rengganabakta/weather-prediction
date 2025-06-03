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
