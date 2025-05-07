#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <DHT.h>

// Konfigurasi WiFi
const char* ssid = "Biznet Lt1 4G";
const char* password = "@LantaiSatu";

// API endpoint
const char* serverUrl = "http://192.168.0.109:8080/data";  

// Pin DHT11
#define DHTPIN 4       // Ganti sesuai pin kamu
#define DHTTYPE DHT11

// Objek sensor
Adafruit_BMP280 bmp;   // I2C default
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  delay(500);

  // WiFi
  WiFi.begin(ssid, password);
  Serial.println("\n=== Koneksi WiFi ===");
  Serial.print("Menghubungkan ke WiFi: ");
  Serial.println(ssid);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi terkoneksi!");
    Serial.print("IP Address ESP32: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println("\nGagal terhubung ke WiFi!");
  }
  Serial.println("===================\n");

  // Inisialisasi BMP280
  if (!bmp.begin(0x76)) {
    Serial.println("BMP280 tidak terdeteksi!");
    while (1);
  }

  // Inisialisasi DHT11
  dht.begin();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // Baca sensor
    float temperature = bmp.readTemperature();
    float pressure = bmp.readPressure() / 100.0F;  // dalam hPa
    float humidity = dht.readHumidity();

    // Cek validitas data DHT11
    if (isnan(humidity)) {
      Serial.println("Gagal membaca dari sensor DHT11");
      return;
    }

    // Tampilkan di Serial Monitor
    Serial.printf("Suhu: %.2f °C, Tekanan: %.2f hPa, Kelembapan: %.2f %%\n", temperature, pressure, humidity);

    // Buat JSON payload
    String payload = "{";
    payload += "\"value1\":" + String(temperature, 2) + ",";  // temperature
    payload += "\"value2\":" + String(humidity, 2) + ",";     // humidity
    payload += "\"value3\":" + String(pressure, 2);           // pressure
    payload += "}";

    // Kirim ke server
    HTTPClient http;
    
    Serial.println("\n=== Koneksi ke Server ===");
    Serial.print("Mencoba koneksi ke: ");
    Serial.println(serverUrl);
    
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000);  // Naikkan timeout ke 10 detik

    Serial.println("Payload: " + payload);
    Serial.println("Mengirim POST request...");

    int httpCode = http.POST(payload);
    
    if (httpCode > 0) {
      String response = http.getString();
      Serial.printf("HTTP Response Code: %d\n", httpCode);
      Serial.printf("Response: %s\n", response.c_str());
    } else {
      Serial.printf("Error Code: %d\n", httpCode);
      Serial.printf("Error: %s\n", http.errorToString(httpCode).c_str());
      
      // Cek status WiFi
      Serial.println("\nStatus WiFi:");
      Serial.printf("SSID: %s\n", WiFi.SSID().c_str());
      Serial.printf("RSSI: %d dBm\n", WiFi.RSSI());
      Serial.printf("IP: %s\n", WiFi.localIP().toString().c_str());
      Serial.printf("Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
      Serial.printf("Subnet: %s\n", WiFi.subnetMask().toString().c_str());
      
      // Coba reconnect WiFi jika error
      if (httpCode == -11) {  // Timeout error
        Serial.println("\nMencoba reconnect WiFi...");
        WiFi.disconnect();
        delay(1000);
        WiFi.begin(ssid, password);
        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
          delay(500);
          Serial.print(".");
          attempts++;
        }
        if (WiFi.status() == WL_CONNECTED) {
          Serial.println("\nWiFi reconnected!");
        } else {
          Serial.println("\nWiFi reconnection failed!");
        }
      }
    }
    Serial.println("===================\n");
    http.end();

    // Tambah delay lebih lama jika error
    if (httpCode <= 0) {
      delay(15000);  // Tunggu 15 detik jika error
    } else {
      delay(5000);   // Tunggu 5 detik jika sukses
    }
  } else {
    Serial.println("WiFi terputus, reconnecting...");
    WiFi.reconnect();
  }
}
