from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import pymysql

# Inisialisasi Flask
app = Flask(__name__)

# Path model
MODEL_PATH = "bnn1.pkl"
model = joblib.load(MODEL_PATH)

# Cloud SQL connection settings
CLOUD_SQL_CONNECTION_NAME = "artful-mystery-441112-u2:asia-southeast2:db-cow-predict"  # Ganti dengan nama instance Cloud SQL Anda
DB_USER = "rifal"  # Ganti dengan username MySQL Anda
DB_PASSWORD = "12345678"  # Ganti dengan password MySQL Anda
DB_NAME = "db-cow-predict"  # Ganti dengan nama database Anda

# Fungsi untuk mendapatkan koneksi database menggunakan unix socket
def get_db_connection():
    unix_socket = f"/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"
    connection = pymysql.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        unix_socket=unix_socket
    )
    return connection

@app.route("/")
def index():
    return jsonify({"message": "Model backend is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari request
        input_data = request.json.get("input", [])
        if not input_data or not isinstance(input_data, list):
            return jsonify({"error": "Input harus berupa array 2D"}), 400

        # Pastikan input_data memiliki dua kolom untuk lebar_dada dan panjang_badan
        if len(input_data[0]) != 2:
            return jsonify({"error": "Input harus memiliki dua kolom: lebar_dada dan panjang_badan"}), 400

        # Ambil data tambahan
        bobot_real = request.json.get("bobot_real")
        suhu_badan = request.json.get("suhu_badan")

        if None in (bobot_real, suhu_badan):
            return jsonify({"error": "Semua parameter tambahan harus diisi"}), 400

        # Konversi ke numpy array
        input_array = np.array(input_data)

        # Lakukan prediksi
        predictions = model.predict(input_array)

        # Menyimpan hasil prediksi ke MySQL Cloud SQL
        connection = get_db_connection()
        cursor = connection.cursor()

        query = """
        INSERT INTO prediction (lebar_dada, panjang_badan, bobot_real, suhu_badan, prediction)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (input_data[0][0], input_data[0][1], bobot_real, suhu_badan, predictions[0]))
        connection.commit()

        # Buat response
        response = {
            "predictions": predictions.tolist(),
            "input_details": {
                "lebar_dada": input_data[0][0],
                "panjang_badan": input_data[0][1],
                "bobot_real": bobot_real,
                "suhu_badan": suhu_badan,
            }
        }

        # Menutup koneksi database
        cursor.close()
        connection.close()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)