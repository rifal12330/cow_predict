from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import pymysql
import logging

# Inisialisasi Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Path model
MODEL_PATH = "bnn1.pkl"
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file {MODEL_PATH} not found.")
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
model = joblib.load(MODEL_PATH)

# Cloud SQL connection settings
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "artful-mystery-441112-u2:asia-southeast2:db-cow-predict")
DB_USER = os.getenv("DB_USER", "rifal")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345678")
DB_NAME = os.getenv("DB_NAME", "db-cow-predict")

# Fungsi untuk mendapatkan koneksi database menggunakan unix socket
def get_db_connection():
    try:
        # Cloud SQL instance connection string
        unix_socket = f"/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"
        connection = pymysql.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            unix_socket=unix_socket  # This is the Cloud SQL socket connection
        )
        return connection
    except pymysql.MySQLError as e:
        logging.error(f"Error connecting to database: {e}")
        raise e

@app.route("/")
def index():
    return jsonify({"message": "Model backend is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari request
        input_data = request.json.get("lebar dada & panjang badan", [])
        if not input_data or not isinstance(input_data, list):
            return jsonify({"error": "Input harus berupa array 2D"}), 400

        # Pastikan input_data memiliki dua kolom untuk lebar_dada dan panjang_badan
        if len(input_data[0]) != 2:
            return jsonify({"error": "Input harus memiliki dua kolom: lebar_dada dan panjang_badan"}), 400

        # Ambil data tambahan
        bobot_real = request.json.get("bobot_real")
        suhu_badan = request.json.get("suhu_badan")

        if bobot_real is None or suhu_badan is None:
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
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
