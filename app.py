from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model dari file .pkl
MODEL_PATH = "bnn1.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return jsonify({"message": "Model backend is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        #Ambil data dari request
        input_data = request.json.get("input", [])
        if not input_data or not isinstance(input_data, list):
            return jsonify({"error": "Input harus berupa array 2D"}), 400

        #Konversi ke numpy array
        input_array = np.array(input_data)

        #Lakukan prediksi
        predictions = model.predict(input_array)
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
