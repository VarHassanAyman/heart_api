from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = None
scaler = None
feature_names = None

def load_model():
    global model, scaler, feature_names
    try:
        model_path = "model.pkl"

        if not os.path.exists(model_path):
            print("❌ model.pkl not found in server directory")
            return False
        
        model_data = joblib.load(model_path)
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]

        print("✅ Model loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
load_model()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "model_loaded": model is not None,
        "features": feature_names if model else [],
        "message": "Heart Disease Prediction API is running"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json(force=True)

        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        df = pd.DataFrame([data], columns=feature_names)
        X = scaler.transform(df)

        prediction = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]

        return jsonify({
            "prediction": prediction,
            "probability": float(probs[1] * 100),
            "details": {
                "no_disease": float(probs[0] * 100),
                "disease": float(probs[1] * 100)
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/predict", "/health"]
    }), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
