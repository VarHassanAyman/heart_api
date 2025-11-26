from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
feature_names = None

def load_model():
    global model, scaler, feature_names
    try:
        if not os.path.exists("model.pkl"):
            return False
        
        model_data = joblib.load("model.pkl")
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
if not load_model():
    print("Warning: Server will run without model!")

@app.route("/", methods=["GET"])
def home():
    if model is None:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "message": "Model not loaded"
        }), 503
    
    return jsonify({
        "status": "success",
        "model_loaded": True,
        "features": feature_names,
        "message": "Heart Disease API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json(force=True)
        
        # Check required fields
        missing_fields = [field for field in feature_names if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing fields: {missing_fields}",
                "required_fields": feature_names
            }), 400
        
        # Process data
        df = pd.DataFrame([data], columns=feature_names)
        X = scaler.transform(df)
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        result = {
            "prediction": int(prediction),
            "probability": float(probabilities[1] * 100),
            "details": {
                "no_disease": float(probabilities[0] * 100),
                "disease": float(probabilities[1] * 100)
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/predict", "/health"]
    }), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

