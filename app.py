from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
feature_names = None


def load_model():
    """
    Loads the saved ML model, scaler, and feature names.
    Handles sklearn version mismatch warnings safely.
    """
    global model, scaler, feature_names

    # Suppress sklearn InconsistentVersionWarning (we log instead)
    warnings.filterwarnings(
        "ignore",
        category=InconsistentVersionWarning
    )

    try:
        if not os.path.exists("model.pkl"):
            print("❌ model.pkl not found.")
            return False

        print("🔄 Loading model.pkl ...")

        model_data = joblib.load("model.pkl")

        model = model_data.get("model")
        scaler = model_data.get("scaler")
        feature_names = model_data.get("feature_names")

        # Basic validation
        if model is None or scaler is None or feature_names is None:
            print("❌ model.pkl is missing required components.")
            return False

        print("✅ Model loaded successfully.")
        print(f"📌 Feature count: {len(feature_names)}")
        return True

    except InconsistentVersionWarning:
        print("⚠️ WARNING: sklearn version mismatch detected!")
        return False

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# Load model on startup
if not load_model():
    print("⚠️ Server started WITHOUT a loaded model.")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success" if model else "error",
        "model_loaded": model is not None,
        "features": feature_names if model else None,
        "message": "Heart Disease API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    try:
        data = request.get_json(force=True)

        # Missing feature check
        missing_fields = [field for field in feature_names if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing_fields,
                "required_fields": feature_names
            }), 400

        # Convert to DataFrame
        df = pd.DataFrame([data], columns=feature_names)

        # Transform input using scaler
        X = scaler.transform(df)

        # Run prediction
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        response = {
            "prediction": prediction,
            "probability": float(probabilities[1] * 100),
            "details": {
                "no_disease": float(probabilities[0] * 100),
                "disease": float(probabilities[1] * 100)
            }
        }

        return jsonify(response), 200

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
