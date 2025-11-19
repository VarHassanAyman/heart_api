from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Enhanced CORS settings
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Global variables
model = None
scaler = None
feature_names = None

# Load model on startup
def load_model():
    global model, scaler, feature_names
    
    try:
        # Check if file exists
        if not os.path.exists("model.pkl"):
            print("❌ The model.pkl file does not exist!")
            return False
        
        # Load model
        model_data = joblib.load("model.pkl")
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        
        print("=" * 60)
        print("✅ The model has been loaded successfully!")
        print(f"📊 Number of features: {len(feature_names)}")
        print(f"📋 Feature names: {feature_names}")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load the model
if not load_model():
    print("⚠️ Warning: The server will run but without a model!")

@app.route("/", methods=["GET"])
def home():
    """Home page to check if the server is working"""
    if model is None:
        return jsonify({
            "status": "running",
            "model_loaded": False,
            "message": "⚠️ The server is working but the model is not loaded"
        }), 503
    
    return jsonify({
        "status": "running",
        "model_loaded": True,
        "features": feature_names,
        "message": "✅ Server and model are ready!"
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Prediction endpoint"""
    
    # Handle OPTIONS request (CORS Preflight)
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Accept")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200
    
    # Check if model is loaded
    if model is None or scaler is None:
        return jsonify({
            "error": "Model not loaded. Please check the model.pkl file"
        }), 500
    
    try:
        # Receive data
        data = request.get_json(force=True)
        print(f"\n📥 Data received from Flutter:")
        print(f"   {data}")
        
        # Check if all required fields exist
        missing_fields = [field for field in feature_names if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing fields: {missing_fields}",
                "required_fields": feature_names
            }), 400
        
        # Arrange data according to feature_names
        df = pd.DataFrame([data], columns=feature_names)
        print(f"📊 Created DataFrame:")
        print(df)
        
        # Scaling
        X = scaler.transform(df)
        print(f"🔄 Data after Scaling: {X}")
        
        # Prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Convert to percentage
        prob_no_disease = float(probabilities[0] * 100)
        prob_disease = float(probabilities[1] * 100)
        
        result = {
            "prediction": int(prediction),
            "probability": prob_disease,  # What Flutter expects
            "details": {
                "no_disease": prob_no_disease,
                "disease": prob_disease
            }
        }
        
        print(f"✅ Result sent:")
        print(f"   Classification: {prediction}")
        print(f"   Disease probability: {prob_disease:.2f}%")
        print("=" * 60)
        
        return jsonify(result), 200
        
    except KeyError as e:
        error_msg = f"Missing or incorrect field: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 400
        
    except ValueError as e:
        error_msg = f"Invalid value: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 400
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Server health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

# General error handler
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Path not found",
        "available_endpoints": ["/", "/predict", "/health"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Flask Server...")
    print("=" * 60)
    
    # Check files
    if os.path.exists("model.pkl"):
        size = os.path.getsize("model.pkl")
        print(f"📦 model.pkl file exists - Size: {size:,} bytes")
    else:
        print("❌ model.pkl file not found!")
    
    print("\n🌐 Server available at:")
    print("   - Local: http://192.168.1.2:5000")
    print("   - Network: http://0.0.0.0:5000")
    print("   - Emulator: http://10.0.2.2:5000")
    print("\n📡 Available endpoints:")
    print("   - GET  /          → Server information")
    print("   - POST /predict   → Prediction")
    print("   - GET  /health    → Health check")
    print("=" * 60 + "\n")
    
    # Run server
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
