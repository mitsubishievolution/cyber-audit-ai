# app.py
# Hello! This is the "AI Prediction Server" file. Think of it like a restaurant where
# people can come and ask our trained AI chef for predictions about cybersecurity risks.
#
# What this server does:
# - Loads our trained AI chef (Random Forest model)
# - Provides a menu (API endpoints) for customers
# - Takes vulnerability orders and gives risk predictions
# - Handles special requests and error situations
#
# Imagine a fancy restaurant where:
# - The chef (AI model) is already trained and ready
# - Waiters (API endpoints) take orders from customers
# - Orders are vulnerability descriptions, predictions are risk scores

from flask import Flask, jsonify, request  # Flask is our restaurant framework
from model_trainer import CybersecurityRiskModel  # Our AI chef
from feature_engineering import VulnerabilityFeatureEngineer  # Our data translator
import os  # For file operations
import numpy as np  # For number operations

app = Flask(__name__)

# Global variables to hold the trained model and feature engineer
trained_model = None
feature_engineer = None

def transform_single_vulnerability(vulnerability_data, saved_feature_engineer):
    """
    Transform a single vulnerability using the already-fitted feature engineer.

    This ensures we use the same TF-IDF vocabulary and one-hot encoding as training,
    preventing the "111 features vs 11 features" mismatch.

    Args:
        vulnerability_data (list): Single vulnerability dictionary
        saved_feature_engineer: The fitted feature engineer from training

    Returns:
        dict: Processed features in same format as training
    """
    if not vulnerability_data or len(vulnerability_data) != 1:
        raise ValueError("transform_single_vulnerability expects exactly one vulnerability")

    vuln = vulnerability_data[0]

    # Extract individual components
    description = vuln['description']
    vulnerability_type = vuln['type']
    cvss_score = vuln['cvss']

    # Use the already-fitted TF-IDF vectorizer (same vocabulary as training)
    text_features = saved_feature_engineer.vectorizer.transform([description]).toarray()

    # Use the already-fitted one-hot encoder (same categories as training)
    type_features = saved_feature_engineer.encoder.transform([[vulnerability_type]])

    # CVSS score as feature (same as training)
    cvss_features = np.array([[cvss_score]])

    # Combine features in same order as training
    combined_features = np.hstack([
        text_features,      # TF-IDF text features
        type_features,      # One-hot encoded types
        cvss_features       # CVSS score
    ])

    return {
        'features': combined_features,
        'targets': np.array([cvss_score]),  # Not used for prediction, but for consistency
        'feature_names': saved_feature_engineer.feature_names,
        'original_data': vulnerability_data
    }

def load_trained_model():
    """
    This is the "Restaurant Opening" function! It gets called when the restaurant
    first opens its doors. It makes sure our AI chef and kitchen tools are ready.

    Think of it like a restaurant manager who:
    - Checks if the chef (AI model) is available
    - Checks if the kitchen tools (feature engineer) are ready
    - Makes sure everything is set up before opening for customers

    Returns:
        bool: True if restaurant is ready, False if something is missing
    """
    global trained_model, feature_engineer

    model_path = "model/trained_model.pkl"  # Where our trained chef is stored

    # Check if our trained chef exists
    if not os.path.exists(model_path):
        print(f"Warning: Trained chef not found at {model_path}")
        print("Please run model_trainer.py first to train the AI chef.")
        return False

    try:
        # Bring in the trained chef and kitchen tools
        trained_model, feature_engineer = CybersecurityRiskModel.load_model(model_path)

        # Make sure we have both the chef AND the tools
        if feature_engineer is None:
            print("❌ Error: Kitchen tools (feature engineer) not found with the chef")
            print("Please retrain the model with the updated code.")
            return False

        print(" AI chef loaded successfully!")
        print(" Kitchen tools loaded successfully!")
        print(" Restaurant is now open for predictions!")
        return True

    except Exception as e:
        print(f"❌ Error opening restaurant: {e}")
        return False

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the API is running and model is loaded.
    """
    model_status = "loaded" if trained_model and trained_model.is_trained else "not loaded"

    return jsonify({
        "status": "healthy",
        "message": "Cybersecurity Audit AI API is running",
        "model_status": model_status,
        "endpoints": {
            "/predict": "POST - Make risk predictions",
            "/health": "GET - Check API status"
        }
    })

@app.route("/predict", methods=["POST"])
def predict_risk():
    """
    This is the main "Order Taking" endpoint - the most important function in our restaurant!
    Customers (other computer programs) send us vulnerability orders and we give them predictions.

    Think of it like a waiter who:
    1. Takes the customer's order (vulnerability description)
    2. Gives it to the chef (AI model) to prepare
    3. Serves the finished dish (risk prediction) back to the customer

    Expected customer order format:
    {
        "description": "SQL injection vulnerability in login form...",
        "type": "SQL Injection",
        "cvss": 7.5
    }

    Returns the chef's prediction:
    {
        "risk_score": 7.8,
        "confidence": "high",
        "explanation": "Why the chef made this prediction"
    }
    """
    global trained_model, feature_engineer

    # Check if our restaurant is actually open (chef and kitchen ready)
    if not trained_model or not trained_model.is_trained:
        return jsonify({
            "error": "Restaurant not open",
            "message": "Please train the AI chef first by running model_trainer.py"
        }), 503

    try:
        # Take the customer's order (JSON data)
        customer_order = request.get_json()

        if not customer_order:
            return jsonify({
                "error": "Empty order",
                "message": "Please provide vulnerability information in your order"
            }), 400

        # Check that the order has all required ingredients
        required_ingredients = ['description', 'type', 'cvss']
        for ingredient in required_ingredients:
            if ingredient not in customer_order:
                return jsonify({
                    "error": f"Missing ingredient: {ingredient}",
                    "message": f"Please include '{ingredient}' in your vulnerability order"
                }), 400

        # Check that the danger score is reasonable (0-10 scale)
        if not isinstance(customer_order['cvss'], (int, float)) or not (0 <= customer_order['cvss'] <= 10):
            return jsonify({
                "error": "Invalid danger score",
                "message": "CVSS score must be a number between 0 and 10"
            }), 400

        # Prepare the vulnerability for the chef
        vulnerability_recipe = [{
            'cve': 'customer-order',  # Just a label for this order
            'description': customer_order['description'],
            'type': customer_order['type'],
            'cvss': customer_order['cvss']
        }]

        # Use our kitchen tools to prepare the ingredients exactly like training
        prepared_dish = transform_single_vulnerability(vulnerability_recipe, feature_engineer)

        if prepared_dish is None:
            return jsonify({
                "error": "Kitchen error",
                "message": "Could not prepare your vulnerability for analysis"
            }), 500

        # Send to the chef for cooking (prediction)
        ingredients = prepared_dish['features']
        risk_score = trained_model.predict(ingredients)[0]

        # Figure out how confident the chef is in this prediction
        confidence = "medium"  # Default confidence level
        prediction_difference = abs(risk_score - customer_order['cvss'])

        if prediction_difference <= 1.0:
            confidence = "high"    # Chef is very sure (within 1 point)
        elif prediction_difference <= 2.0:
            confidence = "medium"  # Chef is reasonably sure (within 2 points)
        else:
            confidence = "low"     # Chef is not very confident

        # Generate a friendly explanation for the customer
        explanation = generate_prediction_explanation(customer_order, risk_score, confidence)

        # Serve the finished dish back to the customer
        return jsonify({
            "risk_score": round(float(risk_score), 2),  # The main prediction
            "confidence": confidence,                    # How sure the chef is
            "explanation": explanation,                  # Why this prediction
            "input_summary": {                          # Order summary
                "type": customer_order['type'],
                "cvss_score": customer_order['cvss'],
                "description_length": len(customer_order['description'])
            }
        })

    except Exception as e:
        print(f"Kitchen accident: {e}")
        return jsonify({
            "error": "Service error",
            "message": f"Something went wrong in the kitchen: {str(e)}"
        }), 500

def generate_prediction_explanation(input_data, predicted_score, confidence):
    """
    Generate a human-readable explanation of the AI prediction.

    Args:
        input_data (dict): Original vulnerability data
        predicted_score (float): AI predicted risk score
        confidence (str): Confidence level

    Returns:
        str: Explanation text
    """
    vuln_type = input_data['type']
    cvss_score = input_data['cvss']
    score_diff = abs(predicted_score - cvss_score)

    explanation = f"The AI model predicts a risk score of {predicted_score:.1f} "

    if confidence == "high":
        explanation += f"with high confidence, closely matching the input CVSS score of {cvss_score}."
    elif confidence == "medium":
        explanation += f"with medium confidence. The prediction differs from the CVSS score by {score_diff:.1f} points."
    else:
        explanation += f"with lower confidence. There is a notable difference from the CVSS score of {score_diff:.1f} points."

    explanation += f" This assessment considers the {vuln_type.lower()} vulnerability type and description text."

    return explanation

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested URL was not found on this server"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Load the model when the app starts
if __name__ == "__main__":
    print("🔄 Starting Cybersecurity Audit AI Flask API...")

    if load_trained_model():
        print("🌐 Starting Flask server on http://127.0.0.1:5000")
        print("📖 API Documentation:")
        print("   GET  /health  - Check API status")
        print("   POST /predict - Make risk predictions")
        print("\nExample prediction request:")
        print('curl -X POST http://127.0.0.1:5000/predict \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"description": "SQL injection vulnerability", "type": "SQL Injection", "cvss": 7.5}\'')

    app.run(host="0.0.0.0", port=5000, debug=True)