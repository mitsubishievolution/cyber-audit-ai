# app.py
# This is a lightweight Flask microservice that will eventually
# host your AI model for risk scoring vulnerabilities.
# For now, it simply returns a placeholder JSON response to confirm it's running.

from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    # This is a placeholder endpoint.
    # Later, you'll send vulnerability data (e.g., from OWASP ZAP/Nmap)
    # and return a computed risk score.
    return jsonify({
        "score": 42,
        "explanation": "This is a sample AI response — the model isn't trained yet."
    })

if __name__ == "__main__":
    # Runs the app locally for testing.
    # Use http://127.0.0.1:5000/predict to check output.
    app.run(host="0.0.0.0", port=5000, debug=True)
