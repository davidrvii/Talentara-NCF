# app.py

from flask import Flask, request, jsonify
from inference import predict_match

app = Flask(__name__)

# Route utama untuk health check
@app.route("/", methods=["GET"])
def home():
    return "✅ NCF API is running."

# Route untuk predict
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        project_features = data["project"]
        talent_features = data["talent"]

        # Jalankan predict
        score = predict_match(project_features, talent_features)

        # Return response
        response = {
            "score": float(score)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
