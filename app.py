# app.py

import os
from flask import Flask, request, jsonify
from inference import predict_match, rank_talent_for_project

app = Flask(__name__)

# === Route utama → health check ===
@app.route("/test", methods=["GET"])
def home():
    return "NCF API is running."

# === Route → predict single match ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("📦 Request Payload:", data)

        project_features = data["project"]
        talent_features = data["talent"]

        # Run predict
        score = predict_match(project_features, talent_features)

        # Return response
        response = {
            "score": float(score)
        }

        print(f"🎯 Final Score: {score:.6f}")
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Route → rank talents ===
@app.route("/rank_talent", methods=["POST"])
def rank_talent():
    try:
        data = request.json
        print("📦 Request Payload:", data)

        project_features = data["project"]
        talents = data["talents"]  # list of talent dict

        # Run rank_talent_for_project
        ranked_result = rank_talent_for_project(project_features, talents)

        # Return response
        return jsonify(ranked_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
