# inference.py

import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load model ===
model_path = "best_ncf_with_embedding_improved.h5"
model = tf.keras.models.load_model(model_path)

# === Load Mapping ===
with open("mapping_platform.json", "r") as f:
    mapping_platform = json.load(f)

with open("mapping_product.json", "r") as f:
    mapping_product = json.load(f)

with open("mapping_role.json", "r") as f:
    mapping_role = json.load(f)

with open("mapping_language.json", "r") as f:
    mapping_language = json.load(f)

with open("mapping_tools.json", "r") as f:
    mapping_tools = json.load(f)

# === Load Maxlen ===
with open("maxlen.json", "r") as f:
    maxlen_dict = json.load(f)

# === Helper function: encode and pad ===
def encode_and_pad(list_values, mapping, maxlen):
    # Ubah list string → list index
    sequence = [mapping.get(val, 0) for val in list_values]
    # Pad ke maxlen
    padded = pad_sequences([sequence], maxlen=maxlen, padding="post", truncating="post")
    return padded[0]  # ambil array 1 dimensi

# === Main function: predict match ===
def predict_match(project_features_dict, talent_features_dict):
    # Encode + pad untuk masing-masing fitur
    
    # Project
    X_proj_platform = encode_and_pad(project_features_dict["platform"], mapping_platform, maxlen_dict["platform"])
    X_proj_product  = encode_and_pad(project_features_dict["product"], mapping_product, maxlen_dict["product"])
    X_proj_role     = encode_and_pad(project_features_dict["role"], mapping_role, maxlen_dict["role"])
    X_proj_language = encode_and_pad(project_features_dict["language"], mapping_language, maxlen_dict["language"])
    X_proj_tools    = encode_and_pad(project_features_dict["tools"], mapping_tools, maxlen_dict["tools"])
    
    # Talent
    X_tal_platform = encode_and_pad(talent_features_dict["platform"], mapping_platform, maxlen_dict["platform"])
    X_tal_product  = encode_and_pad(talent_features_dict["product"], mapping_product, maxlen_dict["product"])
    X_tal_role     = encode_and_pad(talent_features_dict["role"], mapping_role, maxlen_dict["role"])
    X_tal_language = encode_and_pad(talent_features_dict["language"], mapping_language, maxlen_dict["language"])
    X_tal_tools    = encode_and_pad(talent_features_dict["tools"], mapping_tools, maxlen_dict["tools"])
    
    # Siapkan input list sesuai urutan model
    input_list = [
        np.array([X_proj_platform]),
        np.array([X_proj_product]),
        np.array([X_proj_role]),
        np.array([X_proj_language]),
        np.array([X_proj_tools]),
        np.array([X_tal_platform]),
        np.array([X_tal_product]),
        np.array([X_tal_role]),
        np.array([X_tal_language]),
        np.array([X_tal_tools])
    ]
    
    # Predict → return float
    score = model.predict(input_list, verbose=0)[0][0]  # ambil scalar float
    return score

# === Function: rank talent for project ===
def rank_talent_for_project(project_features_dict, list_of_talent_features_dicts):
    """
    Mengurutkan talent berdasarkan score kecocokan dengan project.
    """
    result = []
    
    for talent_features_dict in list_of_talent_features_dicts:
        talent_id = talent_features_dict["talent_id"]
        
        talent_features = {
            "platform": talent_features_dict.get("platform", []),
            "product": talent_features_dict.get("product", []),
            "role": talent_features_dict.get("role", []),
            "language": talent_features_dict.get("language", []),
            "tools": talent_features_dict.get("tools", [])
        }
        
        score = predict_match(project_features_dict, talent_features)
        
        result.append({
            "talent_id": talent_id,
            "score": float(score)
        })
    
    result_sorted = sorted(result, key=lambda x: x["score"], reverse=True)
    
    return result_sorted
