# inference.py

import tensorflow as tf
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from mapping_loader import load_mapping_from_db

# === Lazy load model ===
model = None

def get_model():
    global model
    if model is None:
        print("Loading model ...")
        model_path = "best_ncf_with_embedding_improved.keras"
        model = tf.keras.models.load_model(model_path)
        print("Model loaded.")
    return model

# === Load Mapping ===
mapping_platform = load_mapping_from_db("platform", "platform_name", "platform_id")
mapping_product  = load_mapping_from_db("product_type", "product_type_name", "product_type_id")
mapping_role     = load_mapping_from_db("role", "role_name", "role_id")
mapping_language = load_mapping_from_db("language", "language_name", "language_id")
mapping_tools    = load_mapping_from_db("tools", "tools_name", "tools_id")

#Debug
print("📚 Mapping sizes:")
print(f"Platform: {len(mapping_platform)}")
print(f"Product: {len(mapping_product)}")
print(f"Role: {len(mapping_role)}")
print(f"Language: {len(mapping_language)}")
print(f"Tools: {len(mapping_tools)}")

# === Load Maxlen ===
# Auto calculate maxlen
maxlen_platform = len(mapping_platform)
maxlen_product  = len(mapping_product)
maxlen_role     = len(mapping_role)
maxlen_language = len(mapping_language)
maxlen_tools    = len(mapping_tools)

# Insert to dictionary
maxlen_dict = {
    "platform": maxlen_platform,
    "product": maxlen_product,
    "role": maxlen_role,
    "language": maxlen_language,
    "tools": maxlen_tools
}

# === Helper function: encode and pad ===
def encode_and_pad(list_values, mapping, maxlen):
    #Debug
    sequenceTesting = []
    for val in list_values:
        mapped = mapping.get(val, 0)
        if mapped == 0:
            print(f"⚠️ Mapping not found for: '{val}'")
        sequenceTesting.append(mapped)
    paddedTesting = pad_sequences([sequenceTesting], maxlen=maxlen, padding="post", truncating="post")
    
    print(f"\n🔍 [{val}]")
    print(f"Input list       : {list_values}")
    print(f"Encoded indices  : {sequenceTesting}")
    print(f"Padded final     : {paddedTesting}")

    # Convert list string → list index
    sequence = [mapping.get(val, 0) for val in list_values]
    # Pad to maxlen
    padded = pad_sequences([sequence], maxlen=maxlen, padding="post", truncating="post")
    
    return padded[0]  # ambil array 1 dimensi

# === Main function: predict match ===
def predict_match(project_features_dict, talent_features_dict):
    model = get_model()  # lazy load

    print("Predict Match – START")

    # Debug: print features
    print(f"\n📦 Project Features: {project_features_dict}")
    print(f"👤 Talent Features : {talent_features_dict}")

    # Encode + pad for each feature
    # Project
    X_proj_platform = encode_and_pad(project_features_dict["platform"], mapping_platform, maxlen_dict["platform"])
    X_proj_product  = encode_and_pad(project_features_dict["product"], mapping_product, maxlen_dict["product"])
    X_proj_role     = encode_and_pad(project_features_dict["role"], mapping_role, maxlen_dict["role"])
    X_proj_language = encode_and_pad(project_features_dict["language"], mapping_language, maxlen_dict["language"])
    X_proj_tools    = encode_and_pad(project_features_dict["tools"], mapping_tools, maxlen_dict["tools"])
    
    #Debug

    # Talent
    X_tal_platform = encode_and_pad(talent_features_dict["platform"], mapping_platform, maxlen_dict["platform"])
    X_tal_product  = encode_and_pad(talent_features_dict["product"], mapping_product, maxlen_dict["product"])
    X_tal_role     = encode_and_pad(talent_features_dict["role"], mapping_role, maxlen_dict["role"])
    X_tal_language = encode_and_pad(talent_features_dict["language"], mapping_language, maxlen_dict["language"])
    X_tal_tools    = encode_and_pad(talent_features_dict["tools"], mapping_tools, maxlen_dict["tools"])
    
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

    # Debug: Show input vectors
    print("\n🧾 Final Input Vectors to Model:")
    for i, arr in enumerate(input_list):
        print(f"Vector {i+1}: {arr.tolist()}")
    print(f"\n🎯 Prediction Score: {score:.8f}")

    # Predict → return float
    score = model.predict(input_list, verbose=0)[0][0]  # ambil scalar float
    return score

# === Function: rank talent for project ===
def rank_talent_for_project(project_features_dict, list_of_talent_features_dicts):

    print("\n🧪 Matching Project vs Talent:")
    print(f"Project → '{project_features_dict}'")
    print(f"Talent  → '{talent_features_dict}'")

    result = []

    model = get_model() 
    
    for talent_features_dict in list_of_talent_features_dicts:
        talent_id = talent_features_dict["talent_id"]
        print(f"\n🔎 Evaluating Talent ID: {talent_id}")
        
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

    #Debug
    print(f"🎯 Talent {talent_id} → Score: {score:.6f}")
    print(f"\n🏁 Final Sorted Ranking: {result_sorted}")
    
    return result_sorted
