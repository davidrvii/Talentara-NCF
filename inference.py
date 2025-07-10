# inference.py

import tensorflow as tf
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from mapping_loader import load_mapping_from_db
from math import log1p  # log(1 + x)

# === Lazy load model ===
model = None

def get_model():
    global model
    if model is None:
        print("Loading model ...")
        model = tf.keras.models.load_model("best_ncf_with_embedding_improved.keras")
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
with open("maxlen.json", "r") as f:
    maxlen_dict = json.load(f)

maxlen_platform = maxlen_dict["platform"]
maxlen_product  = maxlen_dict["product"]
maxlen_role     = maxlen_dict["role"]
maxlen_language = maxlen_dict["language"]
maxlen_tools    = maxlen_dict["tools"]

# === Helper function: encode and pad ===
def encode_and_pad(list_values, mapping, maxlen):
    sequence = []
    unknown_index = len(mapping)
    # Convert list string → list index
    for val in list_values:
        if val in mapping:
            mapped = mapping[val]
        else:
            print(f"⚠️ Mapping not found for: '{val}' → using OOV index {unknown_index}")
            mapped = unknown_index
        sequence.append(mapped)
    # Pad to maxlen
    padded = pad_sequences([sequence], maxlen=maxlen, padding="post", truncating="post")
    
    print(f"\n🔍 [{val}]")
    print(f"Input list       : {list_values}")
    print(f"Encoded indices  : {sequence}")
    print(f"Padded final     : {padded}")
    
    return padded[0] 

# === Main function: predict match ===
def predict_match(project_features_dict, talent_features_dict):
    try:
        model = get_model()  # lazy load

        print("Predict Match – START")
        print(f"\n📦 Project Features: {project_features_dict}")
        print(f"👤 Talent Features : {talent_features_dict}")

        # Encode + pad for each feature Project
        X_proj_platform = encode_and_pad(project_features_dict["platform"], mapping_platform, maxlen_dict["platform"])
        X_proj_product  = encode_and_pad(project_features_dict["product"], mapping_product, maxlen_dict["product"])
        X_proj_role     = encode_and_pad(project_features_dict["role"], mapping_role, maxlen_dict["role"])
        X_proj_language = encode_and_pad(project_features_dict["language"], mapping_language, maxlen_dict["language"])
        X_proj_tools    = encode_and_pad(project_features_dict["tools"], mapping_tools, maxlen_dict["tools"])

        # Encode + pad for each feature Talent
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

        # Show input vectors
        print("\n🧾 Final Input Vectors to Model:")
        for i, arr in enumerate(input_list):
            print(f"Vector {i+1}: {arr.tolist()}")

        # Predict → return float
        raw_score = model.predict(input_list, verbose=0)[0][0]

        # Adjust score by penalizing total stack size
        total_stack = sum([
            len(project_features_dict["platform"]),
            len(project_features_dict["product"]),
            len(project_features_dict["role"]),
            len(project_features_dict["language"]),
            len(project_features_dict["tools"])
        ])
        penalty = 1 / log1p(total_stack)  # log(1 + total_stack)
        adjusted_score = float(raw_score) * penalty

        print(f"\n🎯 Raw Score: {raw_score:.6f}")
        print(f"📏 Stack Count: {total_stack} → Penalty: {penalty:.4f}")
        print(f"✅ Adjusted Score: {adjusted_score:.6f}")
        return adjusted_score
    
    except Exception as e:
        print(f"❗ predict_match error: {e}")
        raise

# === Function: rank talent for project ===
def rank_talent_for_project(project_features_dict, list_of_talent_features_dicts):
    result = []
    ADJUSTED_THRESHOLD = 0.25
    model = get_model() 
    
    for talent_features_dict in list_of_talent_features_dicts:
        talent_id = talent_features_dict["talent_id"]

        print(f"\n🔎 Evaluating Talent ID: {talent_id}")
        print("\n🧪 Matching Project vs Talent:")
        print(f"Project → '{project_features_dict}'")
        print(f"Talent  → '{talent_features_dict}'")
        
        talent_features = {
            "platform": talent_features_dict.get("platform", []),
            "product": talent_features_dict.get("product", []),
            "role": talent_features_dict.get("role", []),
            "language": talent_features_dict.get("language", []),
            "tools": talent_features_dict.get("tools", [])
        }
        
        try:
            score = predict_match(project_features_dict, talent_features)
            print(f"🎯 Talent {talent_id} → Adjusted Score: {score:.6f}")
        except Exception as e:
            print(f"❌ Error while predicting for talent {talent_id}: {e}")
            score = 0.0
        
        if score >= ADJUSTED_THRESHOLD:
            result.append({
                "talent_id": talent_id,
                "score": float(score)
            })
        else: 
            print(f"⚠️ Talent {talent_id} skor {score:.4f} < {ADJUSTED_THRESHOLD}, filtered.")
    
    result_sorted = sorted(result, key=lambda x: x["score"], reverse=True)
    print(f"\n🏁 Final Sorted Ranking: {result_sorted}")
    
    return result_sorted
