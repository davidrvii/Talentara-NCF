# inference.py

import tensorflow as tf
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences

# === Lazy load model ===
model = None

def get_model():
    global model
    if model is None:
        print("Loading model ...")
        initialize_mappings()
        model = tf.keras.models.load_model("talent_filtering_model.keras")
        print("Model loaded.")
    return model

def initialize_mappings():
    global mapping_platform, mapping_product, mapping_role, mapping_language, mapping_tools
    global maxlen_platform, maxlen_product, maxlen_role, maxlen_language, maxlen_tools
    global maxlen_dict

    base_path = "model_assets"

    # Load mapping
    with open(f"{base_path}/mapping_platform.json") as f: mapping_platform = json.load(f)
    with open(f"{base_path}/mapping_product.json") as f: mapping_product = json.load(f)
    with open(f"{base_path}/mapping_role.json") as f: mapping_role = json.load(f)
    with open(f"{base_path}/mapping_language.json") as f: mapping_language = json.load(f)
    with open(f"{base_path}/mapping_tools.json") as f: mapping_tools = json.load(f)

    # Load maxlen
    with open(f"{base_path}/maxlen.json") as f: maxlen_dict = json.load(f)

    maxlen_platform = maxlen_dict["platform"]
    maxlen_product  = maxlen_dict["product"]
    maxlen_role     = maxlen_dict["role"]
    maxlen_language = maxlen_dict["language"]
    maxlen_tools    = maxlen_dict["tools"]

    print("📚 Mapping sizes:")
    print(f"Platform: {len(mapping_platform)}")
    print(f"Product: {len(mapping_product)}")
    print(f"Role: {len(mapping_role)}")
    print(f"Language: {len(mapping_language)}")
    print(f"Tools: {len(mapping_tools)}")

# === Helper function: encode and pad ===
def encode_and_pad(list_values, mapping, maxlen):
    sequence = []
    unknown_index = len(mapping) + 1
    
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
        model = get_model()

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
        
        input_dict = {
            "input_proj_platform":  np.array([X_proj_platform]),
            "input_proj_product":   np.array([X_proj_product]),
            "input_proj_role":      np.array([X_proj_role]),
            "input_proj_language":  np.array([X_proj_language]),
            "input_proj_tools":     np.array([X_proj_tools]),
            "input_tal_platform":   np.array([X_tal_platform]),
            "input_tal_product":    np.array([X_tal_product]),
            "input_tal_role":       np.array([X_tal_role]),
            "input_tal_language":   np.array([X_tal_language]),
            "input_tal_tools":      np.array([X_tal_tools])
        }

        # Log each input model
        print("\n🔢 Raw Input Values Before Feeding to Model:")
        for key, val in input_dict.items():
            print(f"{key} → shape: {val.shape}, values: {val.tolist()}")

        mapping_dict = {
            "platform": mapping_platform,
            "product": mapping_product,
            "role": mapping_role,
            "language": mapping_language,
            "tools": mapping_tools
        }
        explain_match_score(project_features_dict, talent_features_dict, mapping_dict)

        # Show input vectors
        print("\n🧾 Final Input Vectors to Model:")
        for key, arr in input_dict.items():
            if isinstance(arr, np.ndarray):
                print(f"{key}: {arr.tolist()}")
            else:
                print(f"{key}: (not ndarray) → {arr}")
    
        # Predict → return float
        score = model.predict(input_dict, verbose=0)[0][0]
        print(f"\n🎯 Prediction Score: {score:.8f}")
        return score
    
    except Exception as e:
        print(f"❗ predict_match error: {e}")
        raise

# === Function: rank talent for project ===
def rank_talent_for_project(project_features_dict, list_of_talent_features_dicts):
    result = []
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
            print(f"🎯 Talent {talent_id} → Score: {score:.6f}")
        except Exception as e:
            print(f"❌ Error while predicting for talent {talent_id}: {e}")
            score = 0.0

        result.append({
            "talent_id": talent_id,
            "score": float(score)
        })
    
    result_sorted = sorted(result, key=lambda x: x["score"], reverse=True)
    print(f"\n🏁 Final Sorted Ranking: {result_sorted}")
    
    return result_sorted

def explain_match_score(project_dict, talent_dict, mapping_dict):
    explanation = {}
    for key in ["platform", "product", "role", "language", "tools"]:
        project_vals = set(project_dict[key])
        talent_vals  = set(talent_dict[key])
        matched = project_vals.intersection(talent_vals)
        missed  = project_vals - matched

        explanation[key] = {
            "matched": list(matched),
            "missed": list(missed),
            "matched_count": len(matched),
            "project_count": len(project_vals),
            "coverage": round(len(matched) / len(project_vals), 2) if project_vals else 0.0
        }

    print("\n📊 Match Explanation per Feature:")
    for key, val in explanation.items():
        print(f"\n🧩 {key.upper()}")
        print(f"✅ Matched: {val['matched']}")
        print(f"❌ Missed : {val['missed']}")
        print(f"📈 Coverage: {val['coverage']*100:.1f}% ({val['matched_count']} of {val['project_count']})")
    return explanation
