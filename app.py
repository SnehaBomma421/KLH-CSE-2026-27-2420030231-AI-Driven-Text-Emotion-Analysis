import os
import json
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from preprocess import normalize_text, CONTRACTIONS

app = Flask(__name__)

def detect_signals(original_text, normalized_text):
    signals = []
    
    # 1. Repeated characters
    if re.search(r'(?i)(.)\1{2,}', original_text):
        signals.append("Repeated Characters")
        
    # 2. Capitalization (Shouting / Intensity)
    words = re.findall(r'\b\w+\b', original_text)
    if any(w.isupper() and len(w) >= 2 for w in words):
        signals.append("Capitalization")
        
    # 3. Strong Punctuation
    if re.search(r'[!?]{2,}', original_text):
        signals.append("Strong Punctuation")
        
    # 4. Contractions
    original_words_lower = set(w.lower() for w in re.findall(r"\b[\w']+\b", original_text))
    if any(c in original_words_lower for c in CONTRACTIONS.keys()):
        signals.append("Contractions")
        
    # 5. Intensifiers
    intensifiers = {"so", "very", "extremely", "really", "incredibly", "totally", "absolutely", "too", "sooo", "soooo"}
    if any(i in original_words_lower for i in intensifiers):
        signals.append("Intensifiers")
        
    return signals

# ============================================================
# LOAD MODEL & CONFIGURATION
# ============================================================

MODEL_FOLDER = "./emotion_model"
NUM_CLASSES = 28

print("\nLoading emotion detection model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER)
model.eval()
print("Model loaded successfully!")

# Retrieve the model's own id2label mapping
id2label = model.config.id2label
emotion_names = [id2label.get(c, id2label.get(str(c), f"Class_{c}")) for c in range(NUM_CLASSES)]
print(f"Mapped {len(emotion_names)} classes from model config. First 5 classes: {emotion_names[:5]}")

# Load per-class optimal thresholds
threshold_path = os.path.join(MODEL_FOLDER, "thresholds.json")
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)
    print("Tuned class thresholds loaded successfully.")
else:
    thresholds = [0.3] * NUM_CLASSES
    print("WARNING: thresholds.json not found. Falling back to default 0.3 threshold.")

# ============================================================
# ROUTING
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received."}), 400

        original_text = data.get("text", "").strip()
        if not original_text:
            return jsonify({"error": "Please enter some text."}), 400

        # Normalize text using preprocess pipeline
        normalized_text = normalize_text(original_text)

        # Run tokenization on normalized text
        inputs = tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        # Model Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Compute probabilities using sigmoid (multi-label)
        probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        # Map predictions using per-class thresholds
        detected_emotions = []
        for index, prob in enumerate(probabilities):
            score = float(prob)
            thresh = thresholds[index]
            if score >= thresh:
                detected_emotions.append({
                    "emotion": emotion_names[index],
                    "confidence": round(score * 100, 2)
                })

        # Fallback to single highest-confidence emotion if no class exceeded threshold
        if not detected_emotions:
            best_index = int(torch.argmax(torch.tensor(probabilities)).item())
            best_score = float(probabilities[best_index])
            detected_emotions.append({
                "emotion": emotion_names[best_index],
                "confidence": round(best_score * 100, 2)
            })

        # Sort detected emotions by confidence descending
        detected_emotions.sort(key=lambda x: x["confidence"], reverse=True)

        primary_emotion = detected_emotions[0]["emotion"]
        primary_confidence = detected_emotions[0]["confidence"]

        # Formulate API response payload
        return jsonify({
            "emotion": primary_emotion,
            "confidence": primary_confidence,
            "emotions": detected_emotions,
            "original_text": original_text,
            "normalized_text": normalized_text,
            "signals": detect_signals(original_text, normalized_text)
        })

    except Exception as e:
        # Gracefully handle server exceptions internally
        print("Prediction server error:", str(e))
        return jsonify({"error": "An error occurred during text analysis. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)