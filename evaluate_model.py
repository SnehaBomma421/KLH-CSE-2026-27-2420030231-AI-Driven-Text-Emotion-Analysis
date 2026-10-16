import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_recall_fscore_support
from preprocess import normalize_text

MODEL_FOLDER = "emotion_model"
TEST_DATA_PATH = "go_emotions_test.csv"
NUM_CLASSES = 28

# The 20 minimum test sentences requested in the prompt
PROMPT_TEST_SENTENCES = [
    "I am happy",
    "I am HAPPYYYY",
    "I am sooo happy right now",
    "I HAAAAATE this",
    "I love this soooo much",
    "you are good for nothing",
    "you are useless",
    "you are amazing",
    "thank you so much",
    "I am really disappointed",
    "I can't believe you did that",
    "OMG NOOOOO",
    "Why would you do this to me?",
    "I am scared",
    "I am nervous about tomorrow",
    "That was hilarious",
    "I am so proud of myself",
    "Great, another problem.",
    "I guess that's fine.",
    "I don't know how I feel about this."
]

# 10 custom sentences covering specific categories
CUSTOM_TEST_SENTENCES = [
    "Oh great, my flight is delayed again. Fantastic.", # Sarcasm / Annoyance
    "Holy cow! I just won the lottery!!!", # Surprise / Joy / Excitement
    "This is so disgusting, please take it away.", # Disgust
    "I'm extremely anxious and my hands are shaking...", # Nervousness / Fear
    "I am so sorry for what I did, I feel terrible.", # Remorse / Sadness
    "The meeting is at 2 PM in Room 4B.", # Neutral
    "I admire your dedication and hard work, you're an inspiration.", # Admiration / Approval
    "I feel so safe and cared for when I'm with you.", # Caring / Love
    "Why does she always act like she knows everything? It's so frustrating!", # Annoyance / Disapproval
    "Wait, did he really say that? I'm so confused." # Confusion / Surprise
]

def load_data(filepath):
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath)
    texts = df['text'].apply(normalize_text).tolist()
    
    labels = []
    for labels_str in df['labels']:
        # Robustly parse labels like "[8 20]" or "[8, 20]"
        cleaned = labels_str.strip().replace('[', '').replace(']', '').replace(',', ' ')
        label_list = [int(x) for x in cleaned.split()]
        multi_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
        for l in label_list:
            multi_hot[l] = 1.0
        labels.append(multi_hot)
        
    return df['text'].tolist(), texts, np.array(labels)

def main():
    print("\n============================================================")
    print("EMOTION DETECTION SYSTEM EVALUATION REPORT")
    print("============================================================\n")

    if not os.path.exists(MODEL_FOLDER):
        print(f"ERROR: Model folder '{MODEL_FOLDER}' not found. Please train the model first.")
        return

    # Load Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device.upper()}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER).to(device)
    model.eval()

    # Load thresholds
    threshold_path = os.path.join(MODEL_FOLDER, "thresholds.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            thresholds = json.load(f)
        print("Tuned thresholds loaded successfully.")
    else:
        thresholds = [0.3] * NUM_CLASSES
        print("WARNING: thresholds.json not found. Using default 0.3 threshold.")

    id2label = model.config.id2label

    # Load test dataset
    orig_texts, normalized_texts, true_labels = load_data(TEST_DATA_PATH)

    # Run predictions on test set in batches
    print("Running inference on test dataset...")
    probs_list = []
    batch_size = 32
    for i in range(0, len(normalized_texts), batch_size):
        batch_texts = normalized_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            probs_list.append(probs)
            
    probs = np.concatenate(probs_list, axis=0)

    # Evaluate under two threshold schemes: Standard 0.5 vs. Tuned thresholds
    for scheme_name, thresh_arr in [("Standard 0.5 Threshold", [0.5] * NUM_CLASSES), ("Tuned Thresholds", thresholds)]:
        preds = np.zeros_like(probs)
        for c in range(NUM_CLASSES):
            preds[:, c] = (probs[:, c] >= thresh_arr[c]).astype(float)
            
        micro_f1 = f1_score(true_labels, preds, average='micro', zero_division=0)
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
        
        print("\n" + "="*50)
        print(f"Metrics using {scheme_name}")
        print("="*50)
        print(f"Micro F1-Score:    {micro_f1:.4f}")
        print(f"Macro F1-Score:    {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        
        if scheme_name == "Tuned Thresholds":
            # Print per-class metrics only for the final Tuned Thresholds
            print("\nPer-Class Metrics (with Tuned Thresholds):")
            print("-" * 75)
            print(f"{'Class ID':<8} | {'Emotion':<15} | {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<8}")
            print("-" * 75)
            for c in range(NUM_CLASSES):
                p, r, f, _ = precision_recall_fscore_support(true_labels[:, c], preds[:, c], average='binary', zero_division=0)
                label_name = id2label.get(c, id2label.get(str(c), f"Class_{c}"))
                print(f"{c:<8} | {label_name:<15} | {thresh_arr[c]:<10.2f} | {p:<10.4f} | {r:<10.4f} | {f:<8.4f}")
            print("-" * 75)

    # ------------------------------------------------------------
    # INFERENCE ON SAMPLE TEST SENTENCES
    # ------------------------------------------------------------
    print("\n" + "="*75)
    print("RUNNING INFERENCE ON SPECIFIED TEST SENTENCES")
    print("="*75 + "\n")

    test_sentences = PROMPT_TEST_SENTENCES + CUSTOM_TEST_SENTENCES

    for i, raw_sent in enumerate(test_sentences):
        norm_sent = normalize_text(raw_sent)
        inputs = tokenizer(norm_sent, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prob_tensor = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        # Gather positive predictions above tuned thresholds
        detected = []
        for c in range(NUM_CLASSES):
            if prob_tensor[c] >= thresholds[c]:
                label_name = id2label.get(c, id2label.get(str(c), f"Class_{c}"))
                detected.append((label_name, prob_tensor[c]))

        # Fallback to highest if nothing detected
        if not detected:
            best_idx = np.argmax(prob_tensor)
            label_name = id2label.get(best_idx, id2label.get(str(best_idx), f"Class_{best_idx}"))
            detected.append((label_name, prob_tensor[best_idx]))

        # Sort by confidence
        detected.sort(key=lambda x: x[1], reverse=True)

        print(f"Test Sentence #{i+1}")
        print(f"Original:   \"{raw_sent}\"")
        print(f"Normalized: \"{norm_sent}\"")
        print("Predictions:")
        for rank, (emotion, score) in enumerate(detected):
            prefix = "Primary Emotion:       " if rank == 0 else "Secondary Emotion:     "
            print(f"  - {prefix}{emotion:<15} | Confidence: {score * 100:.2f}% (Threshold: {thresholds[list(id2label.values()).index(emotion)]:.2f})")
        print("-" * 75)

if __name__ == "__main__":
    main()
