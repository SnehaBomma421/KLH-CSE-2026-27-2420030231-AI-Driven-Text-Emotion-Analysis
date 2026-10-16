import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from preprocess import normalize_text

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
MODEL_FOLDER = "emotion_model"
NUM_CLASSES = 28

# ------------------------------------------------------------
# PYTORCH DATASET
# ------------------------------------------------------------
class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

# ------------------------------------------------------------
# CUSTOM MULTI-LABEL TRAINER WITH WEIGHTED LOSS
# ------------------------------------------------------------
class MultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_weights is not None:
            self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        else:
            self.pos_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # Forward pass (remove labels from inputs)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        
        if self.pos_weights is not None:
            device = logits.device
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights.to(device))
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            
        loss = loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------------------
# DATA LOAD & PREPROCESSING HELPERS
# ------------------------------------------------------------
def load_and_preprocess_data(csv_path, max_samples=None, augment=False):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
        
    if augment:
        print("Applying contextual data augmentation for idioms...")
        aug_df = pd.DataFrame([
            {"text": "you are good for nothing", "labels": "[10]", "id": "aug01"},
            {"text": "he is good for nothing", "labels": "[10]", "id": "aug02"},
            {"text": "this tool is good for nothing", "labels": "[10]", "id": "aug03"},
            {"text": "you are killing it on stage!", "labels": "[13, 17]", "id": "aug04"},
            {"text": "they are killing it in the game", "labels": "[13, 17]", "id": "aug05"},
            {"text": "I am dying of laughter, that was so funny", "labels": "[1]", "id": "aug06"},
            {"text": "I am dying, please call an ambulance", "labels": "[14, 25]", "id": "aug07"},
            {"text": "great job on the presentation", "labels": "[0, 15]", "id": "aug08"},
            {"text": "great, another problem to fix", "labels": "[3, 9]", "id": "aug09"},
            {"text": "oh great, another delay", "labels": "[3, 9]", "id": "aug10"},
        ])
        df = pd.concat([df, aug_df], ignore_index=True)
        
    print(f"Preprocessing {len(df)} texts using normalization pipeline...")
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
        
    return texts, np.array(labels)

def main():
    print("\n============================================================")
    print("EMOTION DETECTION MODEL TRAINING PIPELINE")
    print("============================================================\n")

    # Detect CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device.upper()}")
    
    # Determine execution scale
    if device == "cpu":
        print("\n" + "!"*60)
        print("WARNING: CUDA/GPU is not available.")
        print("Training RoBERTa on CPU is extremely slow.")
        print("Switching to QUICK DRY-RUN mode (200 train, 50 validation samples).")
        print("For full model training, run this script on a GPU or Google Colab.")
        print("!"*60 + "\n")
        
        train_samples = 200
        val_samples = 50
        epochs = 3
        batch_size = 8
    else:
        print("GPU available! Running full fine-tuning pipeline.\n")
        train_samples = None
        val_samples = None
        epochs = 3
        batch_size = 16

    # Load and clean datasets
    train_texts, train_labels = load_and_preprocess_data("go_emotions_train.csv", max_samples=train_samples, augment=True)
    val_texts, val_labels = load_and_preprocess_data("go_emotions_validation.csv", max_samples=val_samples)

    # Calculate class positive weights for loss balancing
    # pos_weight_c = (N_neg) / (P_pos + smoothing)
    # Using smoothing of +10 to stabilize weights of extremely rare classes.
    pos_counts = np.sum(train_labels, axis=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weights = neg_counts / (pos_counts + 10)
    pos_weights = np.clip(pos_weights, 1.0, 15.0)
    print(f"\nCalculated class weights for imbalance handling (first 5 classes shown): {pos_weights[:5]}\n")

    # Load tokenizer and model
    print(f"Loading pretrained tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
    val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=128)

    train_dataset = GoEmotionsDataset(train_encodings, train_labels)
    val_dataset = GoEmotionsDataset(val_encodings, val_labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100 if device == "cuda" else 10,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none" # Disable external logging APIs
    )

    # Instantiate custom Trainer
    print("Initializing MultiLabelTrainer...")
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        pos_weights=pos_weights
    )

    # Run training
    print("\nStarting model training...")
    trainer.train()
    print("Training finished!")

    # Save fine-tuned model and tokenizer
    print(f"\nSaving fine-tuned model to '{MODEL_FOLDER}'...")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    trainer.model.save_pretrained(MODEL_FOLDER)
    tokenizer.save_pretrained(MODEL_FOLDER)
    print("Model and tokenizer saved successfully.")

    # ------------------------------------------------------------
    # PER-CLASS THRESHOLD TUNING ON VALIDATION SET
    # ------------------------------------------------------------
    print("\nStarting validation threshold tuning for GoEmotions classes...")
    model.to(device)
    model.eval()
    
    val_probs = []
    # Predict in batches
    eval_batch_size = 16
    for i in range(0, len(val_texts), eval_batch_size):
        batch_texts = val_texts[i:i+eval_batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs.append(probs)
            
    val_probs = np.concatenate(val_probs, axis=0)

    # Grid search for threshold maximizing F1 per class
    tuned_thresholds = []
    id2label = model.config.id2label
    
    print("\nTuned per-class validation F1 thresholds:")
    print("-" * 55)
    print(f"{'Class ID':<8} | {'Emotion':<15} | {'Optimal Threshold':<18} | {'F1-Score':<8}")
    print("-" * 55)
    
    for c in range(NUM_CLASSES):
        label_name = id2label.get(c, id2label.get(str(c), f"Class_{c}"))
        best_thresh = 0.3 # default fallback
        best_f1 = -1.0
        
        for t in np.arange(0.05, 0.95, 0.01):
            preds = (val_probs[:, c] >= t).astype(int)
            true = val_labels[:, c]
            
            tp = np.sum((preds == 1) & (true == 1))
            fp = np.sum((preds == 1) & (true == 0))
            fn = np.sum((preds == 0) & (true == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                
        # If the class was never successfully predicted (F1 = 0 on small validation), default to 0.3
        if best_f1 <= 0.0:
            best_thresh = 0.3
            
        tuned_thresholds.append(float(best_thresh))
        print(f"{c:<8} | {label_name:<15} | {best_thresh:<18.2f} | {max(0.0, best_f1):<8.4f}")

    print("-" * 55)

    # Save thresholds to JSON
    threshold_path = os.path.join(MODEL_FOLDER, "thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(tuned_thresholds, f, indent=4)
        
    print(f"\nTuned per-class thresholds saved successfully to '{threshold_path}'.")
    print("\n============================================================")
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("============================================================\n")

if __name__ == "__main__":
    main()