import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import PeftModel
from typing import Dict, List
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import random
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Constants and mapping
SUBREDDIT_TO_LABEL = {
    "depression": 0,
    "Anxiety": 1,
    "bipolar": 2,
    "mentalhealth": 3,
    "BPD": 4,
    "schizophrenia": 5,
    "autism": 6
}

LABEL_TO_SUBREDDIT = {v: k for k, v in SUBREDDIT_TO_LABEL.items()}
# Match LSTM model's sequence length
MAX_LENGTH = 256

# Set cache directory
os.environ['TRANSFORMERS_CACHE'] = '/mnt/mistral_data/huggingface_cache'
os.environ['HF_HOME'] = '/mnt/mistral_data/huggingface_home'
os.environ['HF_DATASETS_CACHE'] = '/mnt/mistral_data/datasets_cache'
cache_dir = '/mnt/mistral_data/huggingface_cache'

def load_model(checkpoint_path):
    """Load the base model and LoRA adapter."""
    print("Loading model...")
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure quantization for inference
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        num_labels=len(SUBREDDIT_TO_LABEL),
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
    )
    
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def predict_batch(model, tokenizer, texts, labels=None, batch_size=4):
    """Make predictions for a batch of texts."""
    all_predictions = []
    all_probabilities = []
    
    # Process in batches to avoid OOM
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_probabilities

def top_k_accuracy(y_true, probabilities, k=3):
    """Calculate top-k accuracy."""
    top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_predictions[i]:
            correct += 1
    return correct / len(y_true)

def load_test_data(csv_path='cleaned_reddit_posts.csv', sample_size=5000, exclude_class=None, balanced=False):
    """Load test data from CSV file."""
    print("Loading test data...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Basic filtering
    df = df[df['cleaned_text'].notna()]
    df = df[df['cleaned_text'].str.len() > 5]
    df = df[df['cleaned_text'].str.len() <= MAX_LENGTH * 4]
    
    # Create labels
    df['label'] = df['Subreddit'].map(SUBREDDIT_TO_LABEL)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Exclude a class if specified
    if exclude_class is not None:
        if exclude_class in SUBREDDIT_TO_LABEL:
            exclude_label = SUBREDDIT_TO_LABEL[exclude_class]
            print(f"Excluding class: {exclude_class} (label {exclude_label})")
            df = df[df['label'] != exclude_label]
        else:
            print(f"Warning: Class {exclude_class} not found. Using all classes.")
    
    # Sample data based on strategy
    if balanced:
        # Balanced sampling (equal from each class)
        sampled_data = []
        for label in SUBREDDIT_TO_LABEL.values():
            # Skip excluded label
            if exclude_class in SUBREDDIT_TO_LABEL and label == SUBREDDIT_TO_LABEL[exclude_class]:
                continue
                
            class_df = df[df['label'] == label]
            if len(class_df) > 0:
                # Sample up to sample_size/len(active_classes) from each class
                active_classes = len(SUBREDDIT_TO_LABEL) - (1 if exclude_class in SUBREDDIT_TO_LABEL else 0)
                class_samples = min(len(class_df), sample_size // active_classes)
                sampled_data.append(class_df.sample(class_samples))
        
        # Combine the samples
        sampled_df = pd.concat(sampled_data)
    else:
        # Natural distribution sampling (matches LSTM approach)
        if len(df) > sample_size:
            sampled_df = df.sample(sample_size, random_state=42)
        else:
            sampled_df = df
    
    # Shuffle the data
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print class distribution
    class_counts = sampled_df['label'].value_counts().sort_index()
    print("\nClass distribution in test data:")
    for label, count in class_counts.items():
        print(f"{LABEL_TO_SUBREDDIT[label]}: {count} samples ({count/len(sampled_df)*100:.1f}%)")
    
    print(f"\nSampled {len(sampled_df)} entries for testing")
    
    return sampled_df

def evaluate_model(model, tokenizer, test_df):
    """Evaluate model on test data."""
    texts = test_df['cleaned_text'].tolist()
    true_labels = test_df['label'].tolist()
    
    print(f"Evaluating on {len(texts)} samples...")
    predictions, probabilities = predict_batch(model, tokenizer, texts)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=list(SUBREDDIT_TO_LABEL.keys()), output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate top-k accuracy
    top_1_acc = top_k_accuracy(true_labels, probabilities, k=1)
    top_2_acc = top_k_accuracy(true_labels, probabilities, k=2)
    top_3_acc = top_k_accuracy(true_labels, probabilities, k=3)
    
    # Calculate ROC AUC and Average Precision (matching the LSTM evaluation)
    # One-hot encode true labels for ROC calculations
    n_classes = len(SUBREDDIT_TO_LABEL)
    y_true_onehot = np.zeros((len(true_labels), n_classes))
    for i, label in enumerate(true_labels):
        y_true_onehot[i, label] = 1
    
    # Calculate ROC AUC scores
    roc_auc_micro = roc_auc_score(y_true_onehot, probabilities, average='micro')
    roc_auc_macro = roc_auc_score(y_true_onehot, probabilities, average='macro')
    
    # Calculate Average Precision scores
    ap_micro = average_precision_score(y_true_onehot, probabilities, average='micro')
    ap_macro = average_precision_score(y_true_onehot, probabilities, average='macro')
    
    # Print results
    print(f"\nModel Evaluation Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Top-1 Accuracy: {top_1_acc:.4f}")
    print(f"Top-2 Accuracy: {top_2_acc:.4f}")
    print(f"Top-3 Accuracy: {top_3_acc:.4f}")
    print(f"Micro-Averaged ROC AUC: {roc_auc_micro:.4f}")
    print(f"Macro-Averaged ROC AUC: {roc_auc_macro:.4f}")
    print(f"Micro-Averaged AUPRC: {ap_micro:.4f}")
    print(f"Macro-Averaged AUPRC: {ap_macro:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Subreddit':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    
    for subreddit, label in SUBREDDIT_TO_LABEL.items():
        if subreddit in report:
            metrics = report[subreddit]
            print(f"{subreddit:<15} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1-score']:.4f}     {metrics['support']}")
    print("-" * 80)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 10))
    plt.title("Confusion Matrix - Mistral 7B Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    cm_df = pd.DataFrame(
        conf_matrix,
        index=[LABEL_TO_SUBREDDIT[i] for i in range(len(SUBREDDIT_TO_LABEL))],
        columns=[LABEL_TO_SUBREDDIT[i] for i in range(len(SUBREDDIT_TO_LABEL))],
    )
    
    sns.heatmap(cm_df, annot=True, fmt="d", cbar=True)
    plt.tight_layout()
    plt.savefig('mistral_confusion_matrix.png')
    plt.figure()
    
    # Generate ROC curves (per class)
    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    average_precision = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], probabilities[:, i])
        average_precision[i] = average_precision_score(y_true_onehot[:, i], probabilities[:, i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i], 
            label=f"{LABEL_TO_SUBREDDIT[i]} (AUC = {roc_auc[i]:.2f})"
        )
    
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - One-vs-Rest (Mistral 7B)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig('mistral_roc_curve.png')
    plt.figure()
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(
            recall[i],
            precision[i],
            label=f"{LABEL_TO_SUBREDDIT[i]} (AP = {average_precision[i]:.2f})",
        )
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - One-vs-Rest (Mistral 7B)")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig('mistral_pr_curve.png')
    plt.figure()
    
    # Print some examples with top-3 predictions
    print("\nRandom Example Predictions (with top-3):")
    print("=" * 100)
    
    num_examples = min(10, len(texts))
    indices = random.sample(range(len(texts)), num_examples)
    
    for idx in indices:
        text = texts[idx]
        true_label = true_labels[idx]
        pred_label = predictions[idx]
        
        # Get top 3 predictions and probabilities
        top_3_indices = np.argsort(probabilities[idx])[-3:][::-1]
        top_3_probs = probabilities[idx][top_3_indices] * 100
        
        # Truncate text if too long
        if len(text) > 100:
            text = text[:97] + "..."
        
        print(f"Text: {text}")
        print(f"True: {LABEL_TO_SUBREDDIT[true_label]}")
        print("Top 3 predictions:")
        for i, (idx, prob) in enumerate(zip(top_3_indices, top_3_probs)):
            print(f"  {i+1}. {LABEL_TO_SUBREDDIT[idx]} ({prob:.2f}%)")
        print("-" * 100)
    
    return accuracy, report, conf_matrix, roc_auc_micro, ap_micro

def main():
    parser = argparse.ArgumentParser(description="Evaluate mental health classification model")
    parser.add_argument("--checkpoint", type=str, default="/mnt/mistral_data/mistral_mental_health/checkpoint-3000", 
                        help="Path to model checkpoint")
    parser.add_argument("--exclude", type=str, default=None, 
                        help="Exclude this class from evaluation (e.g., 'depression')")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of samples to evaluate (default: 5000)")
    parser.add_argument("--balanced", action="store_true", 
                        help="Use balanced class distribution instead of natural distribution")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results and plots")
    parser.add_argument("--csv_path", type=str, default="cleaned_reddit_posts.csv",
                        help="Path to the CSV file with test data")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nMistral Mental Health Classification Model Evaluation\n")
    print(f"Using {'balanced' if args.balanced else 'natural'} class distribution")
    
    # Load the model
    model, tokenizer = load_model(args.checkpoint)
    
    # Load test data
    test_df = load_test_data(
        csv_path=args.csv_path,
        sample_size=args.samples, 
        exclude_class=args.exclude,
        balanced=args.balanced
    )
    
    # Evaluate the model
    results = evaluate_model(model, tokenizer, test_df)
    
    print(f"\nEvaluation complete. Plots saved to current directory.")

if __name__ == "__main__":
    main() 