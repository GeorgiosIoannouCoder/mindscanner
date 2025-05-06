import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
import logging
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
MAX_LEN = 256
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORMED_DATA_PATH = '../transformed_dataset_full.csv'
CLEANED_DATA_PATH = '../cleaned_reddit_posts.csv'

# Model paths
MODEL_PATHS = {
    'all_classes': './model_all_classes',
    'cleaned_six': './model_cleaned_six',
    'transformed_six': './model_transformed_six',
    'cleaned_six_balanced': './model_cleaned_six_balanced',
    'hybrid_six': './model_hybrid_six'
}

class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from given path"""
    # Load label mapping
    label_mapping_path = os.path.join(model_path, 'label_mapping.txt')
    label_to_id = {}
    id_to_label = {}
    
    with open(label_mapping_path, 'r') as f:
        for line in f:
            subreddit, idx = line.strip().split('\t')
            label_to_id[subreddit] = int(idx)
            id_to_label[int(idx)] = subreddit
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer, label_to_id, id_to_label

def prepare_test_data(label_to_id, data_type='transformed'):
    """Prepare test data for evaluation"""
    if data_type == 'transformed':
        df = pd.read_csv(TRANSFORMED_DATA_PATH)
        text_column = 'transformed_text'
        label_column = 'subreddit'
    else:
        df = pd.read_csv(CLEANED_DATA_PATH)
        text_column = 'cleaned_text'
        label_column = 'Subreddit'
    
    # Filter for the classes in the model
    subreddits = list(label_to_id.keys())
    df = df[df[label_column].isin(subreddits)].dropna()
    
    # Map labels to ids
    df['label_id'] = df[label_column].map(label_to_id)
    
    # Sample a balanced test set
    test_dfs = []
    sample_size = 1000  # Sample size per class, similar to the reference evaluation
    
    for label in subreddits:
        class_df = df[df[label_column] == label]
        if len(class_df) > sample_size:
            class_df = class_df.sample(sample_size, random_state=42)
        test_dfs.append(class_df)
    
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return test_df[text_column].tolist(), test_df['label_id'].tolist(), subreddits

def evaluate_model(model, test_dataloader, id_to_label):
    """Evaluate model on test data"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Create one-hot encoded labels for ROC AUC calculation
    n_classes = len(id_to_label)
    y_true = np.zeros((len(all_labels), n_classes))
    for i, label in enumerate(all_labels):
        y_true[i, label] = 1
    
    try:
        roc_auc_micro = roc_auc_score(y_true, all_probs, average='micro')
        roc_auc_macro = roc_auc_score(y_true, all_probs, average='macro')
        
        ap_micro = average_precision_score(y_true, all_probs, average='micro')
        ap_macro = average_precision_score(y_true, all_probs, average='macro')
    except Exception as e:
        logger.error(f"Error calculating ROC AUC: {e}")
        roc_auc_micro = roc_auc_macro = ap_micro = ap_macro = 0
    
    # Format results
    results = {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_auc_micro': roc_auc_micro,
        'roc_auc_macro': roc_auc_macro,
        'average_precision_micro': ap_micro,
        'average_precision_macro': ap_macro,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'classification_report': classification_report(
            all_labels, all_preds, 
            target_names=[id_to_label[i] for i in range(len(id_to_label))], 
            output_dict=True
        )
    }
    
    return results, all_labels, all_preds, all_probs, y_true

def plot_confusion_matrix(cm, class_names, model_name, output_dir='./results'):
    """Plot confusion matrix"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curves(y_true, y_score, class_names, model_name, output_dir='./results'):
    """Plot ROC curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        plt.plot(
            fpr[i], 
            tpr[i], 
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/{model_name}_roc_curves.png')
    plt.close()

def plot_pr_curves(y_true, y_score, class_names, model_name, output_dir='./results'):
    """Plot precision-recall curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_classes = len(class_names)
    
    # Compute PR curve and average precision for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        avg_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])
    
    # Plot all PR curves
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        plt.plot(
            recall[i], 
            precision[i], 
            label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(f'{output_dir}/{model_name}_pr_curves.png')
    plt.close()

def evaluate_all_models():
    """Evaluate all models and compare results"""
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    all_results = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            logger.warning(f"Model path {model_path} does not exist. Skipping...")
            continue
            
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Load model
            model, tokenizer, label_to_id, id_to_label = load_model_and_tokenizer(model_path)
            
            # Determine test data type
            if model_name in ['transformed_six', 'hybrid_six']:
                data_type = 'transformed'
            else:
                data_type = 'cleaned'
            
            # Prepare test data
            test_texts, test_labels, class_names = prepare_test_data(label_to_id, data_type)
            
            # Create test dataset and dataloader
            test_dataset = TestDataset(
                texts=test_texts,
                labels=test_labels,
                tokenizer=tokenizer,
                max_len=MAX_LEN
            )
            
            test_dataloader = DataLoader(
                test_dataset,
                sampler=SequentialSampler(test_dataset),
                batch_size=BATCH_SIZE
            )
            
            # Evaluate model
            results, all_labels, all_preds, all_probs, y_true = evaluate_model(
                model, test_dataloader, id_to_label
            )
            
            # Store results
            all_results[model_name] = results
            
            # Create confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            plot_confusion_matrix(
                cm_normalized, 
                [id_to_label[i] for i in range(len(id_to_label))],
                model_name
            )
            
            # Plot ROC curves
            plot_roc_curves(
                y_true, 
                all_probs, 
                [id_to_label[i] for i in range(len(id_to_label))],
                model_name
            )
            
            # Plot PR curves
            plot_pr_curves(
                y_true, 
                all_probs, 
                [id_to_label[i] for i in range(len(id_to_label))],
                model_name
            )
            
            logger.info(f"Model {model_name} evaluation complete")
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
    
    # Save all results to file
    with open('./results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    comparison_metrics = [
        'accuracy', 'precision_micro', 'recall_micro', 'f1_micro',
        'precision_macro', 'recall_macro', 'f1_macro',
        'roc_auc_micro', 'roc_auc_macro',
        'average_precision_micro', 'average_precision_macro'
    ]
    
    comparison_table = {}
    for metric in comparison_metrics:
        comparison_table[metric] = {model_name: results.get(metric, 'N/A') 
                                   for model_name, results in all_results.items()}
    
    comparison_df = pd.DataFrame(comparison_table)
    comparison_df = comparison_df.transpose()
    
    # Save comparison table
    comparison_df.to_csv('./results/model_comparison.csv')
    
    # Plot comparison bar charts
    plt.figure(figsize=(12, 8))
    comparison_df.loc[['accuracy', 'f1_micro', 'f1_macro']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('./results/model_comparison_chart.png')
    
    # Print summary
    logger.info("\n=== Model Comparison Summary ===")
    for metric in ['accuracy', 'f1_micro', 'f1_macro', 'roc_auc_micro']:
        logger.info(f"\n{metric.upper()}:")
        for model_name in all_results.keys():
            logger.info(f"  {model_name}: {all_results[model_name].get(metric, 'N/A'):.4f}")

if __name__ == "__main__":
    evaluate_all_models()