import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import random
from tqdm import tqdm

# Set seed for reproducibility
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'
DATA_PATH = '../cleaned_reddit_posts.csv'
OUTPUT_DIR = './model_all_classes'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class RedditDataset(Dataset):
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
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    # Load data
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Use the cleaned text for training
    df = df[['cleaned_text', 'Subreddit']]
    df = df.dropna()
    
    # Keep only the 7 major subreddits
    major_subreddits = ['depression', 'Anxiety', 'bipolar', 'mentalhealth', 'BPD', 'schizophrenia', 'autism']
    df = df[df['Subreddit'].isin(major_subreddits)]
    
    # Convert subreddit names to integer labels
    subreddit_to_id = {subreddit: idx for idx, subreddit in enumerate(major_subreddits)}
    df['label'] = df['Subreddit'].map(subreddit_to_id)
    
    # Display class distribution
    logger.info("Class distribution:")
    for subreddit, label_id in subreddit_to_id.items():
        count = len(df[df['label'] == label_id])
        percentage = count / len(df) * 100
        logger.info(f"{subreddit} (ID: {label_id}): {count} samples ({percentage:.2f}%)")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed_val, stratify=df['label'])
    
    return train_df, val_df, subreddit_to_id

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs):
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            model.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            train_steps += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        val_preds = []
        val_true = []
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                val_preds.extend(preds)
                val_true.extend(labels)
        
        avg_val_loss = val_loss / val_steps
        accuracy = accuracy_score(val_true, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='weighted')
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"Saving model to {OUTPUT_DIR}")
            model.save_pretrained(OUTPUT_DIR)
            
    return model

def main():
    logger.info(f"Using device: {DEVICE}")
    
    # Load data
    train_df, val_df, subreddit_to_id = load_data()
    
    # Save label mapping for inference
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.txt'), 'w') as f:
        for subreddit, idx in subreddit_to_id.items():
            f.write(f"{subreddit}\t{idx}\n")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    logger.info("Creating datasets")
    train_dataset = RedditDataset(
        texts=train_df['cleaned_text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = RedditDataset(
        texts=val_df['cleaned_text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH_SIZE
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE
    )
    
    # Initialize model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(subreddit_to_id),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train the model
    logger.info("Starting training")
    model = train(model, train_dataloader, val_dataloader, optimizer, scheduler, DEVICE, EPOCHS)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()