'''
Enrico M. Aldorasi - 2131576
MNLP Homework 2
Adversarial Natural Language Inference 
Main script
'''

# Import necessary libraries and modules
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from datasets import load_dataset
from torch import nn
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define constants
MAX_LENGTH = 128
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
EPOCHS = 20
DROPOUT_RATE = 0.1

# Define model options
MODEL_OPTIONS = {
    'roberta': 'roberta-base',
    'deberta': 'microsoft/deberta-v3-large'
}

# Dataset class for Natural Language Inference (NLI) task
class NLIDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            item['premise'],
            item['hypothesis'],
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label = self.label_map.get(item['label'])
        if label is None:
            raise ValueError(f"Label '{item['label']}' not found in label map for item at index {idx}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Plain model class without additional layers for NLI task
class PlainNLIModel(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(PlainNLIModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits
    
# Enhanced model class for NLI task
class EnhancedNLIModel(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(EnhancedNLIModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Function to plot training progress
def plot_training_progress(train_losses, val_metrics):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(2, 1, 2)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(epochs, [m[metric] for m in val_metrics], 'o-', label=f'Validation {metric.capitalize()}')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Training function
def train(model, train_loader, dev_loader, optimizer, scheduler, device, num_epochs):
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    scaler = amp.GradScaler()
    
    train_losses = []
    val_metrics = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, dev_loader, device)
        
        train_losses.append(train_loss)
        val_metrics.append({
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    plot_training_progress(train_losses, val_metrics)
    return best_val_f1

# Training epoch function
def train_epoch(model, train_loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        with amp.autocast():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * ACCUMULATION_STEPS

    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    plot_confusion_matrix(true_labels, predictions)
    
    return accuracy, precision, recall, f1

# Main function to handle training and testing
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and model
    if args.mode == 'train':
        MODEL_NAME = MODEL_OPTIONS[args.model]
    else:  # For test mode, we'll load the saved model, so we don't need to specify the model type
        MODEL_NAME = MODEL_OPTIONS['roberta']  # Default to RoBERTa for tokenizer initialization
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # Always load the original dataset for validation and test sets
    original_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
    
    # Load and prepare datasets
    if args.data == 'augmented':
        if not args.augmented_data:
            raise ValueError("Path to augmented data JSONL file must be provided when using augmented data.")
        with open(args.augmented_data, 'r') as f:
            augmented_data = [json.loads(line) for line in f]
        train_dataset = NLIDataset(augmented_data, tokenizer)
    else:
        train_dataset = NLIDataset(original_dataset['train'], tokenizer)
    
    dev_dataset = NLIDataset(original_dataset['validation'], tokenizer)
    
    if args.data == 'adversarial':
        adversarial_dataset = load_dataset("iperbole/adversarial_fever_nli")
        test_dataset = NLIDataset(adversarial_dataset['test'], tokenizer)
    else:
        test_dataset = NLIDataset(original_dataset['test'], tokenizer)

    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    if args.mode == 'train':
        pretrained_model = AutoModel.from_pretrained(MODEL_NAME, config=config)
        if args.architecture == 'enhanced':
            model = EnhancedNLIModel(pretrained_model, num_labels=3)
        else:
            model = PlainNLIModel(pretrained_model, num_labels=3)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        # Training setup
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        num_training_steps = EPOCHS * len(train_loader) // ACCUMULATION_STEPS
        warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

        best_val_f1 = train(model, train_loader, dev_loader, optimizer, scheduler, device, EPOCHS)
        print(f"Best Validation F1: {best_val_f1:.4f}")

        # Save the entire model
        torch.save(model.module if isinstance(model, nn.DataParallel) else model, 'best_model.pth')
        print("Model saved successfully.")

    elif args.mode == 'test':
        # Load the model architecture first
        model = torch.load('best_model.pth', map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.to(device)
        model.eval()

        # Test
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
        print(f"Test Results on {'Original' if args.data == 'original' else 'Adversarial' if args.data == 'adversarial' else 'Augmented'} Data:")
        print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

# Entry point for the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--data', choices=['original', 'adversarial', 'augmented'], default='original')
    parser.add_argument('--augmented_data', type=str, help='Path to augmented data JSONL file')
    parser.add_argument('--model', choices=['roberta', 'deberta'], default='roberta', help='Choose the model to use (only for training)')
    parser.add_argument('--architecture', choices=['plain', 'enhanced'], default='plain', help='Choose the model architecture (plain or enhanced)')
    args = parser.parse_args()
    
    if args.mode == 'train' and args.data == 'augmented' and not args.augmented_data:
        parser.error("--augmented_data is required when using augmented data for training")
    
    if args.mode == 'test' and args.model:
        print("Warning: --model argument is ignored in test mode. The saved model will be used.")
    
    main(args)