import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForMaskedLM, AdamW
from datasets import load_from_disk
from torch.cuda.amp import autocast, GradScaler

def load_data(dir_path, max_samples=None):
    """Loads a dataset from disk and optionally limits the number of samples."""
    print(f"Loading data from directory: {dir_path}")
    try:
        dataset = load_from_disk(dir_path)
        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))
        print(f"Dataset info: {dataset}")
        return dataset
    except Exception as err:
        print(f"Error loading data from {dir_path}: {err}")
        return None

def truncate_sequences(examples, max_length):
    """Truncates the sequences in the dataset to a specified maximum length."""
    return {
        'input_ids': examples['input_ids'][:max_length],
        'attention_mask': examples['attention_mask'][:max_length],
        'labels': examples['labels'][:max_length]
    }

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, accumulation_steps):
    """Trains the model using the provided data loaders, optimizer, and other parameters."""
    scaler = GradScaler()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            labels[labels == 0] = -100
            
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                labels[labels == 0] = -100
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_DIR = '/content/Project1/data/processed'
    
    MAX_SAMPLES = 50000  
    MAX_LENGTH = 256  
    BATCH_SIZE = 32  
    ACCUMULATION_STEPS = 4 
    NUM_EPOCHS = 2
    
    train_path = os.path.join(DATA_DIR, 'tokenized_data_train')
    val_path = os.path.join(DATA_DIR, 'tokenized_data_val')
    
    tokenized_train = load_data(train_path, MAX_SAMPLES)
    tokenized_val = load_data(val_path, MAX_SAMPLES // 20)
    
    if tokenized_train is None or tokenized_val is None:
        print("Failed to load data. Exiting.")
        exit(1)
    
    tokenized_train = tokenized_train.map(lambda x: truncate_sequences(x, MAX_LENGTH))
    tokenized_val = tokenized_val.map(lambda x: truncate_sequences(x, MAX_LENGTH))
    
    # Converting the datasets to TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(tokenized_train['input_ids']),
        torch.tensor(tokenized_train['attention_mask']),
        torch.tensor(tokenized_train['labels'])
    )
    val_dataset = TensorDataset(
        torch.tensor(tokenized_val['input_ids']),
        torch.tensor(tokenized_val['attention_mask']),
        torch.tensor(tokenized_val['labels'])
    )
    
    # making the datasets iterable
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, NUM_EPOCHS, device, ACCUMULATION_STEPS
    )
    
    model_save_path = os.path.join(DATA_DIR, 'trained_model')
    model.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(DATA_DIR, 'training_results.pth'))
    
    print("Model training completed successfully.")
