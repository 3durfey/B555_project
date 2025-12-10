import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm
import numpy as np

from data_loader import MetaphorDataset
from model import MetaphorFusionModel

from sklearn.metrics import precision_recall_fscore_support

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Dataset
    print(f"Loading data from {args.data_path}...")
    full_dataset = MetaphorDataset(args.data_path, model_name=args.model_name, max_len=args.max_len)
    
    # Split into Train/Val
    # Assuming standard 80/20 split if no separate validation file provided
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    print("Initializing model...")

    model = MetaphorFusionModel(model_name=args.model_name, hidden_dim=args.hidden_dim, dropout_prob=args.dropout)
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    
    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_index = batch['target_index'].to(device)
            pos_id = batch['pos_id'].to(device)
            dep_id = batch['dep_id'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask, target_index, pos_id, dep_id)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Calculate train metrics for this batch
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='macro', zero_division=0)
        
        # Validation Loop
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_index = batch['target_index'].to(device)
                pos_id = batch['pos_id'].to(device)
                dep_id = batch['dep_id'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask, target_index, pos_id, dep_id)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Calculate Validation Metrics
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train: Loss = {avg_train_loss:.4f}, Acc = {train_acc:.2f}%, P = {train_precision:.4f}, R = {train_recall:.4f}, F1 = {train_f1:.4f}")
        print(f"Val:   Loss = {avg_val_loss:.4f},   Acc = {val_acc:.2f}%,   P = {val_precision:.4f},   R = {val_recall:.4f},   F1 = {val_f1:.4f}")
        print("-" * 60)
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_path)
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Metaphor Detection Model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data (CSV/JSON)')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Transformer model name')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for MLP')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    args = parser.parse_args()
    train(args)
