import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from data_loader import MetaphorDataset
from model import MetaphorFusionModel

def test(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Dataset
    print(f"Loading test data from {args.data_path}...")
    test_dataset = MetaphorDataset(args.data_path, model_name=args.model_name, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    print("Initializing model...")
    model = MetaphorFusionModel(model_name=args.model_name, hidden_dim=args.hidden_dim, dropout_prob=0.0) # Dropout 0 for eval
    
    # Load Saved Weights
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_index = batch['target_index'].to(device)
            pos_id = batch['pos_id'].to(device)
            dep_id = batch['dep_id'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, target_index, pos_id, dep_id)
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    print("\n" + "="*30)
    print("       TEST RESULTS       ")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 (Macro):{f1:.4f}")
    print("="*30 + "\n")
    
    # Save Predictions
    print(f"Saving predictions to {args.output_path}...")
    

    
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds
    })
    
    results_df.to_csv(args.output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Metaphor Detection Model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data (CSV/JSON)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model state_dict')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions CSV')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Transformer model name')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for MLP')
    
    args = parser.parse_args()
    test(args)
