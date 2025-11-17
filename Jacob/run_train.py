import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv('/Users/jacoblindsey/Documents/Machine Learning Class 2025/Team Project/train.csv')

# Clean labels
df["label"] = (
    df["label"].astype(str)
    .str.strip()
    .str.upper()
    .map({"TRUE": 1, "FALSE": 0})
)
df = df.dropna(subset=["label", "text", "metaphorID"])

# Split into train/validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# ---------------------------
# 2. TF-IDF VECTORIZATION
# ---------------------------
vectorizer = TfidfVectorizer(max_features=5000)

# Include target word in input
train_df['target_text'] = train_df['text'] + ' ' + train_df['metaphorID'].astype(str)
val_df['target_text'] = val_df['text'] + ' ' + val_df['metaphorID'].astype(str)

X_train_tfidf = vectorizer.fit_transform(train_df["target_text"]).toarray()
X_val_tfidf = vectorizer.transform(val_df["target_text"]).toarray()

# ---------------------------
# 3. ONE-HOT ENCODE METAPHOR ID
# ---------------------------
num_words = df['metaphorID'].nunique()  # e.g., 7
def one_hot_id(mid):
    vec = np.zeros(num_words)
    vec[mid] = 1
    return vec

X_train_id = np.array([one_hot_id(x) for x in train_df['metaphorID']])
X_val_id = np.array([one_hot_id(x) for x in val_df['metaphorID']])

# Combine TF-IDF features + one-hot target word ID
X_train = np.hstack([X_train_tfidf, X_train_id])
X_val = np.hstack([X_val_tfidf, X_val_id])

y_train = train_df["label"].values
y_val = val_df["label"].values

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ---------------------------
# 4. SIMPLE NEURAL NETWORK
# ---------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # binary classification
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = SimpleNN(input_size=X_train.shape[1])

# ---------------------------
# 5. TRAINING SETUP
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# ---------------------------
# 6. TRAIN LOOP WITH VALIDATION
# ---------------------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

