import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.model import SimpleTextClassifier

# 1. Load Data
df = pd.read_csv("labeled_data_final.csv")
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])

# 2. Build Vocab
all_words = " ".join(df['query'].astype(str)).lower().split()
vocab = {word: i+1 for i, word in enumerate(set(all_words))}
vocab_size = len(vocab) + 1

def prepare_sequence(text):
    idxs = [vocab.get(w, 0) for w in str(text).lower().split()]
    return torch.tensor([idxs if idxs else [0]], dtype=torch.long)

# 3. Setup Model
model = SimpleTextClassifier(vocab_size, 16, 3)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Translating {len(df)} rows to math tensors (this takes a few seconds)...")

# OPTIMIZATION 1: Pre-process everything OUTSIDE the loop
# This stops Python from reading text 500,000 times
queries_data = [prepare_sequence(q) for q in df['query']]
targets_data = [torch.tensor([int(l)], dtype=torch.long) for l in df['label']]

# OPTIMIZATION 2: Lower the Epochs
# With 5,300 rows of data, 15 loops is more than enough!
epochs = 15

print(f"Starting high-speed training for {epochs} epochs...")

# 4. Fast Training Loop
for epoch in range(epochs):
    total_loss = 0
    # Iterate through our pre-calculated lists instead of Pandas
    for i in range(len(queries_data)):
        model.zero_grad()
        output = model(queries_data[i])
        loss = loss_function(output, targets_data[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(df):.4f}")

# 5. Save Model
torch.save(model.state_dict(), "search_model.pth")
print("\nSuccess! Model saved.")