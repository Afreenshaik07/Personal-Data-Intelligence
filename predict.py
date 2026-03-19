import torch
import sys
import os
import pandas as pd
from scripts.model import SimpleTextClassifier

# Load the same vocab used in training
df = pd.read_csv("labeled_data.csv")
all_words = " ".join(df['query'].astype(str)).lower().split()
vocab = {word: i+1 for i, word in enumerate(set(all_words))}

def predict_category(text):
    vocab_size = len(vocab) + 1
    model = SimpleTextClassifier(vocab_size, 16, 3)
    model.load_state_dict(torch.load("search_model.pth", weights_only=True))
    model.eval()
    
    idxs = [vocab.get(w, 0) for w in text.lower().split()]
    if not idxs: idxs = [0]
    query_in = torch.tensor([idxs], dtype=torch.long)
    
    with torch.no_grad():
        output = model(query_in)
        cat_id = torch.argmax(output, dim=1).item()
        
    categories = {0: "Coding", 1: "Entertainment", 2: "Life"}
    return categories[cat_id]

if __name__ == "__main__":
    user_input = input("Enter a search query: ")
    print(f"Result: {predict_category(user_input)}")