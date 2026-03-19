import sqlite3
import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.model import SimpleTextClassifier

# 1. Database Setup
conn = sqlite3.connect('my_history.db')
cursor = conn.cursor()
cursor.execute('DROP TABLE IF EXISTS searches') 
cursor.execute('''
    CREATE TABLE searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        platform TEXT,
        category TEXT,
        timestamp DATETIME
    )
''')

# 2. Load Data from the BRAND NEW file
full_df = pd.read_csv("labeled_data_final.csv")

# 3. Rebuild Vocab
train_df = full_df.copy()
train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
train_df = train_df.dropna(subset=['label'])

all_words = " ".join(train_df['query'].astype(str)).lower().split()
vocab = {word: i+1 for i, word in enumerate(set(all_words))}
vocab_size = len(vocab) + 1

# 4. Load Model
model = SimpleTextClassifier(vocab_size, 16, 3)
model.load_state_dict(torch.load("search_model.pth", weights_only=True))
model.eval()

categories = {0: "Coding", 1: "Entertainment", 2: "Life"}

print("Processing into SQL...")

# 5. Predict and Insert
for _, row in full_df.iterrows():
    text = str(row['query'])
    platform = str(row['platform']) if 'platform' in full_df.columns else "Unknown"
    time_val = str(row['timestamp']) if 'timestamp' in full_df.columns and pd.notnull(row.get('timestamp')) else None
    
    idxs = [vocab.get(w, 0) for w in text.lower().split()]
    query_in = torch.tensor([idxs if idxs else [0]], dtype=torch.long)
    
    with torch.no_grad():
        output = model(query_in)
        cat_id = torch.argmax(output, dim=1).item()
        category = categories[cat_id]
    
    cursor.execute("INSERT INTO searches (query, platform, category, timestamp) VALUES (?, ?, ?, ?)", 
                   (text, platform, category, time_val))

conn.commit()
conn.close()
print("Success! 'my_history.db' is ready.")