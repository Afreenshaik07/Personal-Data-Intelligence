import pandas as pd
from bs4 import BeautifulSoup
import os

def clean_html_history(file_path, platform):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

    records = []
    cells = soup.find_all('div', class_='content-cell')
    
    for cell in cells:
        text = cell.get_text()
        if "Searched for" in text or "Watched" in text:
            # 1. Extract Query
            query = text.replace("Searched for", "").replace("Watched", "").strip()
            clean_query = query.split('\xa0')[0] 
            
            # 2. Extract Timestamp (Handles multiple formats)
            try:
                raw_date = text.split('\xa0')[-1].strip()
                # 'format="mixed"' is the modern way to handle varied Google date strings
                timestamp = pd.to_datetime(raw_date, format='mixed', errors='coerce')
            except:
                timestamp = None

            records.append({
                "query": clean_query, 
                "platform": platform, 
                "timestamp": timestamp
            })

    return pd.DataFrame(records)