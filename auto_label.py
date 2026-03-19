import pandas as pd
import os

if not os.path.exists("labeled_data.csv"):
    print("Error: Could not find labeled_data.csv")
    exit()

df = pd.read_csv("labeled_data.csv")

tech_keywords = ['deepseek', 'lms', 'intellipaat', 'n8n', 'aws', 'credly', 'snowflake', 'data', 'ai', 'python', 'pytorch', 'code', 'error', 'sql', 'react', 'tutorial', 'tech', 'software', 'developer', 'api', 'certification', 'github', 'google bit']
ent_keywords = ['song', 'music', 'video', 'shorts', 'youtube', 'movie', 'trailer', 'bts', 'comedy', 'funny', 'game', 'play', 'kiss', 'photography', 'trending', 'viral', 'watch']
life_keywords = ['food', 'streetfood', 'recipe', 'weather', 'news', 'shop', 'buy', 'garden', 'plants', 'pi day', 'surat', 'maharaja', 'coco', 'momos', 'jaipur', 'crab']

def auto_categorize(row):
    query = str(row['query']).lower()
    platform = str(row['platform']).lower()
    
    for kw in tech_keywords:
        if kw in query: return 0
    for kw in ent_keywords:
        if kw in query: return 1
    for kw in life_keywords:
        if kw in query: return 2
    if 'youtube' in platform:
        return 1
    return 2

print("Scanning and labeling your data...")
df['label'] = df.apply(auto_categorize, axis=1)

# FORCE SAVE TO A BRAND NEW FILE NAME
df.to_csv("labeled_data_final.csv", index=False)
print("✅ Success! Created brand new file: 'labeled_data_final.csv'")
print("\nNew Label Counts:")
print(df['label'].value_counts())