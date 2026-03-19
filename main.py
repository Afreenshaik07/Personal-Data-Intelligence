from scripts.parser import clean_html_history
import pandas as pd

# 1. Define paths to your files
search_path = "data/Takeout/My Activity/Search/My Activity.html"
youtube_path = "data/Takeout/My Activity/YouTube/My Activity.html"

print("Starting the cleaning process...")

# 2. Run the cleaning function
search_df = clean_html_history(search_path, "Google Search")
youtube_df = clean_html_history(youtube_path, "YouTube")

# 3. Combine them into one big list
all_data = pd.concat([search_df, youtube_df], ignore_index=True)

# 4. Save to CSV so we can use it for PyTorch next
all_data.to_csv("my_cleaned_data.csv", index=False)

print(f"Done! Created 'my_cleaned_data.csv' with {len(all_data)} searches.")
print(all_data.head()) # Shows you the first 5 rows