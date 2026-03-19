# 🚀 Personal Data Intelligence

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)

An end-to-end Data Science and Machine Learning pipeline that extracts, cleans, classifies, and visualizes personal Google Search history to uncover behavioral patterns and learning habits.

## 🧠 Project Overview

This project transforms raw HTML exports from Google Takeout into an interactive Business Intelligence dashboard. It features a custom-built **PyTorch Neural Network** that automatically categorizes searches (e.g., "Coding", "Entertainment", "Life") and tracks productivity metrics like coding streaks and peak activity hours.

### ✨ Key Features
* **Custom ETL Pipeline:** Parses messy HTML data, extracts queries, and standardizes varied datetime formats into clean Pandas DataFrames.
* **Deep Learning Classifier:** A custom PyTorch Neural Network featuring Embedding layers and Dropout for robust Natural Language Processing (NLP).
* **Automated Data Engineering:** Runs batch inference on the dataset and structures the results into a relational SQLite database.
* **Interactive BI Dashboard:** A Streamlit web application featuring Plotly data visualizations, dynamic filtering, and custom KPI tracking (e.g., "🔥 Coding Streaks").

---

## 🛠️ Architecture Workflow

1. **Extraction:** `Google Takeout (HTML)` ➔ Parsed via BeautifulSoup.
2. **Auto-Labeling:** Rule-based keyword tagging to generate training data.
3. **Model Training:** `PyTorch` neural network learns search patterns.
4. **Inference & Storage:** Model predicts categories for all historical searches ➔ Saved to `SQLite`.
5. **Visualization:** `Streamlit` reads from SQLite to render the interactive dashboard.

---

## 📁 Project Structure

```text
Personal-Search-Analyzer/
│
├── scripts/
│   └── model.py             # PyTorch Neural Network architecture
│
├── data/                    # Directory for raw HTML exports (Ignored in Git)
├── auto_label.py            # Rule-based script to generate training dataset
├── train.py                 # High-speed PyTorch model training script
├── database_setup.py        # SQLite database builder and batch inference engine
├── dashboard.py             # Streamlit interactive web dashboard
└── README.md                # Project documentation
