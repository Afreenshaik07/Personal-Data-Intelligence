import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import datetime

# Setup page layout
st.set_page_config(page_title="Personal Search Analyzer", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { border-radius: 10px; background-color: #ffffff; padding: 10px; border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Personal Search Intelligence")

# 1. Load Data
conn = sqlite3.connect('my_history.db')
df = pd.read_sql_query("SELECT * FROM searches", conn)
conn.close()

# 2. Pre-process Time Data
# errors='coerce' ensures that if a date is still messy, it doesn't crash the app
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date
df['day_name'] = df['timestamp'].dt.day_name()

# --- SIDEBAR & STREAK CALCULATION ---
st.sidebar.header("Control Panel")

# Calculate Coding Streak
coding_df = df[(df['category'] == 'Coding') & (df['timestamp'].notnull())]
coding_days = sorted(coding_df['date'].unique(), reverse=True)

streak = 0
if len(coding_days) > 0:
    current_date = datetime.date.today()
    # Check if latest coding day was today or yesterday
    if coding_days[0] >= current_date - datetime.timedelta(days=1):
        streak = 1
        for i in range(len(coding_days) - 1):
            if (coding_days[i] - coding_days[i+1]).days == 1:
                streak += 1
            else:
                break

# Display the Streak
if streak > 0:
    st.sidebar.success(f"🔥 Current Coding Streak: {streak} Days")
else:
    st.sidebar.info("No active coding streak. Time to learn!")

# Filters
selected_cat = st.sidebar.multiselect("Category Filter", options=df['category'].dropna().unique(), default=df['category'].dropna().unique())
search_q = st.sidebar.text_input("Quick Find (Keyword)", "")

# Apply Filters
mask = (df['category'].isin(selected_cat)) & (df['query'].str.contains(search_q, case=False, na=False))
filtered_df = df[mask]

# --- KEY METRICS (CRASH-PROOF VERSION) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total History", f"{len(df):,}")
m2.metric("Filtered Results", f"{len(filtered_df):,}")

# Safely extract Top Category
top_cat_series = filtered_df['category'].mode()
m3.metric("Top Category", top_cat_series.iloc[0] if not top_cat_series.empty else "N/A")

# Safely extract Peak Hour
peak_hour_series = filtered_df['hour'].mode()
m4.metric("Peak Hour", f"{int(peak_hour_series.iloc[0])}:00" if not peak_hour_series.empty else "N/A")

st.divider()

# --- VISUALIZATIONS ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⏰ Hourly Activity")
    if not filtered_df.empty and not filtered_df['hour'].dropna().empty:
        hourly_counts = filtered_df.groupby(['hour', 'category']).size().reset_index(name='counts')
        fig_hour = px.bar(hourly_counts, x='hour', y='counts', color='category', barmode='stack')
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("No data available for the current filters.")

with col2:
    st.subheader("🎯 Category Split")
    if not filtered_df.empty:
        fig_pie = px.pie(filtered_df, names='category', hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data available.")

# --- LOGS ---
st.subheader("🔍 Search Logs")
st.dataframe(filtered_df.sort_values(by='timestamp', ascending=False), use_container_width=True)