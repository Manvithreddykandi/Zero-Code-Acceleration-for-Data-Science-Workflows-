# scripts/baseline_cpu.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import os

# --- Data Loading ---
start_load = time.time()
# Use an absolute or relative path that works from where you run the script
df = pd.read_csv("../datasets/nyc_taxi_sample.csv")
load_time = time.time() - start_load

# --- Data Processing ---
start_process = time.time()
avg_distance = df.groupby('VendorID')['trip_distance'].mean()
process_time = time.time() - start_process

# --- Model Training ---
# Prepare data for Scikit-learn
# Using .values is important for compatibility with some accelerators
X = df[['passenger_count', 'trip_distance']].astype('float32').values
y = (df['trip_distance'] > 2).astype('int32').values

start_train = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
train_time = time.time() - start_train

# --- Print Results ---
print("--- Baseline (Pandas + Scikit-learn) ---")
print(f"Data Loading Time:    {load_time:.4f} seconds")
print(f"GroupBy Aggregation Time: {process_time:.4f} seconds")
print(f"Model Training Time:    {train_time:.4f} seconds")