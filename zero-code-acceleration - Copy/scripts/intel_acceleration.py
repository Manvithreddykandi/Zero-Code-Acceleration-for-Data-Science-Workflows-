# scripts/intel_acceleration.py
from sklearnex import patch_sklearn
patch_sklearn() # <<< ONLY CHANGE: Patch scikit-learn

# Imports are now accelerated automatically
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import os

# --- Data Loading (Unaffected) ---
start_load = time.time()
df = pd.read_csv("../datasets/nyc_taxi_sample.csv")
load_time = time.time() - start_load

# --- Data Processing (Unaffected) ---
start_process = time.time()
avg_distance = df.groupby('VendorID')['trip_distance'].mean()
process_time = time.time() - start_process

# --- Model Training ---
X = df[['passenger_count', 'trip_distance']].astype('float32').values
y = (df['trip_distance'] > 2).astype('int32').values

start_train = time.time()
# This RandomForestClassifier is now accelerated by oneAPI
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
train_time = time.time() - start_train

# --- Print Results ---
print("--- Intel(R) Extension for Scikit-learn* Acceleration ---")
print(f"Data Loading Time:    {load_time:.4f} seconds (not accelerated)")
print(f"GroupBy Aggregation Time: {process_time:.4f} seconds (not accelerated)")
print(f"Model Training Time:    {train_time:.4f} seconds")