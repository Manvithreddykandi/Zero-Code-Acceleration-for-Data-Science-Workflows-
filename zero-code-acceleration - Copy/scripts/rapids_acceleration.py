# scripts/rapids_acceleration.py
import cudf as pd  # <<< CHANGE #1: Use cuDF for GPU DataFrames
from cuml.ensemble import RandomForestClassifier  # <<< CHANGE #2: Use cuML for GPU ML
import time
import os

# --- Data Loading ---
start_load = time.time()
df = pd.read_csv("../datasets/nyc_taxi_sample.csv")
load_time = time.time() - start_load

# --- Data Processing ---
start_process = time.time()
avg_distance = df.groupby('VendorID')['trip_distance'].mean()
process_time = time.time() - start_process

# --- Model Training ---
# For cuML, it's common to keep data as cuDF series/DataFrames
X = df[['passenger_count', 'trip_distance']].astype('float32')
y = (df['trip_distance'] > 2).astype('int32')

start_train = time.time()
# Use cuML's RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
train_time = time.time() - start_train

# --- Print Results ---
print("--- RAPIDS (GPU) Acceleration ---")
print(f"Data Loading Time:    {load_time:.4f} seconds")
print(f"GroupBy Aggregation Time: {process_time:.4f} seconds")
print(f"Model Training Time:    {train_time:.4f} seconds")