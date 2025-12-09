import pandas as pd
import os

# Paths
RAW_DATA_PATH = "data/raw/MachineLearningRating_v3.txt"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

# Ensure processed folder exists
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

# Read raw data (pipe-separated)
df = pd.read_csv(RAW_DATA_PATH, sep='|', low_memory=False)

# Optional: basic cleaning
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Replace empty strings with NaN
df.replace('', pd.NA, inplace=True)

# Save processed data
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"Processed data saved to {PROCESSED_DATA_PATH}")
