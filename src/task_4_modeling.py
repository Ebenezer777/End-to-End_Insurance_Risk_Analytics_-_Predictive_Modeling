# task_4_modeling.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -----------------------------
# Paths
# -----------------------------
data_path = "data/processed/ml_clean.csv"
visualizations_path = "data/visualizations"
os.makedirs(visualizations_path, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(data_path)

# -----------------------------
# Column selection
# -----------------------------
# Target
target = 'TotalClaims'

# Numeric columns
num_cols = [
    'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
    'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding',
    'SumInsured', 'CalculatedPremiumPerTerm'
]

# Categorical columns (selecting a few for modeling)
cat_cols = [
    'IsVATRegistered', 'LegalType', 'Title', 'Language', 'Bank',
    'AccountType', 'MaritalStatus', 'Gender', 'Country', 'Province',
    'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType',
    'make', 'Model', 'bodytype', 'NewVehicle', 'WrittenOff'
]

# -----------------------------
# Data cleaning
# -----------------------------
# Convert dates
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')

# Convert RegistrationYear to numeric
if df['RegistrationYear'].dtype == object:
    df['RegistrationYear'] = pd.to_datetime(df['RegistrationYear'], errors='coerce').dt.year

# Convert numeric columns
for col in num_cols + [target]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Fill missing values
df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -----------------------------
# Quick visualization
# -----------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols + [target]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(visualizations_path, "correlation_heatmap.png"))
plt.close()

# -----------------------------
# Feature encoding
# -----------------------------
X = df[num_cols + cat_cols].copy()
y = df[target]

# One-hot encoding
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------
print("Training LinearRegression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("LinearRegression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("LinearRegression R2:", r2_score(y_test, y_pred_lr))

joblib.dump(lr_model, "models/linear_regression.pkl")

# -----------------------------
# Random Forest
# -----------------------------
print("Training RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("RandomForest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("RandomForest R2:", r2_score(y_test, y_pred_rf))

joblib.dump(rf_model, "models/random_forest.pkl")

# -----------------------------
# XGBoost
# -----------------------------
print("Training XGBRegressor...")
xgb_model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))

joblib.dump(xgb_model, "models/xgb_model.pkl")

# -----------------------------
# Feature importance visualization (Random Forest example)
# -----------------------------
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10,8))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(visualizations_path, "top20_feature_importances.png"))
plt.close()

print("Task 4 modeling complete. Models and visualizations saved.")
