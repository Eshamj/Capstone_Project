import boto3
import pandas as pd
import numpy as np
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
BUCKET = "iot-traffic-esha"
KEY = "filtered_air_quality.csv"         # preprocessed file from edge
RESULTS_FILE = "hybrid_airquality_results_xgboost.csv"
MEMBER_NAME = "Esha Maria Joseph"
ARCHITECTURE = "Edge–Cloud Hybrid"
MODEL_NAME = "XGBoostRegressor"

# ----------------------------------------------------
# STEP 1 — Download data from S3
# ----------------------------------------------------
s3 = boto3.client("s3")
print(" Downloading preprocessed data from S3...")

obj = s3.get_object(Bucket=BUCKET, Key=KEY)
df = pd.read_csv(io.BytesIO(obj["Body"].read()))
print(" Data downloaded successfully!")
print("Initial shape:", df.shape)

# ----------------------------------------------------
# STEP 2 — Data cleaning
# ----------------------------------------------------
target_col = "CO(GT)"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset")

# Remove invalid or missing CO values (usually -200)
invalid_count = (df[target_col] <= 0).sum()
df = df[df[target_col] > 0]
print(f" Removed {invalid_count} invalid CO(GT) readings. New shape: {df.shape}")

# ----------------------------------------------------
# STEP 3 — Prepare features and target
# ----------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------------------------------
# STEP 4 — Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------
# STEP 5 — Train fine-tuned XGBoost model
# ----------------------------------------------------
print(" Training optimized XGBoost model (cloud simulation)...")

t0 = time.time()
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.2,       # L2 regularization
    reg_alpha=0.3,        # L1 regularization
    gamma=0.2,            # Minimum loss reduction to make a split
    random_state=42,
    tree_method="hist",   # optimized for speed
    verbosity=0
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
t1 = time.time()
latency = t1 - t0

# ----------------------------------------------------
# STEP 6 — Evaluate Metrics
# ----------------------------------------------------
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n Training complete!")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Latency: {latency:.2f} sec")

# ----------------------------------------------------
# STEP 7 — Save results locally
# ----------------------------------------------------
results_df = pd.DataFrame([{
    "Member": MEMBER_NAME,
    "Dataset": "Air Quality",
    "Architecture": ARCHITECTURE,
    "Model": MODEL_NAME,
    "R2_Score": r2,
    "MAE": mae,
    "RMSE": rmse,
    "Latency_train_predict_sec": latency
}])

results_df.to_csv(RESULTS_FILE, index=False)
print(f" Results saved locally as {RESULTS_FILE}")

# ----------------------------------------------------
# STEP 8 — Upload results to S3
# ----------------------------------------------------
print("☁ Uploading results CSV to S3...")
s3.upload_file(RESULTS_FILE, BUCKET, RESULTS_FILE)
print(f" Uploaded results successfully to s3://{BUCKET}/{RESULTS_FILE}")
print(" XGBoost hybrid training phase completed successfully!")

