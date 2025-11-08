import boto3
import pandas as pd
import numpy as np
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
BUCKET = "iot-traffic-esha"               # your bucket name
KEY = "filtered_air_quality.csv"          # file uploaded from edge
RESULTS_FILE = "hybrid_airquality_results.csv"
MEMBER_NAME = "Esha Maria Joseph"         # your name for result logging
ARCHITECTURE = "Edge–Cloud Hybrid"        # to compare with Cloud-Only later
MODEL_NAME = "RandomForestRegressor"

# ----------------------------------------------------
# STEP 1 — Download preprocessed dataset from S3
# ----------------------------------------------------
s3 = boto3.client("s3")
print("📥 Downloading preprocessed data from S3...")

obj = s3.get_object(Bucket=BUCKET, Key=KEY)
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

print("✅ Data downloaded successfully!")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ----------------------------------------------------
# STEP 2 — Split features and target
# ----------------------------------------------------
target_col = "CO(GT)"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset")

X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------------------------------
# STEP 3 — Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------
# STEP 4 — Model Training (simulate cloud training)
# ----------------------------------------------------
print("☁ Training RandomForestRegressor in cloud (simulated)...")

t0 = time.time()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
t1 = time.time()

latency = t1 - t0

# ----------------------------------------------------
# STEP 5 — Evaluate Metrics
# ----------------------------------------------------
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"✅ Training complete!")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Latency: {latency:.2f} sec")

# ----------------------------------------------------
# STEP 6 — Save results to CSV
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
print(f"📄 Results saved locally as {RESULTS_FILE}")

# ----------------------------------------------------
# STEP 7 — Upload results CSV to S3
# ----------------------------------------------------
print("☁ Uploading results CSV to S3...")

s3.upload_file(RESULTS_FILE, BUCKET, RESULTS_FILE)

print(f"✅ Results uploaded successfully to s3://{BUCKET}/{RESULTS_FILE}")
print("🏁 Hybrid training phase completed successfully!")
