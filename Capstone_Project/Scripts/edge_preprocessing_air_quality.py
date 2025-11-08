import pandas as pd
import numpy as np
import boto3
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Step 1: Load data
t0 = time.time()
data = pd.read_excel("AirQualityUCI.xlsx")
print("Initial shape:", data.shape)
print("Columns:", list(data.columns))

# Step 2: Basic cleaning
# Drop useless columns and rows with NaN
drop_cols = ['Date', 'Time']
data = data.drop(columns=drop_cols, errors='ignore')
data = data.dropna()

# Step 3: Define target column
target_col = 'CO(GT)'

# Filter only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Step 4: Split features and target
X = numeric_data.drop(columns=[target_col])
y = numeric_data[target_col]

# Step 5: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Step 6: Feature reduction (simulate edge filtering)
sel = VarianceThreshold(threshold=0.1)
X_reduced = sel.fit_transform(X)
selected_features = X.columns[sel.get_support()]  # ✅ keep feature names
print("Reduced feature count:", X_reduced.shape[1])
print("Selected features:", list(selected_features))

# Step 7: Combine and save (preserve feature names)
filtered_df = pd.DataFrame(X_reduced, columns=selected_features)
filtered_df[target_col] = y.values
filtered_df.to_csv("filtered_air_quality.csv", index=False)

t1 = time.time()
preprocessing_time = t1 - t0
print(f"✅ Preprocessing completed in {preprocessing_time:.2f} seconds")

# Step 8: Upload to AWS S3
s3 = boto3.client('s3')
bucket = "iot-traffic-esha"  # your existing bucket
key = "filtered_air_quality.csv"

t_upload_start = time.time()
s3.upload_file("filtered_air_quality.csv", bucket, key)
t_upload_end = time.time()
upload_time = t_upload_end - t_upload_start

print(f"✅ Uploaded to S3 ({bucket}/{key}) successfully!")
print(f"Upload Time: {upload_time:.2f} sec, File Size: {os.path.getsize('filtered_air_quality.csv')} bytes")
