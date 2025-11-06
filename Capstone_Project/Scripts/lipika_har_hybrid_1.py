# Placeholder script - original version executed on AWS EC2
import boto3
import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime

# --- AWS S3 Setup ---
bucket_name = "lipika-capstone-har-hybrid"
prefix = "HAR/"
region = "ap-south-1"

s3 = boto3.client("s3", region_name=region)

# --- Download dataset files from S3 to EC2 ---
files = ["X_train.txt", "X_test.txt", "y_train.txt", "y_test.txt", "features.txt"]

for file in files:
    s3.download_file(bucket_name, prefix + file, file)
    print(f"Downloaded {file}")

# --- Load Data ---
X_train = pd.read_csv("X_train.txt", sep=r"\s+", header=None)
X_test = pd.read_csv("X_test.txt", sep=r"\s+", header=None)
y_train = pd.read_csv("y_train.txt", sep=r"\s+", header=None)
y_test = pd.read_csv("y_test.txt", sep=r"\s+", header=None)
features = pd.read_csv("features.txt", sep=r"\s+", header=None)

feature_names = features[1].tolist()
X_train.columns = feature_names
X_test.columns = feature_names

# --- Feature Selection (Edge preprocessing) ---
start_pre = time.time()
selected_features = [col for col in feature_names if "mean()" in col or "std()" in col]
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
end_pre = time.time()
preprocessing_time = end_pre - start_pre
print(f" Preprocessing completed in {preprocessing_time:.2f} sec")

# --- Model Training (Cloud inference) ---
start_latency = time.time()

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'max_features': ['sqrt']
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=4,
    cv=2,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X_train_reduced, y_train.values.ravel())
best_rf = random_search.best_estimator_
print("Best Parameters Found:", random_search.best_params_)

y_pred = best_rf.predict(X_test_reduced)

end_latency = time.time()
latency_train_predict = end_latency - start_latency

accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model Accuracy: {accuracy:.4f}")
print(f"Latency (train+predict): {latency_train_predict:.2f} sec")

# --- Save Results ---
timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")

results = pd.DataFrame([{
    "Timestamp": timestamp,
    "Dataset": "Human Activity Recognition",
    "Architecture": "Hybrid Edge-Cloud",
    "Model": "Random Forest",
    "Accuracy (%)": round(accuracy * 100, 2),
    "Latency (sec)": round(latency_train_predict, 2),
    "Preprocessing Time (s)": round(preprocessing_time, 2),
    "Upload Time (s)": 6,  # fixed simulated value
    "Cloud Instance": "t3.micro",
    "Cloud Hours Used": 0.5,
    "Estimated Cost (₹)": 8.3,
    "Notes": "Hybrid setup with AWS Free Tier"
}])

results.to_csv("lipika.csv", index=False)
print(" Results saved to lipika.csv successfully!")
