import time
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# Step 1: Download the new dataset
s3 = boto3.client('s3')
bucket_name = 'iot-traffic-esha'
file_name = 'filtered_data_v2.csv'
s3.download_file(bucket_name, file_name, file_name)
# Step 2: Load dataset
data = pd.read_csv(file_name)
print("Data loaded:", data.shape)
# Step 3: Encode categorical target
X = data.drop(columns=['Traffic_Congestion_Level', 'Timestamp'])
y = data['Traffic_Congestion_Level']
# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Train model
start_time = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
end_time = time.time()
# Step 6: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
latency = end_time - start_time
print(f"Model trained successfully!")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Training Latency: {latency:.2f} seconds")
# Step 7: Save and upload model
model_file = 'rf_model_v2.pkl'
joblib.dump(model, model_file)
s3.upload_file(model_file, bucket_name, model_file)
print("Model uploaded successfully as rf_model_v2.pkl")
