import boto3, io, time, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BUCKET = 'iot-project-ash-20251101'
KEY = 'smart_mobility_dataset.csv'
RESULTS_FILE = 'ash_results.csv'

def main():
    s3 = boto3.client('s3')
    print("📥 Downloading dataset from S3...")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    df = df.dropna()

    target_col = 'Traffic_Congestion_Level'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found! Columns are:", list(df.columns))
        return

    le = LabelEncoder()
    df['target'] = le.fit_transform(df[target_col])

    X = df.drop(columns=[target_col, 'target']).select_dtypes(include=['float64','int64'])
    y = df['target']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)

    print("🧠 Training RandomForestClassifier...")
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    t1 = time.time()

    acc = accuracy_score(y_test, y_pred)
    duration = t1 - t0

    print(f"✅ Accuracy = {acc:.4f}, Training+Prediction = {duration:.2f} sec")

    pd.DataFrame([{
        "Member": "Ash",
        "Dataset": "Smart_Mobility",
        "Architecture": "Cloud-Only",
        "Model": "RandomForest",
        "Accuracy": acc,
        "Latency_train_predict_sec": duration
    }]).to_csv(RESULTS_FILE, index=False)

    s3.upload_file(RESULTS_FILE, BUCKET, RESULTS_FILE)
    print("☁ Uploaded ash_results.csv to S3 successfully!")

if _name_ == "_main_":
    main()