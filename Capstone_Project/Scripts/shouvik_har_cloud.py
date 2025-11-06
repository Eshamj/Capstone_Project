import time, datetime, os, sys
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BUCKET = ""
RESULTS_CSV = "shouvik_results.csv"
MODEL_FILE = "rf_model.joblib"

def read_space(path):
    return pd.read_csv(path, header=None, delim_whitespace=True)

def main():
    t0 = time.time()
    X_train = read_space("X_train.txt")
    y_train = read_space("y_train.txt")
    X_test  = read_space("X_test.txt")
    y_test  = read_space("y_test.txt")
    features = read_space("features.txt")
    t1 = time.time()
    download_time = t1 - t0

    p0 = time.time()
    X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_test], axis=0).iloc[:,0].reset_index(drop=True)
    X.columns = features.iloc[:,1].values
    X = X.fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    p1 = time.time()
    preprocess_time = p1 - p0

    m0 = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_enc)
    y_pred = model.predict(X_scaled)
    m1 = time.time()
    train_predict_time = m1 - m0
    acc = accuracy_score(y_enc, y_pred)

    joblib.dump(model, MODEL_FILE)

    row = {
        "Timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "Dataset": "HAR",
        "Architecture": "Cloud-Only",
        "Model": "RandomForest",
        "Accuracy": round(float(acc),6),
        "Latency_sec": round(float(train_predict_time),6),
        "Preprocessing_sec": round(float(preprocess_time),6),
        "Upload_sec": round(float(download_time),6),
        "Cloud_Instance": os.environ.get("EC2_INSTANCE_TYPE","t2.micro"),
        "Cloud_Hours_Used": "",
        "Estimated_Cost_USD": "",
        "Notes": "RandomForest n_estimators=100"
    }

    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False)
    print("\n Experiment finished. Results saved to", RESULTS_CSV)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
