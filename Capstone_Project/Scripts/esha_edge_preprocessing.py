import pandas as pd
import boto3

# Load dataset
data = pd.read_csv("smart_mobility_dataset.csv")
print("Initial shape:", data.shape)
# Keep all important features + correct target column
keep_cols = [
    'Timestamp','Vehicle_Count','Traffic_Speed_kmh','Road_Occupancy_%',
    'Traffic_Light_State','Weather_Condition','Accident_Report',
    'Ride_Sharing_Demand','Parking_Availability','Emission_Levels_g_km',
    'Energy_Consumption_L_h','Traffic_Condition'
]
data = data[keep_cols]
# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['Traffic_Light_State','Weather_Condition','Accident_Report'])
print("Filtered shape:", data.shape)
# Save locally
data.to_csv("filtered_data_v2.csv", index=False)
#Upload to S3
s3 = boto3.client('s3')
s3.upload_file("filtered_data_v2.csv", "iot-traffic-esha", "filtered_data_v2.csv")
print("Filtered data (v2) uploaded to S3 successfully!")
# Placeholder script - original version executed on AWS EC2
