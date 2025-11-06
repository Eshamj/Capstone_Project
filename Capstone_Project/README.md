# Empirical Comparison of Cloud-Only and Hybrid Edge–Cloud Architectures for Real-Time IoT Analytics  

This project presents a comparative study between **Cloud-Only** and **Hybrid Edge–Cloud** architectures for real-time IoT data analytics.  
The objective was to evaluate trade-offs in **accuracy**, **latency**, and **cost** across two representative IoT datasets —  
**Human Activity Recognition (HAR)** and **Smart Mobility Traffic** — using the AWS Free Tier environment.

---

##  Overview  

Cloud computing provides scalability and flexibility for large-scale analytics,  
but often suffers from latency and bandwidth limitations in time-sensitive IoT applications.  
To mitigate these, **hybrid edge–cloud architectures** perform partial processing closer to the data source,  
reducing data transmission and inference time.  

This study empirically compares both setups to quantify their real-world performance.

---

##  Architecture Summary  

| Architecture | Description |
|---------------|-------------|
| **Cloud-Only** | Entire workflow (preprocessing, training, inference) performed on AWS EC2. |
| **Hybrid Edge–Cloud** | Data partially preprocessed locally (edge) and only reduced dataset uploaded to the cloud for final model training and inference. |

---

##  Experimental Setup  

- **Platform:** AWS Free Tier (t3.micro EC2 instance, S3 storage)  
- **Languages & Tools:** Python 3.9+, pandas, numpy, scikit-learn, boto3  
- **Datasets:**  
  - *UCI Human Activity Recognition (HAR)*  
  - *Smart Mobility Traffic Dataset (Kaggle)*  
- **Model Used:** Random Forest Classifier  
- **Split:** 70% training / 30% testing  

---

##  Results Summary  

| Member | Dataset | Architecture | Accuracy (%) | Latency (sec) | Preprocessing (s) | Upload (s) | Cloud Instance | Cost (₹) | Notes |
|:--------|:----------|:--------------|:--------------|:----------------|:-------------------|:--------------|:----------------|:-----------|:--------|
| **Lipika** | HAR | Hybrid Edge–Cloud | 92.6 | 16.33 | 16 | 6 | t3.micro | 8.3 | Mean/std feature selection |
| **Shouvik** | HAR | Cloud-Only | 92.6 | 6.00 | 0.9 | 6 | t3.micro | 8.3 | Cloud-only full training |
| **Esha** | Smart Mobility | Hybrid Edge–Cloud | 100.0 | 0.16 | 15 | 6 | t3.micro | 8.3 | Reduced data from edge |
| **Ash** | Smart Mobility | Cloud-Only | 99.86 | 0.60 | 12 | 8 | t3.micro | 8.3 | Full dataset on cloud |

---

##  Key Observations  

- Cloud-only setups achieved slightly higher consistency in accuracy.  
- Hybrid setups demonstrated significantly lower latency and upload size,  
  showing their potential for real-time applications despite small accuracy drops.  
- AWS Free Tier provided cost-effective infrastructure for testing both setups.  

---


---

##  Methodology  

A detailed explanation of the experimental setup, preprocessing pipeline, and model training process is provided in  
**[Methodology.docx](./Methodology.docx)** — outlining each phase from data collection to evaluation.

---

##  Conclusion  

The hybrid edge–cloud approach, while slightly less accurate, achieved faster inference with reduced bandwidth and computation costs.  
This trade-off demonstrates that hybrid designs are well-suited for **real-time IoT analytics** such as smart city and mobility systems,  
where rapid response is more critical than marginal accuracy gains.

---

##  Contributors  

- **Esha Maria Joseph** — Smart Mobility (Hybrid)  
- **Aswathy M** — Smart Mobility (Cloud-Only)  
- **Lipika Kaushal** — HAR (Hybrid)  
- **Shouvik Bhowmick** — HAR (Cloud-Only)

---

##  Repository Information  

This repository contains the complete codebase, experimental results, and documentation  
for the Capstone Project *“Empirical Comparison of Cloud-Only and Hybrid Edge–Cloud Architectures for Real-Time IoT Analytics”*  
conducted under Group 6, VIT Amaravati.


