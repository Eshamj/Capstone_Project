# Empirical Comparison of Cloud-Only and Hybrid Edge–Cloud Architectures for Real-Time IoT Analytics

## Team Members and Assignments
| Member | Dataset | Architecture | Model | Accuracy (%) | Latency (sec) | Preprocessing (s) | Upload (s) | Cloud Instance | Cloud Hours | Cost (₹) |
|---------|----------|---------------|--------|----------------|----------------|-------------------|-------------|----------------|--------------|-----------|
| Esha | Smart Mobility Traffic | Hybrid Edge–Cloud | Random Forest | 100.00 | 0.16 | 15 | 6 | t3.micro | 0.5 | 8.3 |
| Ash | Smart Mobility Traffic | Cloud-only | Random Forest | 99.90 | 0.60 | 12 | 8 | t3.micro | 0.5 | 8.3 |
| Lipika | Human Activity Recognition | Hybrid Edge–Cloud | Random Forest | 92.60 | 16.33 | 0 | 6 | t3.micro | 0.5 | 8.3 |
| Shouvik | Human Activity Recognition | Cloud-only | Random Forest | 92.60 | 17.04 | 0.93 | 0 | t3.micro | 0.5 | 8.3 |

---

## Summary
This project compares cloud-only and hybrid edge–cloud IoT analytics architectures using real-world datasets and AWS infrastructure. The hybrid model reduced latency and upload cost while maintaining accuracy comparable to cloud-only systems.
