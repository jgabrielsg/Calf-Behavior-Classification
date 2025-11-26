# DeepCalf-SSL: Accelerometer-Based Behavior Analysis

---

Participants:
- João Gabriel - FGV (EMAp)
- Gustavo Tironi - FGV (EMAp)

---

This repository contains a PyTorch implementation for classifying pre-weaned calf behaviors using high-frequency accelerometer data. It leverages the AcTBeCalf dataset and implements Semi-Supervised Learning techniques (such as FixMatch) to improve classification performance on rare behaviors by utilizing large-scale unlabeled sensor data.

### Key Features:

- Data Processing: Optimized pipelines for handling large-scale parquet files using Polars.
- Modeling: ResNet-1D and InceptionTime architectures tailored for time-series.
- Methodology: Subject-independent splitting and semi-supervised training loops.
