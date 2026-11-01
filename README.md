# Lightweight ML for IoT Intrusion Detection

> Multi-class attack detection in IoT networks with model explainability (SHAP/LIME) and zero-day generalization testing.

## Project Overview

This project reproduces and extends the baseline from **Alve et al. (2025)** — *"Smart IoT Security: Lightweight Machine Learning Techniques for Multi-Class Attack Detection in IoT Networks"* (arXiv:2502.04057) — which trains Decision Tree / Random Forest on the **CICIoT2023** dataset (34 attack types, ~10 categories), reporting ~99.56% accuracy.

### Novel Contributions
1. **Explainability (SHAP + LIME)** — Which network flow features drive each attack classification? Per-class feature importance, local explanations for misclassified samples, and SHAP vs LIME agreement analysis.
2. **Zero-Day Generalization** — Can tree-based models detect attack types they've never seen during training?

## Dataset

**CICIoT2023** from the Canadian Institute for Cybersecurity, University of New Brunswick.
- 34 attack types across 8 categories (DDoS, DoS, Mirai, Recon, Web, Spoofing, BruteForce) + Benign
- ~46 network flow features (packet lengths, inter-arrival times, protocol flags, etc.)
- 310+ CSV files, ~8.9 GB total

### Download
1. Download from [Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset-2023) or [UNB](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
2. Place CSV files in `data/raw/`

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data to data/raw/

# 4. Run EDA
python notebooks/01_data_loading_eda.py

# 5. Launch Frontend & Live Flow Inspector
python run_app.py
# Access dashboard at http://127.0.0.1:8000
```

## Interactive Web Dashboard (Frontend)

The project includes an interactive web dashboard built with **React, Vite, Tailwind CSS, and FastAPI**:
- **Live Network Flow Inspector**: Test individual flow vectors across all 40 features or edge-pruned 15 features.
- **Curated Test Presets**: 1-click injection of real network traffic flows (Benign, Mirai, DDoS, DoS, Recon, Web, Spoofing, BruteForce) sampled directly from the CICIoT2023 test set.
- **Real-Time Edge Telemetry**: Microsecond-precision inference latency measurements (~30-50ns for Decision Trees).
- **Batch CSV Audit**: Upload bulk flow CSV logs for automated threat breakdown and throughput analysis.
- **Benchmark Comparison**: Side-by-side performance comparison against Alve et al. (2025).

```bash
# To run both the backend API and frontend:
python run_app.py

# For frontend development mode with HMR:
cd frontend && npm run dev

## Project Structure

```
├── configs/                  # Experiment configuration (YAML)
├── data/
│   ├── raw/                  # Original CICIoT2023 CSVs
│   ├── processed/            # Cleaned data
│   └── splits/               # Train/val/test parquet files
├── models/                   # Saved trained models (.joblib)
├── notebooks/                # Step-by-step scripts
├── results/
│   ├── figures/              # All plots
│   ├── tables/               # Metrics tables (CSV)
│   └── logs/                 # Experiment logs
├── report/                   # Final report (LaTeX/docx)
└── src/                      # Source modules
    ├── data_loader.py        # CSV loading, label mapping
    ├── preprocessing.py      # Cleaning, encoding, splitting
    ├── feature_engineering.py # Feature selection
    ├── models.py             # Model training (DT, RF, XGBoost, LR, KNN)
    ├── evaluation.py         # Metrics, confusion matrices
    ├── explainability.py     # SHAP + LIME analysis
    ├── zero_day.py           # Zero-day hold-out experiment
    └── utils.py              # Logging, timing, config
```

## Models Evaluated

| Model | Reference |
|-------|-----------|
| Decision Tree | Baseline (Alve et al.) |
| Random Forest | Baseline (Alve et al.) |
| XGBoost | Additional comparison |
| Logistic Regression | Simple baseline |
| K-Nearest Neighbors | Simple baseline |

## Evaluation Metrics

- Accuracy, Macro/Weighted F1, Precision, Recall
- Per-class F1-score (34-class and 8-category)
- ROC-AUC (one-vs-rest)
- Confusion matrix (normalized)
- Training time, inference latency, model size
- SHAP/LIME feature importance rankings
- Zero-day binary detection recall

## References

1. Alve et al. (2025). "Smart IoT Security: Lightweight Machine Learning Techniques for Multi-Class Attack Detection in IoT Networks." arXiv:2502.04057.
2. Neto et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment." Sensors, MDPI.
3. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
