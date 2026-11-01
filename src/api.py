"""
FastAPI backend for IoT Network Intrusion Detection & Live Flow Inspector.
Serves model inference, flow sample loader, and performance telemetry.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

app = FastAPI(
    title="IoT IDS Live Flow Inspector API",
    description="Backend API for Lightweight ML IoT Intrusion Detection on CICIoT2023",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches
MODELS: Dict[str, Any] = {}
ENCODERS: Dict[str, Any] = {}
CURATED_SAMPLES: List[Dict[str, Any]] = []
FEATURE_STATS: Dict[str, Dict[str, float]] = {}

FEATURE_GROUPS = {
    "Flow & Timing": ["flow_duration", "Duration", "Rate", "Srate", "Drate", "IAT"],
    "Packet & Header Sizes": [
        "Header_Length", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
        "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"
    ],
    "Protocol Flags & Counts": [
        "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
        "ack_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count"
    ],
    "Protocols": [
        "HTTP", "HTTPS", "DNS", "SSH", "TCP", "UDP", "ARP", "ICMP", "IPv", "LLC", "Protocol Type"
    ]
}

MODEL_METADATA = {
    "decision_tree_best": {
        "id": "decision_tree_best",
        "name": "Decision Tree (Unconstrained Best)",
        "type": "Decision Tree",
        "features_count": 40,
        "accuracy": 99.44,
        "macro_f1": 83.10,
        "latency_us": 0.05,
        "size_kb": 402.4,
        "recommended": True,
        "description": "Highest multi-class F1 score with ultra-low sub-microsecond latency."
    },
    "decision_tree_top15": {
        "id": "decision_tree_top15",
        "name": "Decision Tree (Edge-Optimized Top-15)",
        "type": "Decision Tree",
        "features_count": 15,
        "accuracy": 98.72,
        "macro_f1": 79.15,
        "latency_us": 0.03,
        "size_kb": 351.6,
        "recommended": False,
        "description": "Pareto-optimal model pruned to top 15 features for edge IoT hardware."
    },
    "decision_tree_paper": {
        "id": "decision_tree_paper",
        "name": "Decision Tree (Alve et al. Config)",
        "type": "Decision Tree",
        "features_count": 40,
        "accuracy": 98.15,
        "macro_f1": 70.92,
        "latency_us": 0.09,
        "size_kb": 786.6,
        "recommended": False,
        "description": "Direct reproduction of the baseline tree architecture from the 2025 paper."
    },
    "random_forest_paper": {
        "id": "random_forest_paper",
        "name": "Random Forest (Alve et al. Config)",
        "type": "Random Forest",
        "features_count": 40,
        "accuracy": 99.48,
        "macro_f1": 74.50,
        "latency_us": 1.63,
        "size_kb": 72981.9,
        "recommended": False,
        "description": "High-accuracy ensemble baseline with 100+ estimators."
    },
    "xgboost_baseline": {
        "id": "xgboost_baseline",
        "name": "XGBoost (Lightweight Ensemble)",
        "type": "Gradient Boosting",
        "features_count": 40,
        "accuracy": 99.38,
        "macro_f1": 75.03,
        "latency_us": 1.71,
        "size_kb": 1968.8,
        "recommended": False,
        "description": "Extreme Gradient Boosting model with depth-optimized trees."
    }
}


def load_artifacts():
    global MODELS, ENCODERS, CURATED_SAMPLES, FEATURE_STATS
    
    # Load label encoders
    encoders_path = BASE_DIR / "models" / "label_encoders.joblib"
    if encoders_path.exists():
        ENCODERS = joblib.load(encoders_path)
    
    # Load models
    for model_id in MODEL_METADATA.keys():
        path = BASE_DIR / "models" / f"{model_id}.joblib"
        if path.exists():
            MODELS[model_id] = joblib.load(path)
            
    # Load curated samples
    samples_path = BASE_DIR / "data" / "curated_samples.json"
    if samples_path.exists():
        with open(samples_path, "r") as f:
            CURATED_SAMPLES = json.load(f)

    # Load feature statistics
    stats_path = BASE_DIR / "results" / "tables" / "feature_statistics.csv"
    if stats_path.exists():
        df_stats = pd.read_csv(stats_path, index_col=0)
        FEATURE_STATS = df_stats.to_dict(orient="index")


@app.on_event("startup")
def startup_event():
    load_artifacts()

# Load immediately on import
load_artifacts()


class PredictionRequest(BaseModel):
    model_id: str = "decision_tree_best"
    features: Dict[str, float]


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "samples_count": len(CURATED_SAMPLES),
    }


@app.get("/api/models")
def get_models():
    return list(MODEL_METADATA.values())


@app.get("/api/features")
def get_features():
    results = []
    for group_name, feat_names in FEATURE_GROUPS.items():
        for feat in feat_names:
            stat = FEATURE_STATS.get(feat, {})
            results.append({
                "name": feat,
                "group": group_name,
                "mean": round(float(stat.get("mean", 0.0)), 4) if "mean" in stat else 0.0,
                "min": round(float(stat.get("min", 0.0)), 4) if "min" in stat else 0.0,
                "max": round(float(stat.get("max", 1.0)), 4) if "max" in stat else 1.0,
                "default": round(float(stat.get("50%", 0.0)), 4) if "50%" in stat else 0.0,
            })
    return {
        "features": results,
        "groups": FEATURE_GROUPS,
    }


@app.get("/api/samples")
def get_samples(category: Optional[str] = None):
    if not CURATED_SAMPLES:
        return []
    if category and category.lower() != "all":
        filtered = [s for s in CURATED_SAMPLES if str(s.get("label_category", "")).lower() == category.lower()]
        return filtered
    return CURATED_SAMPLES


@app.post("/api/predict")
def predict_flow(req: PredictionRequest):
    model_id = req.model_id
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not found. Available: {list(MODELS.keys())}")
    
    model = MODELS[model_id]
    
    # Required feature names from model
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        # Fallback to all 40 features
        expected_features = [f for grp in FEATURE_GROUPS.values() for f in grp]

    # Build input row
    row_values = []
    for f in expected_features:
        if f in req.features:
            row_values.append(float(req.features[f]))
        else:
            default_val = FEATURE_STATS.get(f, {}).get("50%", 0.0)
            row_values.append(float(default_val))
            
    X_input = pd.DataFrame([row_values], columns=expected_features)
    
    # Measure latency with nanosecond precision
    t0 = time.perf_counter_ns()
    raw_pred = model.predict(X_input)[0]
    t1 = time.perf_counter_ns()
    latency_us = round((t1 - t0) / 1000.0, 3)
    
    # Decode category
    if "category" in ENCODERS:
        category_name = str(ENCODERS["category"].inverse_transform([raw_pred])[0])
    else:
        category_names = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']
        category_name = category_names[int(raw_pred)] if int(raw_pred) < len(category_names) else f"Class_{raw_pred}"
        
    is_attack = (category_name.lower() != "benign")
    
    # Probabilities if available
    probabilities = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_input)[0]
            cat_classes = ENCODERS["category"].classes_ if "category" in ENCODERS else range(len(probas))
            for c_name, p_val in zip(cat_classes, probas):
                probabilities[str(c_name)] = round(float(p_val), 4)
            confidence = round(float(np.max(probas)), 4)
        except Exception:
            pass
            
    # Key deviating features (vs median/mean benign flow)
    anomalous_features = []
    for f in expected_features[:10]:
        val = req.features.get(f, 0.0)
        mean_val = FEATURE_STATS.get(f, {}).get("mean", 1.0)
        std_val = FEATURE_STATS.get(f, {}).get("std", 1.0)
        if std_val and std_val > 0:
            z_score = abs(val - mean_val) / std_val
            if z_score > 1.5 or (val > 0 and mean_val == 0):
                anomalous_features.append({
                    "feature": f,
                    "value": round(float(val), 4),
                    "mean": round(float(mean_val), 4),
                    "deviation_sigma": round(float(z_score), 2),
                })
    anomalous_features.sort(key=lambda x: x["deviation_sigma"], reverse=True)

    return {
        "model_id": model_id,
        "model_name": MODEL_METADATA.get(model_id, {}).get("name", model_id),
        "prediction_code": int(raw_pred),
        "predicted_category": category_name,
        "is_attack": is_attack,
        "verdict": "ATTACK DETECTED" if is_attack else "NORMAL TRAFFIC (BENIGN)",
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_us": latency_us,
        "features_evaluated": len(expected_features),
        "anomalous_indicators": anomalous_features[:6],
    }


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...), model_id: str = "decision_tree_best"):
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not found.")
    
    model = MODELS[model_id]
    expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [
        f for grp in FEATURE_GROUPS.values() for f in grp
    ]

    try:
        contents = await file.read()
        import io
        df_uploaded = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Check matching features
    available = [f for f in expected_features if f in df_uploaded.columns]
    if len(available) < 5:
        raise HTTPException(status_code=400, detail=f"CSV has too few recognized flow features. Found: {available}")

    # Fill missing with median
    for f in expected_features:
        if f not in df_uploaded.columns:
            df_uploaded[f] = FEATURE_STATS.get(f, {}).get("50%", 0.0)

    X_eval = df_uploaded[expected_features].fillna(0.0)
    
    t0 = time.perf_counter()
    preds = model.predict(X_eval)
    t1 = time.perf_counter()
    
    cat_names = ENCODERS["category"].inverse_transform(preds) if "category" in ENCODERS else [str(p) for p in preds]
    
    categories_series = pd.Series(cat_names)
    breakdown = categories_series.value_counts().to_dict()
    total = len(df_uploaded)
    attack_count = sum(count for cat, count in breakdown.items() if cat.lower() != "benign")
    benign_count = breakdown.get("Benign", 0)

    # Preview rows
    preview = []
    for i in range(min(50, total)):
        preview.append({
            "index": i + 1,
            "predicted_category": cat_names[i],
            "is_attack": cat_names[i].lower() != "benign",
            "IAT": float(df_uploaded["IAT"].iloc[i]) if "IAT" in df_uploaded.columns else 0.0,
            "flow_duration": float(df_uploaded["flow_duration"].iloc[i]) if "flow_duration" in df_uploaded.columns else 0.0,
            "Rate": float(df_uploaded["Rate"].iloc[i]) if "Rate" in df_uploaded.columns else 0.0,
        })

    return {
        "total_flows": total,
        "attack_count": attack_count,
        "benign_count": benign_count,
        "attack_percentage": round((attack_count / total) * 100, 2) if total else 0,
        "inference_time_ms": round((t1 - t0) * 1000, 2),
        "throughput_flows_sec": round(total / max(t1 - t0, 0.0001), 0),
        "category_breakdown": breakdown,
        "preview": preview,
    }


# Mount built frontend if available
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
