#!/usr/bin/env python3
"""
train_model.py
----------------
Assignment 4, Task 2 — ML Model Development Pipeline for Threat Detection.

Loads a labeled network-flow dataset (CICIDS2017-structured; see
generate_network_dataset.py / README.md for the data-source note), validates
it, engineers/selects features, handles severe class imbalance, trains and
tunes three candidate algorithms (Random Forest, XGBoost, and an
MLP-based neural network), cross-validates each, and persists the best
model plus its full preprocessing pipeline for reuse in evaluate_model.py.

Usage:
    python train_model.py --config config.yaml

Author: Lokesh Kumar
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                      train_test_split)
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("train_model")


# =====================================================================
# 1. Data ingestion & validation
# =====================================================================

def load_data(cfg: dict) -> pd.DataFrame:
    path = cfg["data"]["raw_path"]
    if not Path(path).exists():
        logger.error("Dataset not found at %s. Run generate_network_dataset.py first.", path)
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info("Loaded dataset: %d rows x %d columns from %s", df.shape[0], df.shape[1], path)

    label_col = cfg["data"]["label_column"]
    if label_col not in df.columns:
        logger.error("Expected label column '%s' not found in dataset.", label_col)
        sys.exit(1)

    # --- Schema validation ---
    numeric_cols = [c for c in df.columns if c != label_col]
    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        logger.warning("Non-numeric feature columns detected (will be coerced): %s", non_numeric)
        for c in non_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Data quality checks ---
    n_missing = df.isna().sum().sum()
    n_dupes = df.duplicated().sum()
    logger.info("Data quality check: %d missing cell(s), %d duplicate row(s)", n_missing, n_dupes)

    if cfg["preprocessing"]["drop_duplicates"] and n_dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped %d duplicate rows -> %d rows remain", n_dupes, len(df))

    # --- Outlier flag (informational; not dropped, since extreme values can
    #     be genuine attack signatures rather than data errors) ---
    numeric_df = df[numeric_cols]
    z = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0).replace(0, np.nan)
    extreme_count = int((z.abs() > 6).sum().sum())
    logger.info("Informational: %d cell(s) exceed |z|>6 (possible outliers, not removed)", extreme_count)

    return df


# =====================================================================
# 2. Feature engineering / preprocessing
# =====================================================================

def preprocess(df: pd.DataFrame, cfg: dict):
    label_col = cfg["data"]["label_column"]
    task_type = cfg["data"]["task_type"]

    y_raw = df[label_col].astype(str)
    if task_type == "binary":
        y_raw = y_raw.apply(lambda v: "BENIGN" if v == "BENIGN" else "ATTACK")

    X = df.drop(columns=[label_col])
    feature_names_in = list(X.columns)

    # --- Impute missing values ---
    imputer = SimpleImputer(strategy=cfg["preprocessing"]["missing_value_strategy"])
    X_imputed = imputer.fit_transform(X)

    # --- Scale ---
    scaler_choice = cfg["preprocessing"]["scaler"]
    scaler = StandardScaler() if scaler_choice == "standard" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # --- Encode labels ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    logger.info("Classes: %s", list(label_encoder.classes_))

    # --- Feature selection (mutual information) ---
    fs_cfg = cfg["preprocessing"]["feature_selection"]
    selector = None
    selected_feature_names = feature_names_in
    if fs_cfg["enabled"]:
        k = min(fs_cfg["top_k"], X_scaled.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        mask = selector.get_support()
        selected_feature_names = [f for f, keep in zip(feature_names_in, mask) if keep]
        logger.info("Feature selection: kept top %d / %d features by mutual information", k, len(feature_names_in))
    else:
        X_selected = X_scaled

    # --- Train / validation / test split (70/15/15), stratified ---
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]
    random_state = cfg["data"]["random_state"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state, stratify=y_temp
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test)
    )

    # --- Handle class imbalance (train split only) ---
    imbalance_strategy = cfg["preprocessing"]["imbalance_strategy"]
    class_weight_mode = None
    if imbalance_strategy == "smote":
        class_counts = pd.Series(y_train).value_counts()
        min_class_count = class_counts.min()
        k_neighbors = min(cfg["preprocessing"]["smote_k_neighbors"], max(min_class_count - 1, 1))
        # Cap oversampling target per minority class (rather than matching
        # the majority class exactly) to keep the resampled training set a
        # tractable size while still substantially correcting the imbalance.
        cap = min(int(class_counts.median() * 1.5), 4000)
        sampling_strategy = {
            cls: max(count, cap) for cls, count in class_counts.items() if count < cap
        }
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors,
                       sampling_strategy=sampling_strategy if sampling_strategy else "auto")
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info("Applied capped SMOTE oversampling (cap=%d/class) -> %d training rows after resampling", cap, len(X_train))
    elif imbalance_strategy == "class_weight":
        class_weight_mode = "balanced"
        logger.info("Using class_weight='balanced' instead of resampling.")

    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "selector": selector,
        "selected_feature_names": selected_feature_names,
        "class_weight_mode": class_weight_mode,
    }
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
    return splits, artifacts


# =====================================================================
# 3. Model training with hyperparameter tuning + cross-validation
# =====================================================================

def stratified_subsample(X, y, max_rows, random_state):
    """Return a class-stratified subsample capped at max_rows, used to make
    hyperparameter search tractable on limited compute; the winning
    configuration is always refit on the FULL training set afterward."""
    if len(X) <= max_rows:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_rows, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def train_models(splits: dict, artifacts: dict, cfg: dict) -> dict:
    X_train, y_train = splits["X_train"], splits["y_train"]
    cv_folds = cfg["training"]["cv_folds"]
    scoring = cfg["training"]["scoring"]
    n_jobs = cfg["training"]["n_jobs"]
    random_state = cfg["data"]["random_state"]
    class_weight_mode = artifacts["class_weight_mode"]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results = {}
    training_log_rows = []

    # --- Random Forest ---
    rf_cfg = cfg["models"]["random_forest"]
    if rf_cfg["enabled"]:
        logger.info("Training Random Forest with RandomizedSearchCV (%d-fold CV)...", cv_folds)
        t0 = time.time()
        X_search, y_search = stratified_subsample(X_train, y_train, max_rows=8000, random_state=random_state)
        rf = RandomForestClassifier(
            random_state=random_state,
            class_weight=class_weight_mode,
            n_jobs=1,
        )
        param_dist = {
            "n_estimators": rf_cfg["param_grid"]["n_estimators"],
            "max_depth": rf_cfg["param_grid"]["max_depth"],
            "min_samples_split": rf_cfg["param_grid"]["min_samples_split"],
        }
        search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=4, cv=cv,
            scoring=scoring, random_state=random_state, n_jobs=1,
        )
        search.fit(X_search, y_search)
        # Refit the winning configuration on the FULL training set for the
        # final model (search itself ran on a subsample for tractability).
        best_rf = RandomForestClassifier(
            random_state=random_state, class_weight=class_weight_mode, n_jobs=1,
            **search.best_params_,
        )
        best_rf.fit(X_train, y_train)
        elapsed = time.time() - t0
        results["random_forest"] = best_rf
        logger.info(
            "Random Forest done in %.1fs | best CV %s=%.4f (on %d-row search sample) | params=%s | refit on full %d-row training set",
            elapsed, scoring, search.best_score_, len(X_search), search.best_params_, len(X_train),
        )
        training_log_rows.append({
            "model": "random_forest", "best_cv_score": search.best_score_,
            "best_params": json.dumps(search.best_params_), "train_time_sec": round(elapsed, 2),
        })

    # --- XGBoost ---
    xgb_cfg = cfg["models"]["xgboost"]
    if xgb_cfg["enabled"]:
        logger.info("Training XGBoost with RandomizedSearchCV (%d-fold CV)...", cv_folds)
        t0 = time.time()
        X_search, y_search = stratified_subsample(X_train, y_train, max_rows=8000, random_state=random_state)
        xgb = XGBClassifier(
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=1,
            tree_method="hist",
        )
        param_dist = {
            "n_estimators": xgb_cfg["param_grid"]["n_estimators"],
            "max_depth": xgb_cfg["param_grid"]["max_depth"],
            "learning_rate": xgb_cfg["param_grid"]["learning_rate"],
            "subsample": xgb_cfg["param_grid"]["subsample"],
        }
        search = RandomizedSearchCV(
            xgb, param_distributions=param_dist, n_iter=4, cv=cv,
            scoring=scoring, random_state=random_state, n_jobs=1,
        )
        search.fit(X_search, y_search)
        best_xgb = XGBClassifier(
            random_state=random_state, eval_metric="mlogloss", n_jobs=1,
            tree_method="hist", **search.best_params_,
        )
        best_xgb.fit(X_train, y_train)
        elapsed = time.time() - t0
        results["xgboost"] = best_xgb
        logger.info(
            "XGBoost done in %.1fs | best CV %s=%.4f (on %d-row search sample) | params=%s | refit on full %d-row training set",
            elapsed, scoring, search.best_score_, len(X_search), search.best_params_, len(X_train),
        )
        training_log_rows.append({
            "model": "xgboost", "best_cv_score": search.best_score_,
            "best_params": json.dumps(search.best_params_), "train_time_sec": round(elapsed, 2),
        })

    # --- Neural Network (MLP) ---
    nn_cfg = cfg["models"]["neural_network"]
    if nn_cfg["enabled"]:
        logger.info("Training Neural Network (MLPClassifier)...")
        t0 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(nn_cfg["hidden_layer_sizes"]),
            alpha=nn_cfg["alpha"],
            max_iter=nn_cfg["max_iter"],
            random_state=random_state,
            early_stopping=True,
        )
        cv_scores = []
        X_mlp_cv, y_mlp_cv = stratified_subsample(X_train, y_train, max_rows=10000, random_state=random_state)
        for train_idx, val_idx in cv.split(X_mlp_cv, y_mlp_cv):
            mlp_fold = MLPClassifier(
                hidden_layer_sizes=tuple(nn_cfg["hidden_layer_sizes"]),
                alpha=nn_cfg["alpha"], max_iter=nn_cfg["max_iter"],
                random_state=random_state, early_stopping=True,
            )
            mlp_fold.fit(X_mlp_cv[train_idx], y_mlp_cv[train_idx])
            from sklearn.metrics import f1_score
            preds = mlp_fold.predict(X_mlp_cv[val_idx])
            cv_scores.append(f1_score(y_mlp_cv[val_idx], preds, average="macro"))
        mlp.fit(X_train, y_train)
        elapsed = time.time() - t0
        results["neural_network"] = mlp
        mean_cv = float(np.mean(cv_scores))
        logger.info("Neural Network done in %.1fs | CV %s=%.4f (mean of %d folds)", elapsed, scoring, mean_cv, cv_folds)
        training_log_rows.append({
            "model": "neural_network", "best_cv_score": mean_cv,
            "best_params": json.dumps({
                "hidden_layer_sizes": nn_cfg["hidden_layer_sizes"], "alpha": nn_cfg["alpha"],
            }),
            "train_time_sec": round(elapsed, 2),
        })

    log_df = pd.DataFrame(training_log_rows)
    return results, log_df


# =====================================================================
# 4. Cross-validated evaluation on held-out validation split
# =====================================================================

def evaluate_cv(models: dict, splits: dict) -> pd.DataFrame:
    from sklearn.metrics import f1_score, precision_score, recall_score

    X_val, y_val = splits["X_val"], splits["y_val"]
    rows = []
    for name, model in models.items():
        preds = model.predict(X_val)
        rows.append({
            "model": name,
            "val_precision_macro": round(precision_score(y_val, preds, average="macro", zero_division=0), 4),
            "val_recall_macro": round(recall_score(y_val, preds, average="macro", zero_division=0), 4),
            "val_f1_macro": round(f1_score(y_val, preds, average="macro", zero_division=0), 4),
        })
    result_df = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    logger.info("Validation comparison:\n%s", result_df.to_string(index=False))
    return result_df


# =====================================================================
# 5. Model persistence
# =====================================================================

def save_artifacts(models: dict, artifacts: dict, val_comparison: pd.DataFrame,
                    training_log: pd.DataFrame, cfg: dict) -> str:
    out_cfg = cfg["output"]
    out_dir = Path(out_cfg["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model_name = val_comparison.iloc[0]["model"]
    best_model = models[best_model_name]

    joblib.dump(best_model, out_dir / out_cfg["best_model_file"])
    logger.info("Saved best model (%s) -> %s", best_model_name, out_dir / out_cfg["best_model_file"])

    preprocessing_pipeline = {
        "imputer": artifacts["imputer"],
        "scaler": artifacts["scaler"],
        "label_encoder": artifacts["label_encoder"],
        "selector": artifacts["selector"],
    }
    joblib.dump(preprocessing_pipeline, out_dir / out_cfg["preprocessing_pipeline_file"])
    logger.info("Saved preprocessing pipeline -> %s", out_dir / out_cfg["preprocessing_pipeline_file"])

    with open(out_dir / out_cfg["feature_list_file"], "w") as f:
        f.write("\n".join(artifacts["selected_feature_names"]))
    logger.info("Saved selected feature list (%d features) -> %s",
                len(artifacts["selected_feature_names"]), out_dir / out_cfg["feature_list_file"])

    merged_log = training_log.merge(val_comparison, on="model", how="left")
    merged_log["is_best_model"] = merged_log["model"] == best_model_name
    merged_log["model_version"] = out_cfg["model_version"]
    merged_log.to_csv(out_dir / out_cfg["training_log_file"], index=False)
    logger.info("Saved training log -> %s", out_dir / out_cfg["training_log_file"])

    meta = {
        "best_model_name": best_model_name,
        "model_version": out_cfg["model_version"],
        "n_features": len(artifacts["selected_feature_names"]),
        "classes": list(artifacts["label_encoder"].classes_),
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return best_model_name


# =====================================================================
# 6. Orchestration
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ML threat detection models (Assignment 4, Task 2).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["data"]["random_state"])

    logger.info("=== Step 1/5: Data ingestion & validation ===")
    df = load_data(cfg)

    logger.info("=== Step 2/5: Preprocessing & feature engineering ===")
    splits, artifacts = preprocess(df, cfg)

    logger.info("=== Step 3/5: Model training & hyperparameter tuning ===")
    models, training_log = train_models(splits, artifacts, cfg)

    logger.info("=== Step 4/5: Cross-validated evaluation (validation split) ===")
    val_comparison = evaluate_cv(models, splits)

    logger.info("=== Step 5/5: Persisting artifacts ===")
    best_model_name = save_artifacts(models, artifacts, val_comparison, training_log, cfg)

    # Persist the held-out test split too, so evaluate_model.py can run
    # completely independently against genuinely unseen data.
    out_dir = Path(cfg["output"]["model_dir"])
    np.savez(out_dir / "test_split.npz", X_test=splits["X_test"], y_test=splits["y_test"])
    logger.info("Saved held-out test split -> %s", out_dir / "test_split.npz")

    logger.info("Training pipeline complete. Best model: %s", best_model_name)


if __name__ == "__main__":
    main()
