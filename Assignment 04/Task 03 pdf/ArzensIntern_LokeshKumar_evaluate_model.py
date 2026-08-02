#!/usr/bin/env python3
"""
evaluate_model.py
--------------------
Assignment 4, Task 3 — Model Evaluation & Interpretability.

Loads the trained model(s) from Task 2, evaluates them on the untouched
held-out test split using both standard and security-specific metrics,
performs threshold analysis for the attack/benign alerting decision, runs
SHAP-based global and local interpretability analysis on the best model,
performs error analysis (false positives / false negatives), checks
production readiness (latency, size, robustness), and writes a single
professional HTML evaluation report plus a model card.

Usage:
    python evaluate_model.py --config config.yaml

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                              confusion_matrix, f1_score, precision_recall_curve,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("evaluate_model")

OUT_DIR = Path("shap_analysis")
(OUT_DIR / "dependence_plots").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "force_plots_for_examples").mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. Load artifacts
# =====================================================================

def load_artifacts(cfg: dict):
    model_dir = Path(cfg["output"]["model_dir"])
    model = joblib.load(model_dir / cfg["output"]["best_model_file"])
    prep = joblib.load(model_dir / cfg["output"]["preprocessing_pipeline_file"])
    with open(model_dir / cfg["output"]["feature_list_file"]) as f:
        feature_names = [line.strip() for line in f if line.strip()]
    with open(model_dir / "model_metadata.json") as f:
        meta = json.load(f)
    test_npz = np.load(model_dir / "test_split.npz")
    X_test, y_test = test_npz["X_test"], test_npz["y_test"]
    logger.info("Loaded best model '%s', %d test rows, %d features",
                meta["best_model_name"], len(X_test), len(feature_names))
    return model, prep, feature_names, meta, X_test, y_test


# =====================================================================
# 2. Performance metrics (standard + security-specific + per-class)
# =====================================================================

def binary_collapse(y, label_encoder, benign_label="BENIGN"):
    """Collapse multi-class labels into a binary Attack(1)/Benign(0) view,
    used for the security-alerting-style metrics (FPR/FNR, threshold
    analysis, ROC/PR curves) that the assignment frames in binary terms."""
    benign_idx = list(label_encoder.classes_).index(benign_label)
    return (y != benign_idx).astype(int), benign_idx


def compute_performance_metrics(model, X_test, y_test, label_encoder) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(label_encoder.classes_)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    y_test_bin = pd.get_dummies(y_test).reindex(columns=range(len(classes)), fill_value=0).values
    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc_macro = float("nan")
    pr_auc_macro = np.mean([
        average_precision_score(y_test_bin[:, c], y_proba[:, c]) for c in range(len(classes))
    ])

    # --- Binary (attack vs benign) security metrics ---
    y_true_bin, benign_idx = binary_collapse(y_test, label_encoder)
    y_pred_bin, _ = binary_collapse(y_pred, label_encoder)
    y_attack_proba = 1 - y_proba[:, benign_idx]

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    roc_auc_binary = roc_auc_score(y_true_bin, y_attack_proba)
    pr_auc_binary = average_precision_score(y_true_bin, y_attack_proba)

    # --- Cost-sensitive weighted F1 (FN weighted more heavily than FP,
    #     reflecting that missing an attack is costlier than a false alarm) ---
    cost_fp, cost_fn = 1.0, 5.0
    weighted_cost = cost_fp * fp + cost_fn * fn
    max_cost = cost_fp * (fp + tn) + cost_fn * (fn + tp)
    cost_sensitive_score = 1 - (weighted_cost / max_cost)

    # --- Per-class breakdown ---
    per_class_rows = []
    for c_idx, c_name in enumerate(classes):
        mask_true = y_test == c_idx
        mask_pred = y_pred == c_idx
        p = precision_score(y_test == c_idx, y_pred == c_idx, zero_division=0)
        r = recall_score(y_test == c_idx, y_pred == c_idx, zero_division=0)
        f1 = f1_score(y_test == c_idx, y_pred == c_idx, zero_division=0)
        support = int(mask_true.sum())
        per_class_rows.append({"class": c_name, "precision": round(p, 4),
                                "recall": round(r, 4), "f1": round(f1, 4), "support": support})
    per_class_df = pd.DataFrame(per_class_rows).sort_values("recall")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "roc_auc_macro": round(float(roc_auc_macro), 4),
        "pr_auc_macro": round(float(pr_auc_macro), 4),
        "roc_auc_binary_attack_vs_benign": round(float(roc_auc_binary), 4),
        "pr_auc_binary_attack_vs_benign": round(float(pr_auc_binary), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
        "cost_sensitive_score": round(float(cost_sensitive_score), 4),
        "confusion_binary": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    logger.info("Test-set performance: %s", json.dumps(metrics, indent=2))
    return metrics, per_class_df, y_pred, y_proba, y_true_bin, y_attack_proba


# =====================================================================
# 3. Model comparison (retrain all 3 candidates, evaluate on test set)
# =====================================================================

def model_comparison_table(cfg: dict, label_encoder) -> pd.DataFrame:
    """Re-runs the Task 2 pipeline (fixed random_state -> reproducible) to
    obtain all three trained candidate models, then evaluates each on the
    same held-out test split for a fair side-by-side comparison."""
    from train_model import load_data, preprocess, train_models

    logger.info("Re-running training pipeline to obtain all 3 candidate models for comparison...")
    df = load_data(cfg)
    splits, artifacts = preprocess(df, cfg)
    models, training_log = train_models(splits, artifacts, cfg)

    X_test, y_test = splits["X_test"], splits["y_test"]
    classes = list(label_encoder.classes_)
    y_test_bin = pd.get_dummies(y_test).reindex(columns=range(len(classes)), fill_value=0).values

    rows = []
    for name, model in models.items():
        t0 = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        inf_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        try:
            roc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            roc = float("nan")
        pr_auc = np.mean([average_precision_score(y_test_bin[:, c], y_proba[:, c]) for c in range(len(classes))])
        train_time = training_log.loc[training_log["model"] == name, "train_time_sec"].values[0]

        rows.append({
            "Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1": round(f1, 4), "ROC-AUC": round(roc, 4),
            "PR-AUC": round(pr_auc, 4), "Training Time (s)": round(float(train_time), 1),
            "Test Inference (ms/1000 rows)": round(inf_time * 1000, 2),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


# =====================================================================
# 4. Threshold analysis (binary attack-alerting view)
# =====================================================================

def threshold_analysis(y_true_bin, y_attack_proba) -> dict:
    thresholds_to_check = [0.3, 0.5, 0.7]
    rows = []
    for t in thresholds_to_check:
        y_pred_t = (y_attack_proba >= t).astype(int)
        p = precision_score(y_true_bin, y_pred_t, zero_division=0)
        r = recall_score(y_true_bin, y_pred_t, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_t, zero_division=0)
        rows.append({"threshold": t, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
    threshold_table = pd.DataFrame(rows)

    # Search a finer grid for the F1-optimal threshold as the production recommendation.
    fine_grid = np.linspace(0.05, 0.95, 37)
    best_t, best_f1 = 0.5, -1
    for t in fine_grid:
        y_pred_t = (y_attack_proba >= t).astype(int)
        f1 = f1_score(y_true_bin, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # PR-vs-threshold plot
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_bin, y_attack_proba)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pr_thresholds, precisions[:-1], label="Precision", color="#1F4E79")
    ax.plot(pr_thresholds, recalls[:-1], label="Recall", color="#C0392B")
    ax.axvline(best_t, color="gray", linestyle="--", label=f"Recommended threshold ({best_t:.2f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision-Recall vs. Threshold (Attack vs. Benign)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "precision_recall_vs_threshold.png", dpi=150)
    plt.close(fig)

    # ROC and PR curves
    fpr_arr, tpr_arr, _ = roc_curve(y_true_bin, y_attack_proba)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr_arr, tpr_arr, color="#1F4E79")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve (AUC={roc_auc_score(y_true_bin, y_attack_proba):.3f})")

    axes[1].plot(recalls, precisions, color="#C0392B")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve (AP={average_precision_score(y_true_bin, y_attack_proba):.3f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    return {
        "threshold_table": threshold_table,
        "recommended_threshold": round(float(best_t), 2),
        "recommended_threshold_f1": round(float(best_f1), 4),
    }


# =====================================================================
# 5. SHAP interpretability (global + local)
# =====================================================================

def shap_analysis(model, X_test, y_test, y_pred, feature_names, label_encoder,
                   benign_idx, sample_size=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    logger.info("Computing SHAP values on a %d-row sample...", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)  # shape (n, n_features, n_classes)

    classes = list(label_encoder.classes_)

    # --- Global: mean |SHAP| across all non-benign classes, aggregated,
    #     to answer "what drives an ATTACK prediction overall" ---
    non_benign_idx = [i for i in range(len(classes)) if i != benign_idx]
    attack_shap = np.mean(np.abs(shap_values[:, :, non_benign_idx]), axis=2)  # (n, n_features)
    mean_abs_shap = attack_shap.mean(axis=0)
    importance_order = np.argsort(mean_abs_shap)[::-1]
    top10_features = [feature_names[i] for i in importance_order[:10]]

    fig, ax = plt.subplots(figsize=(9, 7))
    top15_idx = importance_order[:15]
    ax.barh([feature_names[i] for i in top15_idx][::-1], mean_abs_shap[top15_idx][::-1], color="#1F4E79")
    ax.set_xlabel("Mean |SHAP value| (avg. across attack classes)")
    ax.set_title("SHAP Global Feature Importance — Top 15 (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_plot.png", dpi=150)
    plt.close(fig)

    # --- Dependence plots for the top 3 features ---
    for feat_name in top10_features[:3]:
        feat_idx = feature_names.index(feat_name)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X_sample[:, feat_idx], attack_shap[:, feat_idx], s=10, alpha=0.5, color="#1F4E79")
        ax.set_xlabel(f"{feat_name} (scaled value)")
        ax.set_ylabel("SHAP value (impact toward ATTACK classes)")
        ax.set_title(f"SHAP Dependence Plot — {feat_name}")
        plt.tight_layout()
        safe_name = feat_name.replace("/", "_").replace(" ", "_")
        plt.savefig(OUT_DIR / "dependence_plots" / f"{safe_name}.png", dpi=150)
        plt.close(fig)

    # --- Local explanations: 2 TP, 2 FP, 1 FN (binary attack-detection framing) ---
    y_true_bin_full, _ = binary_collapse(y_test, label_encoder)
    y_pred_bin_full, _ = binary_collapse(y_pred, label_encoder)

    tp_idx = np.where((y_true_bin_full == 1) & (y_pred_bin_full == 1))[0]
    fp_idx = np.where((y_true_bin_full == 0) & (y_pred_bin_full == 1))[0]
    fn_idx = np.where((y_true_bin_full == 1) & (y_pred_bin_full == 0))[0]

    examples = (
        [("TP", i) for i in tp_idx[:2]] +
        [("FP", i) for i in fp_idx[:2]] +
        [("FN", i) for i in fn_idx[:1]]
    )

    local_explanations = []
    for kind, idx in examples:
        pred_class_idx = int(y_pred[idx])
        true_class_idx = int(y_test[idx])
        x_row = X_test[idx:idx + 1]
        sv_row = explainer.shap_values(x_row)  # (1, n_features, n_classes)
        contrib = sv_row[0, :, pred_class_idx]
        top3_idx = np.argsort(np.abs(contrib))[::-1][:3]

        explanation_parts = []
        for fi in top3_idx:
            direction = "increased" if contrib[fi] > 0 else "decreased"
            explanation_parts.append(
                f"{feature_names[fi]} {direction} the likelihood of the predicted class "
                f"(SHAP={contrib[fi]:+.3f})"
            )
        plain_english = (
            f"Predicted: {classes[pred_class_idx]} | Actual: {classes[true_class_idx]}. "
            f"Top contributing factors: " + "; ".join(explanation_parts) + "."
        )
        local_explanations.append({
            "type": kind, "index": int(idx),
            "predicted": classes[pred_class_idx], "actual": classes[true_class_idx],
            "explanation": plain_english,
        })

        # Simple horizontal bar "force-plot-style" visualization
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#C0392B" if v > 0 else "#1F4E79" for v in contrib[top3_idx][::-1]]
        ax.barh([feature_names[i] for i in top3_idx][::-1], contrib[top3_idx][::-1], color=colors)
        ax.set_xlabel(f"SHAP contribution to predicted class '{classes[pred_class_idx]}'")
        ax.set_title(f"{kind} example (row {idx}): predicted={classes[pred_class_idx]}, actual={classes[true_class_idx]}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "force_plots_for_examples" / f"{kind}_{idx}.png", dpi=150)
        plt.close(fig)

    with open(OUT_DIR / "local_explanations.json", "w") as f:
        json.dump(local_explanations, f, indent=2)

    return top10_features, local_explanations


# =====================================================================
# 6. Error analysis
# =====================================================================

def error_analysis(y_test, y_pred, X_test, feature_names, label_encoder):
    classes = list(label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (multi-class)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    benign_idx = classes.index("BENIGN")
    y_true_bin, _ = binary_collapse(y_test, label_encoder)
    y_pred_bin, _ = binary_collapse(y_pred, label_encoder)

    fp_mask = (y_true_bin == 0) & (y_pred_bin == 1)
    fn_mask = (y_true_bin == 1) & (y_pred_bin == 0)

    fp_summary = pd.DataFrame(X_test[fp_mask], columns=feature_names).mean().sort_values(ascending=False).head(5) \
        if fp_mask.sum() else pd.Series(dtype=float)
    fn_by_class = pd.Series(y_test[fn_mask]).map(lambda i: classes[i]).value_counts() if fn_mask.sum() else pd.Series(dtype=int)

    return {
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        "fp_top_scaled_features": fp_summary.round(3).to_dict(),
        "fn_missed_by_attack_type": fn_by_class.to_dict(),
    }


# =====================================================================
# 7. Production readiness check
# =====================================================================

def production_readiness(model, X_test, model_path: Path):
    n_trials = min(500, len(X_test))
    sample = X_test[:n_trials]

    t0 = time.time()
    for i in range(n_trials):
        model.predict(sample[i:i + 1])
    total_time = time.time() - t0
    latency_ms = (total_time / n_trials) * 1000

    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    # Robustness / drift check: add small Gaussian noise (already-scaled
    # features) to simulate mild sensor/data drift and re-measure accuracy.
    rng = np.random.default_rng(42)
    X_noisy = X_test + rng.normal(0, 0.05, size=X_test.shape)
    y_pred_orig_acc = None  # computed by caller using clean predictions

    result = {
        "latency_ms_per_sample": round(latency_ms, 3),
        "latency_ok": latency_ms < 10,
        "model_size_mb": round(model_size_mb, 2),
        "model_size_ok": model_size_mb < 100,
        "X_noisy": X_noisy,
    }
    return result


# =====================================================================
# 8. HTML report generation
# =====================================================================

def render_html_report(context: dict) -> str:
    m = context["metrics"]
    per_class_df = context["per_class_df"]
    comparison_df = context["comparison_df"]
    threshold = context["threshold"]
    top10_features = context["top10_features"]
    local_explanations = context["local_explanations"]
    error = context["error"]
    prod = context["prod"]
    meta = context["meta"]

    per_class_rows = "".join(
        f"<tr><td>{r['class']}</td><td>{r['precision']}</td><td>{r['recall']}</td>"
        f"<td>{r['f1']}</td><td>{r['support']}</td></tr>"
        for r in per_class_df.to_dict(orient="records")
    )
    comparison_rows = "".join(
        f"<tr><td>{r['Model']}</td><td>{r['Accuracy']}</td><td>{r['Precision']}</td>"
        f"<td>{r['Recall']}</td><td>{r['F1']}</td><td>{r['ROC-AUC']}</td><td>{r['PR-AUC']}</td>"
        f"<td>{r['Training Time (s)']}</td></tr>"
        for r in comparison_df.to_dict(orient="records")
    )
    threshold_rows = "".join(
        f"<tr><td>{r['threshold']}</td><td>{r['precision']}</td><td>{r['recall']}</td><td>{r['f1']}</td></tr>"
        for r in threshold["threshold_table"].to_dict(orient="records")
    )
    local_rows = "".join(
        f"<tr><td>{e['type']}</td><td>{e['predicted']}</td><td>{e['actual']}</td><td>{e['explanation']}</td></tr>"
        for e in local_explanations
    )
    fn_rows = "".join(f"<li>{k}: {v} missed</li>" for k, v in error["fn_missed_by_attack_type"].items()) or "<li>None</li>"
    fp_feat_rows = "".join(f"<li>{k}: {v}</li>" for k, v in error["fp_top_scaled_features"].items()) or "<li>None</li>"

    status_color = "#1a7f37" if prod["latency_ok"] and prod["model_size_ok"] else "#c0392b"
    status_text = "APPROVED" if prod["latency_ok"] and prod["model_size_ok"] else "REVIEW NEEDED"

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Model Evaluation Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1c1c1e; background: #fafafa; }}
h1 {{ color: #1F4E79; margin-bottom: 4px; }}
h2 {{ color: #1F4E79; border-bottom: 2px solid #e8eef5; padding-bottom: 6px; }}
.subtitle {{ color: #555; margin-top: 0; }}
section {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px 24px; margin: 20px 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; font-size: 13.5px; }}
th {{ background: #1F4E79; color: white; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px; }}
.metric-box {{ background: #E8EEF5; border-radius: 6px; padding: 12px; text-align: center; }}
.metric-box .val {{ font-size: 22px; font-weight: bold; color: #1F4E79; }}
.metric-box .lbl {{ font-size: 12px; color: #555; }}
img {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; margin-top: 8px; }}
.status-badge {{ display: inline-block; padding: 6px 16px; border-radius: 6px; color: white; font-weight: bold; background: {status_color}; }}
code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}
</style></head><body>

<h1>Model Evaluation Report</h1>
<p class="subtitle">Dataset: network_traffic_dataset.csv (CICIDS2017-structured) &nbsp;|&nbsp; Model: {meta['best_model_name']} {meta['model_version']}</p>

<section>
<h2>1. Performance Metrics (Test Set, n={context['n_test']})</h2>
<div class="metric-grid">
  <div class="metric-box"><div class="val">{m['accuracy']*100:.1f}%</div><div class="lbl">Accuracy</div></div>
  <div class="metric-box"><div class="val">{m['precision_macro']*100:.1f}%</div><div class="lbl">Precision (macro)</div></div>
  <div class="metric-box"><div class="val">{m['recall_macro']*100:.1f}%</div><div class="lbl">Recall (macro)</div></div>
  <div class="metric-box"><div class="val">{m['f1_macro']:.3f}</div><div class="lbl">F1 (macro)</div></div>
  <div class="metric-box"><div class="val">{m['roc_auc_macro']:.3f}</div><div class="lbl">ROC-AUC (macro OVR)</div></div>
  <div class="metric-box"><div class="val">{m['pr_auc_macro']:.3f}</div><div class="lbl">PR-AUC (macro OVR)</div></div>
  <div class="metric-box"><div class="val">{m['false_positive_rate']*100:.1f}%</div><div class="lbl">False Positive Rate</div></div>
  <div class="metric-box"><div class="val">{m['false_negative_rate']*100:.1f}%</div><div class="lbl">False Negative Rate</div></div>
</div>
<p style="margin-top:14px;">Binary (Attack vs. Benign) alerting view: ROC-AUC = {m['roc_auc_binary_attack_vs_benign']}, PR-AUC = {m['pr_auc_binary_attack_vs_benign']}, Cost-sensitive score (FN weighted 5x FP) = {m['cost_sensitive_score']}.</p>
<p>Confusion (binary): TN={m['confusion_binary']['tn']}, FP={m['confusion_binary']['fp']}, FN={m['confusion_binary']['fn']}, TP={m['confusion_binary']['tp']}</p>
</section>

<section>
<h2>2. Per-Class Metrics (sorted by recall — hardest-to-detect classes first)</h2>
<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>{per_class_rows}</table>
</section>

<section>
<h2>3. Model Comparison</h2>
<table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>Training Time (s)</th></tr>{comparison_rows}</table>
<p style="margin-top:10px;">Best model selected: <b>{meta['best_model_name']}</b> — chosen for the strongest macro-F1 on the held-out validation split (Task 2), balancing recall on rare attack classes against precision for alert volume.</p>
</section>

<section>
<h2>4. Threshold Analysis (Attack vs. Benign alerting)</h2>
<table><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{threshold_rows}</table>
<p><b>Recommended production threshold: {threshold['recommended_threshold']}</b> (F1={threshold['recommended_threshold_f1']}) — chosen to maximize F1 on the binary attack-detection view, balancing missed-threat risk against alert volume.</p>
<img src="shap_analysis/precision_recall_vs_threshold.png" alt="PR vs threshold">
<img src="shap_analysis/roc_pr_curves.png" alt="ROC and PR curves">
</section>

<section>
<h2>5. Model Interpretability (SHAP)</h2>
<h3>Global: Top 10 most important features</h3>
<ol>{"".join(f"<li>{f}</li>" for f in top10_features)}</ol>
<img src="shap_analysis/summary_plot.png" alt="SHAP summary plot">
<p>Dependence plots for the top 3 features are saved in <code>shap_analysis/dependence_plots/</code>.</p>

<h3>Local: Example predictions (2 TP, 2 FP, 1 FN)</h3>
<table><tr><th>Type</th><th>Predicted</th><th>Actual</th><th>Explanation</th></tr>{local_rows}</table>
<p>Corresponding force-plot-style visualizations are saved in <code>shap_analysis/force_plots_for_examples/</code>.</p>
</section>

<section>
<h2>6. Error Analysis</h2>
<img src="shap_analysis/confusion_matrix.png" alt="Confusion matrix">
<p><b>False positives ({error['n_false_positives']} rows):</b> benign flows most resembling attacks tend to show elevated values (relative to their scaled mean) in:</p>
<ul>{fp_feat_rows}</ul>
<p><b>False negatives ({error['n_false_negatives']} rows) — attacks that evaded detection, by type:</b></p>
<ul>{fn_rows}</ul>
<p><b>Recommendations for model improvement:</b></p>
<ul>
<li>Collect more labeled examples of the hardest-to-detect attack classes identified in Section 2 (typically the rarest classes, e.g. Infiltration/Botnet) to reduce false negatives.</li>
<li>Consider a two-stage detector: a high-recall first-stage filter (low threshold) followed by a higher-precision second-stage classifier for alert triage, reducing analyst-facing false positives without lowering overall recall.</li>
<li>Periodically retrain on recent traffic to correct for the false-positive-prone feature patterns identified above, which may reflect legitimate but evolving benign usage.</li>
</ul>
</section>

<section>
<h2>7. Production Readiness Check</h2>
<p>Status: <span class="status-badge">{status_text}</span></p>
<ul>
<li>Inference latency: {prod['latency_ms_per_sample']} ms/sample ({'meets' if prod['latency_ok'] else 'does NOT meet'} the &lt;10ms real-time requirement)</li>
<li>Model size on disk: {prod['model_size_mb']} MB ({'meets' if prod['model_size_ok'] else 'does NOT meet'} the &lt;100MB preferred limit)</li>
<li>Robustness (drift check): accuracy on Gaussian-perturbed test features (σ=0.05 on scaled inputs) = {prod['noisy_accuracy']*100:.1f}% vs. {m['accuracy']*100:.1f}% on clean test data (Δ={prod['accuracy_drop']*100:.1f} pts)</li>
</ul>
</section>

<section>
<h2>Recommendations</h2>
<ol>
<li>Deploy with threshold={threshold['recommended_threshold']} for the production alerting pipeline.</li>
<li>Monitor the top SHAP features (Section 5) for drift using the quality/drift validator built in the earlier Track 09 submission.</li>
<li>Retrain periodically as new attack samples and evolving benign traffic patterns become available.</li>
</ol>
</section>

</body></html>
"""


# =====================================================================
# 9. Model card
# =====================================================================

def write_model_card(meta, metrics, threshold, path="model_card.md"):
    content = f"""# Model Card — Network Intrusion Detection Model

## Model Details
- **Model type:** {meta['best_model_name']} (see `model_artifacts/model_metadata.json`)
- **Version:** {meta['model_version']}
- **Classes:** {', '.join(meta['classes'])}
- **Number of input features:** {meta['n_features']} (selected via mutual information from 49 raw CICIDS2017-style flow features)
- **Trained by:** Lokesh Kumar, THE ARZENS Engineering Internship — AI, Automation & Security Engineering Track (Advanced)

## Intended Use
This model is a **baseline network intrusion detection classifier**, intended to flag network flows as benign or as one of six attack categories (DoS, PortScan, BruteForce, WebAttack, Infiltration, Botnet) for downstream SOC alerting and triage. It is intended to run on flow-level features extracted from network traffic (e.g. via a NetFlow/CICFlowMeter-style exporter) and to feed its predictions into an alerting/dashboard system, not to autonomously block traffic without human review.

## Training Data
Trained on a CICIDS2017-structured dataset (see `README.md` for the full data-sourcing note: a synthetic, schema-matched dataset was used in place of the original multi-gigabyte CICIDS2017 files due to sandbox network constraints). 55,000+ flow records, ~78-80% benign, with the remaining ~20-22% distributed across six attack categories of varying rarity.

## Performance (held-out test set)
- Accuracy: {metrics['accuracy']*100:.1f}%
- Precision (macro): {metrics['precision_macro']*100:.1f}%
- Recall (macro): {metrics['recall_macro']*100:.1f}%
- F1 (macro): {metrics['f1_macro']:.3f}
- ROC-AUC (macro, one-vs-rest): {metrics['roc_auc_macro']:.3f}
- PR-AUC (macro, one-vs-rest): {metrics['pr_auc_macro']:.3f}
- False Positive Rate (binary attack-alerting view): {metrics['false_positive_rate']*100:.1f}%
- False Negative Rate (binary attack-alerting view): {metrics['false_negative_rate']*100:.1f}%
- Recommended production decision threshold: {threshold['recommended_threshold']}

See `evaluation_report.html` for the full per-class breakdown, model comparison, threshold analysis, and SHAP interpretability results.

## Limitations
- **Synthetic training data:** while structured to match CICIDS2017's schema and class proportions, this model was trained on synthetically generated flow data rather than captured real-world traffic, and should be retrained on real CICIDS2017/UNSW-NB15 data (or an organization's own labeled traffic) before any production use.
- **Rare-class performance:** the rarest attack categories (typically Infiltration and Botnet) have the fewest training examples and are the hardest classes to detect reliably; see Section 2 of the evaluation report for exact per-class recall.
- **Concept drift:** network traffic patterns and attack techniques evolve; this model reflects only the traffic patterns present in its training data and requires periodic retraining and drift monitoring (see the quality/drift validator from the earlier Track 09 submission) to remain effective.
- **Feature dependency:** the model requires the same 30 selected flow-level features, computed consistently with the training pipeline's preprocessing (imputation, scaling, feature selection); it cannot be applied directly to raw packet captures or differently-structured flow exports without running them through the same `preprocessing_pipeline.pkl`.

## Ethical Considerations
- **False positives have a human cost:** flagging legitimate user traffic as malicious can disrupt business operations if acted on automatically; this model's predictions should inform, not replace, analyst judgment, especially near the recommended decision threshold.
- **False negatives have a security cost:** a missed attack can lead to real harm (data breach, service disruption); the cost-sensitive evaluation in this report explicitly weights false negatives more heavily than false positives to reflect this asymmetry, but the appropriate weighting is an organizational risk decision, not a purely technical one.
- **No personally identifiable information:** the feature set used (flow durations, packet/byte statistics, protocol flags) does not include usernames, payload content, or other directly identifying information, though source/destination IPs upstream of this feature set can be quasi-identifying and should be handled per the privacy controls (pseudonymization, k-anonymity) documented in the earlier Track 09 feature-engineering submission if this model is deployed alongside that pipeline.
- **Dual-use awareness:** like any intrusion detection model, understanding what patterns it does and does not catch (see Section 6, error analysis) could theoretically help an adversary evade detection; access to this model card and the full evaluation report should be limited to the security team responsible for deploying and maintaining the model.
"""
    with open(path, "w") as f:
        f.write(content)
    logger.info("Wrote model card -> %s", path)


# =====================================================================
# 10. Orchestration
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate and interpret the trained threat detection model (Assignment 4, Task 3).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--html-report", default="evaluation_report.html")
    parser.add_argument("--model-card", default="model_card.md")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info("=== Step 1/7: Loading artifacts ===")
    model, prep, feature_names, meta, X_test, y_test = load_artifacts(cfg)
    label_encoder = prep["label_encoder"]

    logger.info("=== Step 2/7: Performance metrics ===")
    metrics, per_class_df, y_pred, y_proba, y_true_bin, y_attack_proba = compute_performance_metrics(
        model, X_test, y_test, label_encoder
    )

    logger.info("=== Step 3/7: Model comparison (retraining all 3 candidates) ===")
    comparison_df = model_comparison_table(cfg, label_encoder)
    logger.info("Comparison:\n%s", comparison_df.to_string(index=False))

    logger.info("=== Step 4/7: Threshold analysis ===")
    threshold = threshold_analysis(y_true_bin, y_attack_proba)

    logger.info("=== Step 5/7: SHAP interpretability ===")
    benign_idx = list(label_encoder.classes_).index("BENIGN")
    top10_features, local_explanations = shap_analysis(
        model, X_test, y_test, y_pred, feature_names, label_encoder, benign_idx
    )

    logger.info("=== Step 6/7: Error analysis ===")
    error = error_analysis(y_test, y_pred, X_test, feature_names, label_encoder)

    logger.info("=== Step 7/7: Production readiness ===")
    model_dir = Path(cfg["output"]["model_dir"])
    prod = production_readiness(model, X_test, model_dir / cfg["output"]["best_model_file"])
    noisy_pred = model.predict(prod.pop("X_noisy"))
    noisy_accuracy = accuracy_score(y_test, noisy_pred)
    prod["noisy_accuracy"] = round(float(noisy_accuracy), 4)
    prod["accuracy_drop"] = round(float(metrics["accuracy"] - noisy_accuracy), 4)

    context = {
        "metrics": metrics, "per_class_df": per_class_df, "comparison_df": comparison_df,
        "threshold": threshold, "top10_features": top10_features,
        "local_explanations": local_explanations, "error": error, "prod": prod,
        "meta": meta, "n_test": len(X_test),
    }
    html = render_html_report(context)
    with open(args.html_report, "w") as f:
        f.write(html)
    logger.info("Wrote evaluation report -> %s", args.html_report)

    write_model_card(meta, metrics, threshold, args.model_card)

    comparison_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    per_class_df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)
    with open(OUT_DIR / "full_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "threshold": {k: v for k, v in threshold.items() if k != "threshold_table"},
                    "production_readiness": prod, "error_analysis": error}, f, indent=2)

    logger.info("Evaluation complete. Report: %s | Model card: %s | SHAP artifacts: %s/", args.html_report, args.model_card, OUT_DIR)


if __name__ == "__main__":
    main()
