"""
Evaluate the fine-tuned cardiac remodeling model on the test set.

Generates:
    - ROC curve with AUC
    - Confusion matrix (heatmap)
    - Precision-Recall curve
    - Training history plots (loss & accuracy)
    - Classification report (sensitivity, specificity, PPV, NPV, F1)
    - Summary text file with all metrics

Usage:
    python evaluate_model.py \
        --model finetune_output/best_model.keras \
        --hdf5 prepared_data/ecg_tracings.hdf5 \
        --labels prepared_data/labels.csv \
        --split_info prepared_data/split_info.csv \
        --training_log finetune_output/finetune_training.log \
        --output_dir finetune_output
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
from tensorflow import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)


def load_test_data(hdf5_path, labels_path, split_info_path):
    """Load test set ECGs and labels based on split_info."""
    split_df = pd.read_csv(split_info_path)
    labels_df = pd.read_csv(labels_path)

    # Test set rows in the HDF5 (ordered: train, val, test)
    n_train = len(split_df[split_df['split'] == 'train'])
    n_val = len(split_df[split_df['split'] == 'val'])
    n_test = len(split_df[split_df['split'] == 'test'])
    test_start = n_train + n_val
    test_end = test_start + n_test

    with h5py.File(hdf5_path, 'r') as f:
        x_test = np.array(f['tracings'][test_start:test_end])

    y_test = labels_df['label'].values[test_start:test_end]

    print(f"Test set: {n_test} ECGs (rows {test_start}-{test_end-1})")
    print(f"  Positive: {y_test.sum()}, Negative: {n_test - y_test.sum()}")

    return x_test, y_test, split_df[split_df['split'] == 'test']


def plot_roc_curve(y_true, y_prob, output_dir):
    """Plot ROC curve and save."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.plot(fpr[best_idx], tpr[best_idx], 'ro', markersize=10,
            label=f'Optimal threshold = {best_threshold:.3f}')
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=12)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
    ax.set_title('ROC Curve — Cardiac Remodeling Detection', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close(fig)

    print(f"  AUC: {roc_auc:.3f}")
    print(f"  Optimal threshold (Youden): {best_threshold:.3f}")

    return roc_auc, best_threshold


def plot_precision_recall(y_true, y_prob, output_dir):
    """Plot Precision-Recall curve and save."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#2563eb', lw=2,
            label=f'PR curve (AP = {ap:.3f})')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', lw=1, linestyle='--',
               label=f'Baseline (prevalence = {baseline:.2f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.set_title('Precision-Recall Curve — Cardiac Remodeling', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=150)
    plt.close(fig)

    print(f"  Average Precision: {ap:.3f}")
    return ap


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix heatmap and save."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    labels = ['No Remodeling', 'Remodeling']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)

    return tn, fp, fn, tp


def plot_training_history(log_path, output_dir):
    """Plot training history (loss and accuracy) from CSV log."""
    if not os.path.exists(log_path):
        print(f"  Training log not found: {log_path}")
        return

    log = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(log['epoch'], log['loss'], label='Train loss', lw=2)
    axes[0].plot(log['epoch'], log['val_loss'], label='Val loss', lw=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(log['epoch'], log['accuracy'], label='Train accuracy', lw=2)
    axes[1].plot(log['epoch'], log['val_accuracy'], label='Val accuracy', lw=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close(fig)

    print(f"  Training history plot saved")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned cardiac remodeling model.')
    parser.add_argument('--model', type=str,
                        default='finetune_output/best_model.keras',
                        help='Path to fine-tuned model')
    parser.add_argument('--hdf5', type=str,
                        default='prepared_data/ecg_tracings.hdf5',
                        help='Path to HDF5 with ECG tracings')
    parser.add_argument('--labels', type=str,
                        default='prepared_data/labels.csv',
                        help='Path to labels CSV')
    parser.add_argument('--split_info', type=str,
                        default='prepared_data/split_info.csv',
                        help='Path to split info CSV')
    parser.add_argument('--training_log', type=str,
                        default='finetune_output/finetune_training.log',
                        help='Path to training log CSV')
    parser.add_argument('--output_dir', type=str,
                        default='finetune_output',
                        help='Output directory for plots and metrics')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: optimal Youden)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("=" * 60)
    print("Loading model")
    print("=" * 60)
    model = keras.models.load_model(args.model, compile=False)
    print(f"Model loaded: {args.model}")

    # Load test data
    print("\n" + "=" * 60)
    print("Loading test data")
    print("=" * 60)
    x_test, y_test, test_info = load_test_data(
        args.hdf5, args.labels, args.split_info)

    # Predict
    print("\n" + "=" * 60)
    print("Running predictions")
    print("=" * 60)
    y_prob = model.predict(x_test, batch_size=32).flatten()
    print(f"  Predictions range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    # ROC curve
    print("\n" + "=" * 60)
    print("ROC Curve")
    print("=" * 60)
    roc_auc, optimal_threshold = plot_roc_curve(y_test, y_prob, args.output_dir)

    # Use optimal threshold or user-specified
    threshold = args.threshold if args.threshold is not None else optimal_threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Precision-Recall curve
    print("\n" + "=" * 60)
    print("Precision-Recall Curve")
    print("=" * 60)
    ap = plot_precision_recall(y_test, y_prob, args.output_dir)

    # Confusion matrix
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)
    tn, fp, fn, tp = plot_confusion_matrix(y_test, y_pred, args.output_dir)

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(y_test, y_pred)

    # Training history
    print("\n" + "=" * 60)
    print("Training History")
    print("=" * 60)
    plot_training_history(args.training_log, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Threshold:     {threshold:.3f}")
    print(f"  AUC:           {roc_auc:.3f}")
    print(f"  Accuracy:      {accuracy:.3f}")
    print(f"  Sensitivity:   {sensitivity:.3f}  (recall, TPR)")
    print(f"  Specificity:   {specificity:.3f}  (TNR)")
    print(f"  PPV:           {ppv:.3f}  (precision)")
    print(f"  NPV:           {npv:.3f}")
    print(f"  F1-score:      {f1:.3f}")
    print(f"  Avg Precision:  {ap:.3f}")
    print(f"\n  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")

    # Save summary to text file
    summary_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(summary_path, 'w') as f:
        f.write("Cardiac Remodeling Model — Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model:          {args.model}\n")
        f.write(f"Test set size:  {len(y_test)}\n")
        f.write(f"Positive:       {int(y_test.sum())}\n")
        f.write(f"Negative:       {int(len(y_test) - y_test.sum())}\n\n")
        f.write(f"Threshold:      {threshold:.3f}\n")
        f.write(f"AUC:            {roc_auc:.3f}\n")
        f.write(f"Accuracy:       {accuracy:.3f}\n")
        f.write(f"Sensitivity:    {sensitivity:.3f}\n")
        f.write(f"Specificity:    {specificity:.3f}\n")
        f.write(f"PPV:            {ppv:.3f}\n")
        f.write(f"NPV:            {npv:.3f}\n")
        f.write(f"F1-score:       {f1:.3f}\n")
        f.write(f"Avg Precision:  {ap:.3f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TP={tp}  FP={fp}\n")
        f.write(f"  FN={fn}  TN={tn}\n")

    print(f"\nSaved to {args.output_dir}/:")
    print(f"  roc_curve.png")
    print(f"  precision_recall_curve.png")
    print(f"  confusion_matrix.png")
    print(f"  training_history.png")
    print(f"  evaluation_results.txt")


if __name__ == '__main__':
    main()
