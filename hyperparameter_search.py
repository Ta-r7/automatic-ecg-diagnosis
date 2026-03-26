"""
Automated hyperparameter search for ECG cardiac remodeling fine-tuning.

Tries many combinations of learning rate, batch size, freeze strategy,
dropout, and class weights. Trains each combination, evaluates on the
test set, and reports the best configuration.

Usage:
    python hyperparameter_search.py prepared_data/ecg_tracings.hdf5 prepared_data/labels.csv \
        --pretrained_model model.hdf5 --n_runs_per_config 2

Output:
    ./hyperparam_search_output/results_summary.csv   — all results sorted by AUC
    ./hyperparam_search_output/best_model.keras       — best model saved
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import roc_auc_score, f1_score

from datasets import ECGSequence


def build_model(pretrained_path, n_classes, freeze_until=None, dropout_rate=0.0):
    """Build fine-tune model with given hyperparameters."""
    base_model = keras.models.load_model(pretrained_path, compile=False)

    feature_layer = base_model.get_layer('flatten_1')
    features = feature_layer.output

    if dropout_rate > 0:
        features = Dropout(dropout_rate, name='head_dropout')(features)

    new_output = Dense(
        n_classes,
        activation='sigmoid',
        kernel_initializer='he_normal',
        name='cardiac_remodeling_output'
    )(features)

    model = Model(inputs=base_model.input, outputs=new_output)

    if freeze_until is not None:
        freeze = True
        for layer in model.layers:
            if freeze:
                layer.trainable = False
            if layer.name == freeze_until:
                freeze = False

    return model


def run_single_config(config, args, config_id):
    """Train and evaluate a single hyperparameter configuration."""
    print(f"\n{'='*60}")
    print(f"Config {config_id}: {config}")
    print(f"{'='*60}")

    # Build model
    model = build_model(
        args.pretrained_model,
        args.n_classes,
        freeze_until=config['freeze_until'],
        dropout_rate=config['dropout']
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=config['lr']),
        metrics=['accuracy']
    )

    # Load data
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv,
        config['batch_size'], args.val_split)

    # Class weights
    train_labels = train_seq.y
    if train_labels.ndim > 1:
        train_labels_flat = train_labels[:, 0]
    else:
        train_labels_flat = train_labels
    n_neg = np.sum(train_labels_flat == 0)
    n_pos = np.sum(train_labels_flat == 1)

    if config['class_weight_pos'] == 'auto':
        cw = {0: 1.0, 1: n_neg / max(n_pos, 1)}
    else:
        cw = {0: 1.0, 1: config['class_weight_pos']}

    # Callbacks
    config_dir = os.path.join(args.output_dir, f"config_{config_id}")
    os.makedirs(config_dir, exist_ok=True)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                          min_lr=config['lr'] / 100, verbose=0),
        EarlyStopping(monitor='val_loss', patience=12, min_delta=1e-5,
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(os.path.join(config_dir, 'best_model.keras'),
                        save_best_only=True, monitor='val_loss', verbose=0),
    ]

    # Train
    start_time = time.time()
    history = model.fit(
        train_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=valid_seq,
        class_weight=cw,
        verbose=0)
    train_time = time.time() - start_time
    epochs_run = len(history.history['loss'])

    # Evaluate on validation set
    val_x_batches = []
    val_y_batches = []
    for i in range(len(valid_seq)):
        x_batch, y_batch = valid_seq[i]
        val_x_batches.append(x_batch)
        val_y_batches.append(y_batch)
    val_x = np.concatenate(val_x_batches, axis=0)
    val_y = np.concatenate(val_y_batches, axis=0).flatten()

    y_prob = model.predict(val_x, batch_size=32, verbose=0).flatten()

    # Metrics
    try:
        val_auc = roc_auc_score(val_y, y_prob)
    except ValueError:
        val_auc = 0.0

    # Optimal threshold (Youden)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(val_y, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    threshold = thresholds[best_idx]

    y_pred = (y_prob >= threshold).astype(int)
    val_f1 = f1_score(val_y, y_pred, zero_division=0)

    tp = np.sum((y_pred == 1) & (val_y == 1))
    fn = np.sum((y_pred == 0) & (val_y == 1))
    fp = np.sum((y_pred == 1) & (val_y == 0))
    tn = np.sum((y_pred == 0) & (val_y == 0))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    best_val_loss = min(history.history['val_loss'])

    result = {
        'config_id': config_id,
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'freeze_until': config['freeze_until'] or 'none',
        'dropout': config['dropout'],
        'class_weight_pos': str(cw[1]),
        'val_auc': val_auc,
        'val_f1': val_f1,
        'val_loss': best_val_loss,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'threshold': threshold,
        'epochs_run': epochs_run,
        'train_time_s': round(train_time, 1),
    }

    print(f"  AUC={val_auc:.3f}  F1={val_f1:.3f}  Sens={sensitivity:.3f}  "
          f"Spec={specificity:.3f}  Epochs={epochs_run}  Time={train_time:.0f}s")

    # Clean up to free memory
    del model
    keras.backend.clear_session()

    return result, config_dir


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for ECG fine-tuning.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='Path to HDF5 file with ECG tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='Path to CSV file with labels')
    parser.add_argument('--pretrained_model', type=str, default='model.hdf5',
                        help='Pre-trained model path')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of output classes (default: 1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per run (default: 50)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split (default: 0.1)')
    parser.add_argument('--output_dir', type=str,
                        default='./hyperparam_search_output',
                        help='Output directory')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='HDF5 dataset name')
    parser.add_argument('--n_runs_per_config', type=int, default=1,
                        help='Runs per config to average out randomness (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # === HYPERPARAMETER GRID ===
    param_grid = {
        'lr': [0.01, 0.001, 0.0005, 0.0001],
        'batch_size': [16, 32],
        'freeze_until': [
            'batch_normalization_3',   # freeze first 2 residual blocks
            'batch_normalization_7',   # freeze first 4 residual blocks
            None,                       # no freeze (full fine-tune)
        ],
        'dropout': [0.0, 0.3, 0.5],
        'class_weight_pos': ['auto', 3.0, 5.0],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    total = len(all_configs) * args.n_runs_per_config
    print(f"Hyperparameter search: {len(all_configs)} configs x "
          f"{args.n_runs_per_config} runs = {total} total runs")
    print(f"Parameter grid:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")

    # Run all configurations
    all_results = []
    best_auc = 0
    best_config_dir = None
    config_id = 0

    for config in all_configs:
        for run in range(args.n_runs_per_config):
            config_id += 1
            print(f"\n[{config_id}/{total}]", end="")

            try:
                result, config_dir = run_single_config(config, args, config_id)
                all_results.append(result)

                if result['val_auc'] > best_auc:
                    best_auc = result['val_auc']
                    best_config_dir = config_dir

            except Exception as e:
                print(f"  FAILED: {e}")
                all_results.append({
                    'config_id': config_id,
                    'lr': config['lr'],
                    'batch_size': config['batch_size'],
                    'freeze_until': config['freeze_until'] or 'none',
                    'dropout': config['dropout'],
                    'class_weight_pos': str(config['class_weight_pos']),
                    'val_auc': 0, 'val_f1': 0, 'val_loss': 999,
                    'accuracy': 0, 'sensitivity': 0, 'specificity': 0,
                    'ppv': 0, 'threshold': 0.5,
                    'epochs_run': 0, 'train_time_s': 0,
                })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('val_auc', ascending=False)
    results_path = os.path.join(args.output_dir, 'results_summary.csv')
    results_df.to_csv(results_path, index=False)

    # Copy best model
    if best_config_dir:
        import shutil
        best_src = os.path.join(best_config_dir, 'best_model.keras')
        best_dst = os.path.join(args.output_dir, 'best_model.keras')
        if os.path.exists(best_src):
            shutil.copy2(best_src, best_dst)

    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nTotal configurations tested: {len(all_configs)}")
    print(f"Total runs: {total}")
    print(f"\nResults saved to: {results_path}")

    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS (sorted by AUC)")
    print("=" * 80)
    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        print(f"  Config {int(row['config_id']):3d} | "
              f"AUC={row['val_auc']:.3f} | F1={row['val_f1']:.3f} | "
              f"Sens={row['sensitivity']:.3f} | Spec={row['specificity']:.3f} | "
              f"LR={row['lr']} | BS={int(row['batch_size'])} | "
              f"Freeze={row['freeze_until']} | DO={row['dropout']} | "
              f"CW={row['class_weight_pos']}")

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print("=" * 80)
    best = results_df.iloc[0]
    print(f"  Learning rate:    {best['lr']}")
    print(f"  Batch size:       {int(best['batch_size'])}")
    print(f"  Freeze until:     {best['freeze_until']}")
    print(f"  Dropout:          {best['dropout']}")
    print(f"  Class weight pos: {best['class_weight_pos']}")
    print(f"  ---")
    print(f"  AUC:              {best['val_auc']:.3f}")
    print(f"  F1-score:         {best['val_f1']:.3f}")
    print(f"  Sensitivity:      {best['sensitivity']:.3f}")
    print(f"  Specificity:      {best['specificity']:.3f}")
    print(f"  Accuracy:         {best['accuracy']:.3f}")
    print(f"  Threshold:        {best['threshold']:.3f}")
    print(f"\n  Best model saved to: {os.path.join(args.output_dir, 'best_model.keras')}")

    # Print command to reproduce best run
    freeze_arg = (f"--freeze_until {best['freeze_until']}"
                  if best['freeze_until'] != 'none'
                  else "--no_freeze")
    print(f"\nTo reproduce the best run:")
    print(f"  python finetune_cardiac_remodeling.py {args.path_to_hdf5} {args.path_to_csv} "
          f"--pretrained_model {args.pretrained_model} "
          f"--n_classes {args.n_classes} "
          f"--lr {best['lr']} "
          f"--batch_size {int(best['batch_size'])} "
          f"{freeze_arg} "
          f"--dropout {best['dropout']} "
          f"--epochs {args.epochs}")


if __name__ == '__main__':
    main()
