"""
Smart hyperparameter search for ECG cardiac remodeling fine-tuning.

Phase 1 (Exploration): Random search across broad parameter space.
Phase 2 (Exploitation): Focuses on the best-performing region and
         fine-tunes around those hyperparameters.

Only keeps top 10 model directories. Deletes the rest to save disk space.

Usage:
    python hyperparameter_search.py prepared_data/ecg_tracings.hdf5 prepared_data/labels.csv \
        --pretrained_model model.hdf5

Output:
    ./hyperparam_search_output/results_summary.csv  — all results sorted by AUC
    ./hyperparam_search_output/top_N/               — top 10 model directories
    ./hyperparam_search_output/best_model.keras      — overall best model
"""

import argparse
import os
import random
import shutil
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
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

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


def run_single_config(config, args, config_id, total):
    """Train and evaluate a single hyperparameter configuration."""
    print(f"\n{'='*60}")
    print(f"[{config_id}/{total}] Config: LR={config['lr']}  BS={config['batch_size']}  "
          f"Freeze={config['freeze_until'] or 'none'}  DO={config['dropout']}  "
          f"CW={config['class_weight_pos']}")
    print(f"{'='*60}")

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

    # Save model in temp dir
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

    try:
        val_auc = roc_auc_score(val_y, y_prob)
    except ValueError:
        val_auc = 0.0

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
        'phase': config.get('phase', 1),
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

    print(f"  => AUC={val_auc:.3f}  F1={val_f1:.3f}  Sens={sensitivity:.3f}  "
          f"Spec={specificity:.3f}  Epochs={epochs_run}  Time={train_time:.0f}s")

    del model
    keras.backend.clear_session()

    return result, config_dir


def generate_phase1_configs(n_configs):
    """Phase 1: broad random exploration."""
    configs = []
    for _ in range(n_configs):
        config = {
            'phase': 1,
            'lr': random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
            'batch_size': random.choice([8, 16, 32]),
            'freeze_until': random.choice([
                'batch_normalization_3',   # first 2 blocks
                'batch_normalization_5',   # first 3 blocks
                'batch_normalization_7',   # first 4 blocks
                None,                       # no freeze
            ]),
            'dropout': random.choice([0.0, 0.2, 0.3, 0.4, 0.5]),
            'class_weight_pos': random.choice(['auto', 2.0, 3.0, 4.0, 5.0]),
        }
        configs.append(config)
    return configs


def generate_phase2_configs(top_results, n_configs):
    """Phase 2: exploit best results — generate variations around top configs."""
    configs = []

    for _, row in top_results.iterrows():
        base_lr = row['lr']
        base_bs = int(row['batch_size'])
        base_freeze = row['freeze_until'] if row['freeze_until'] != 'none' else None
        base_dropout = row['dropout']
        base_cw = row['class_weight_pos']

        # Try small variations around this config
        lr_variations = [base_lr * 0.5, base_lr, base_lr * 2.0]
        dropout_variations = [
            max(0, base_dropout - 0.1),
            base_dropout,
            min(0.7, base_dropout + 0.1)
        ]

        try:
            cw_val = float(base_cw)
            cw_variations = [max(1.0, cw_val - 1.0), cw_val, cw_val + 1.0]
        except (ValueError, TypeError):
            cw_variations = ['auto']

        for lr in lr_variations:
            for do in dropout_variations:
                config = {
                    'phase': 2,
                    'lr': round(lr, 6),
                    'batch_size': base_bs,
                    'freeze_until': base_freeze,
                    'dropout': round(do, 2),
                    'class_weight_pos': random.choice(cw_variations),
                }
                configs.append(config)

    # Deduplicate and limit
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c['lr'], c['batch_size'], c['freeze_until'], c['dropout'],
               str(c['class_weight_pos']))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    random.shuffle(unique_configs)
    return unique_configs[:n_configs]


def cleanup_keep_top_n(all_results, output_dir, top_n=10):
    """Delete all config directories except the top N by AUC."""
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('val_auc', ascending=False)

    top_ids = set(results_df.head(top_n)['config_id'].astype(int).tolist())
    all_ids = set(results_df['config_id'].astype(int).tolist())
    remove_ids = all_ids - top_ids

    removed = 0
    for cid in remove_ids:
        config_dir = os.path.join(output_dir, f"config_{cid}")
        if os.path.exists(config_dir):
            shutil.rmtree(config_dir)
            removed += 1

    print(f"\nCleanup: kept top {top_n} models, removed {removed} directories")
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Smart hyperparameter search for ECG fine-tuning.')
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
    parser.add_argument('--phase1_runs', type=int, default=30,
                        help='Number of random exploration runs (default: 30)')
    parser.add_argument('--phase2_runs', type=int, default=20,
                        help='Number of exploitation runs around best configs (default: 20)')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Keep only top N model directories (default: 10)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_runs = args.phase1_runs + args.phase2_runs
    all_results = []
    config_id = 0

    # ==========================================
    # PHASE 1: EXPLORATION (broad random search)
    # ==========================================
    print("\n" + "#" * 60)
    print("PHASE 1: EXPLORATION — random search across broad parameter space")
    print("#" * 60)

    phase1_configs = generate_phase1_configs(args.phase1_runs)
    print(f"Running {args.phase1_runs} random configurations...\n")

    for config in phase1_configs:
        config_id += 1
        try:
            result, _ = run_single_config(config, args, config_id, total_runs)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Analyze phase 1 results
    phase1_df = pd.DataFrame(all_results)
    phase1_df = phase1_df.sort_values('val_auc', ascending=False)

    print(f"\n{'='*60}")
    print("PHASE 1 RESULTS — Top 5")
    print("=" * 60)
    for _, row in phase1_df.head(5).iterrows():
        print(f"  AUC={row['val_auc']:.3f} | F1={row['val_f1']:.3f} | "
              f"LR={row['lr']} | BS={int(row['batch_size'])} | "
              f"Freeze={row['freeze_until']} | DO={row['dropout']} | "
              f"CW={row['class_weight_pos']}")

    # ==========================================
    # PHASE 2: EXPLOITATION (refine best configs)
    # ==========================================
    print("\n" + "#" * 60)
    print("PHASE 2: EXPLOITATION — refining around best configurations")
    print("#" * 60)

    top3_phase1 = phase1_df.head(3)
    phase2_configs = generate_phase2_configs(top3_phase1, args.phase2_runs)
    print(f"Running {len(phase2_configs)} refined configurations "
          f"based on top 3 from Phase 1...\n")

    for config in phase2_configs:
        config_id += 1
        try:
            result, _ = run_single_config(config, args, config_id, total_runs)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")

    # ==========================================
    # FINAL: cleanup, save results, report
    # ==========================================
    results_df = cleanup_keep_top_n(all_results, args.output_dir, args.top_n)

    # Save CSV
    results_path = os.path.join(args.output_dir, 'results_summary.csv')
    results_df.to_csv(results_path, index=False)

    # Copy best model
    best_id = int(results_df.iloc[0]['config_id'])
    best_src = os.path.join(args.output_dir, f"config_{best_id}", 'best_model.keras')
    best_dst = os.path.join(args.output_dir, 'best_model.keras')
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)

    # Print final report
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total runs: {len(all_results)} "
          f"(Phase 1: {args.phase1_runs}, Phase 2: {len(phase2_configs)})")
    print(f"Results saved to: {results_path}")

    print(f"\n{'='*80}")
    print(f"TOP {args.top_n} CONFIGURATIONS (sorted by AUC)")
    print("=" * 80)
    for rank, (_, row) in enumerate(results_df.head(args.top_n).iterrows(), 1):
        print(f"  #{rank:2d} [Phase {int(row['phase'])}] | "
              f"AUC={row['val_auc']:.3f} | F1={row['val_f1']:.3f} | "
              f"Sens={row['sensitivity']:.3f} | Spec={row['specificity']:.3f} | "
              f"LR={row['lr']} | BS={int(row['batch_size'])} | "
              f"Freeze={row['freeze_until']} | DO={row['dropout']} | "
              f"CW={row['class_weight_pos']}")

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print("=" * 80)
    best = results_df.iloc[0]
    print(f"  Phase:            {int(best['phase'])}")
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
    print(f"\n  Best model: {best_dst}")

    # Reproduce command
    freeze_arg = (f"--freeze_until {best['freeze_until']}"
                  if best['freeze_until'] != 'none'
                  else "--no_freeze")
    print(f"\nReproduce best run:")
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
