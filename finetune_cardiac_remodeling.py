"""
Fine-tune the pre-trained ECG ResNet model for cardiac remodeling detection.

This script loads the pre-trained model (trained on 6 arrhythmia classes) and
replaces the final classification layer to detect cardiac remodeling patterns.

Usage:
    python finetune_cardiac_remodeling.py path/to/ecg_data.hdf5 path/to/labels.csv \
        --pretrained_model model.hdf5 \
        --n_classes 2 \
        --freeze_until flatten_1

Arguments:
    path_to_hdf5        : HDF5 file with ECG tracings, shape (N, 4096, 12)
    path_to_csv         : CSV file with cardiac remodeling labels (0/1 columns)
    --pretrained_model  : Path to pre-trained model weights (default: model.hdf5)
    --n_classes         : Number of output classes (default: 2)
    --freeze_until      : Freeze all layers up to and including this layer name
                          (default: batch_normalization_7, i.e. first 3 residual blocks)
    --epochs            : Max training epochs (default: 50)
    --batch_size        : Batch size (default: 32)
    --lr                : Initial learning rate (default: 0.0001)
    --val_split         : Validation split fraction (default: 0.1)
    --output_dir        : Directory for output files (default: ./finetune_output)
    --dataset_name      : HDF5 dataset name (default: tracings)
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
    CSVLogger, EarlyStopping
)

from datasets import ECGSequence


def build_finetune_model(pretrained_path, n_classes, freeze_until=None):
    """Load pre-trained model and replace the output layer for fine-tuning.

    Parameters
    ----------
    pretrained_path : str
        Path to the pre-trained .hdf5 model file.
    n_classes : int
        Number of output classes for the new task.
    freeze_until : str or None
        Name of the layer up to which all layers are frozen.
        If None, no layers are frozen (full fine-tuning).

    Returns
    -------
    keras.Model
        The modified model ready for fine-tuning.
    """
    # Load the pre-trained model
    base_model = keras.models.load_model(pretrained_path, compile=False)
    print(f"Loaded pre-trained model from: {pretrained_path}")
    print(f"Original output shape: {base_model.output_shape}")

    # Get the feature extraction part (everything before the final Dense layer)
    # The layer before the output Dense is 'flatten_1'
    feature_layer = base_model.get_layer('flatten_1')
    features = feature_layer.output

    # Add new classification head
    new_output = Dense(
        n_classes,
        activation='sigmoid',
        kernel_initializer='he_normal',
        name='cardiac_remodeling_output'
    )(features)

    model = Model(inputs=base_model.input, outputs=new_output)

    # Freeze layers if requested
    if freeze_until is not None:
        freeze = True
        frozen_count = 0
        for layer in model.layers:
            if freeze:
                layer.trainable = False
                frozen_count += 1
            if layer.name == freeze_until:
                freeze = False
        trainable_count = len(model.layers) - frozen_count
        print(f"Frozen {frozen_count} layers (up to '{freeze_until}')")
        print(f"Trainable layers: {trainable_count}")
    else:
        print("Full fine-tuning: all layers are trainable")

    print(f"New output shape: {model.output_shape}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune ECG model for cardiac remodeling detection.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='Path to HDF5 file containing ECG tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='Path to CSV file containing cardiac remodeling labels')
    parser.add_argument('--pretrained_model', type=str, default='model.hdf5',
                        help='Path to pre-trained model (default: model.hdf5)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--freeze_until', type=str, default='batch_normalization_7',
                        help='Freeze layers up to this layer name '
                             '(default: batch_normalization_7 = first 3 residual blocks)')
    parser.add_argument('--no_freeze', action='store_true',
                        help='Do not freeze any layers (full fine-tuning)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='./finetune_output',
                        help='Output directory (default: ./finetune_output)')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='HDF5 dataset name (default: tracings)')
    parser.add_argument('--manual_class_weights', action='store_true', default=False,
                        help='Use manual class weights instead of auto-computed '
                             '(default: False, i.e. auto-compute from data)')
    parser.add_argument('--class_weight_neg', type=float, default=1.0,
                        help='Manual class weight for negative class (default: 1.0)')
    parser.add_argument('--class_weight_pos', type=float, default=4.0,
                        help='Manual class weight for positive class (default: 4.0)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    freeze_until = None if args.no_freeze else args.freeze_until
    model = build_finetune_model(args.pretrained_model, args.n_classes, freeze_until)

    # Compile
    loss = 'binary_crossentropy'
    opt = Adam(learning_rate=args.lr)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.summary()

    # Load data
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv,
        args.batch_size, args.val_split)

    print(f"\nTraining samples: ~{len(train_seq) * args.batch_size}")
    print(f"Validation samples: ~{len(valid_seq) * args.batch_size}")
    print(f"Output classes: {args.n_classes}")

    # Class weights to handle imbalance
    if args.manual_class_weights:
        class_weight = {0: args.class_weight_neg, 1: args.class_weight_pos}
        print(f"Class weights (manual): {{0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}}}")
    else:
        train_labels = train_seq.y
        if train_labels.ndim > 1:
            train_labels_flat = train_labels[:, 0]
        else:
            train_labels_flat = train_labels
        n_neg = np.sum(train_labels_flat == 0)
        n_pos = np.sum(train_labels_flat == 1)
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}
        print(f"Class weights (auto): {{0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}}}")

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=7,
            min_lr=args.lr / 100, verbose=1),
        EarlyStopping(
            monitor='val_loss', patience=12,
            min_delta=1e-5, restore_best_weights=True, verbose=1),
        TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs'),
            write_graph=False),
        CSVLogger(
            os.path.join(args.output_dir, 'finetune_training.log'),
            append=False),
        ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.keras'),
            save_best_only=True, monitor='val_loss', verbose=1),
        ModelCheckpoint(
            os.path.join(args.output_dir, 'last_model.keras'),
            verbose=0),
    ]

    # Train
    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)

    history = model.fit(
        train_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=valid_seq,
        class_weight=class_weight,
        verbose=1)

    # Save final model
    final_path = os.path.join(args.output_dir, 'cardiac_remodeling_model.keras')
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.keras')}")
    print(f"Training log: {os.path.join(args.output_dir, 'finetune_training.log')}")


if __name__ == "__main__":
    main()
