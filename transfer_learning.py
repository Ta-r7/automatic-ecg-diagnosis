"""
Transfer learning script voor cardiac remodeling detectie bij atleten.

Stap 1: Laad het pre-trained ECG model (weights van Zenodo)
Stap 2: Vervang de output laag voor jouw taak
Stap 3: Train op jouw eigen data

Gebruik:
    python transfer_learning.py \
        path/to/athlete_ecgs.hdf5 \
        path/to/labels.csv \
        --pretrained_weights path/to/model.hdf5

Dataset formaat:
    - HDF5: dataset 'tracings' met shape (N, 4096, 12)
    - CSV: kolom 'cardiac_remodeling' met 0 of 1 per ECG
           (of meerdere kolommen als je meerdere klassen wilt)
"""

import argparse
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
)
from model import get_model
from datasets import ECGSequence


def build_transfer_model(pretrained_weights_path, n_classes, freeze_base=True):
    """
    Laad pre-trained model en vervang de output laag.

    Parameters
    ----------
    pretrained_weights_path : str
        Pad naar het .hdf5 bestand met de Zenodo weights.
    n_classes : int
        Aantal output klassen (bijv. 1 voor binaire cardiac remodeling detectie).
    freeze_base : bool
        Als True: bevries alle lagen behalve de nieuwe output laag.
        Als False: alle lagen worden meegetraind (langzamer, maar flexibeler).

    Returns
    -------
    model : tf.keras.Model
    """
    # Laad het originele model met zijn 6 klassen
    base_model = load_model(pretrained_weights_path, compile=False)
    print(f"Pre-trained model geladen: {pretrained_weights_path}")
    print(f"Originele output shape: {base_model.output_shape}")

    # Alles bevroren behalve de laatste Dense laag
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
        print(f"Basis lagen bevroren ({len(base_model.layers) - 1} lagen)")

    # Vervang de output laag
    # De laag voor de output is de Flatten laag
    flatten_output = base_model.layers[-2].output  # laag voor de Dense output

    # Kies activatiefunctie op basis van n_classes
    if n_classes == 1:
        activation = 'sigmoid'   # binaire classificatie
        loss = 'binary_crossentropy'
    else:
        activation = 'sigmoid'   # multi-label (meerdere aandoeningen tegelijk mogelijk)
        loss = 'binary_crossentropy'

    new_output = Dense(
        n_classes,
        activation=activation,
        kernel_initializer='he_normal',
        name='cardiac_remodeling_output'
    )(flatten_output)

    model = Model(inputs=base_model.input, outputs=new_output)

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"Trainbare lagen: {trainable_count} / {len(model.layers)}")

    return model, loss


def unfreeze_top_layers(model, n_layers=4):
    """
    Ontgrendel de bovenste n_layers lagen van het basis model voor fine-tuning.
    Gebruik dit na de eerste trainingsfase.
    """
    for layer in model.layers[-n_layers:]:
        layer.trainable = True
    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"Fine-tune modus: {trainable_count} trainbare lagen")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer learning voor cardiac remodeling detectie.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='Pad naar HDF5 bestand met ECG opnames (shape: N x 4096 x 12)')
    parser.add_argument('path_to_csv', type=str,
                        help='Pad naar CSV met labels (kolom(men) met 0/1 per ECG)')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                        help='Pad naar pre-trained model weights (.hdf5) van Zenodo')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='Naam van de HDF5 dataset. Default: tracings')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fractie van data voor validatie. Default: 0.15')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch grootte. Default: 32')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                        help='Bevriest basis lagen tijdens eerste training fase. Default: True')
    parser.add_argument('--fine_tune', action='store_true', default=False,
                        help='Na eerste fase: ontgrendel bovenste lagen en train verder')
    parser.add_argument('--fine_tune_layers', type=int, default=4,
                        help='Aantal bovenste lagen om te ontgrendelen bij fine-tuning. Default: 4')
    args = parser.parse_args()

    # --- Fase 1: Train alleen de nieuwe output laag ---
    print("\n=== FASE 1: Output laag trainen ===")
    lr = 1e-3

    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv,
        args.batch_size, args.val_split
    )

    n_classes = train_seq.n_classes
    print(f"Aantal klassen: {n_classes}")
    print(f"Trainset grootte: {len(train_seq) * args.batch_size} ECGs")
    print(f"Validatieset grootte: {len(valid_seq) * args.batch_size} ECGs\n")

    model, loss = build_transfer_model(
        args.pretrained_weights,
        n_classes=n_classes,
        freeze_base=args.freeze_base
    )

    model.compile(optimizer=Adam(lr), loss=loss, metrics=['accuracy'])
    model.summary()

    callbacks_phase1 = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=lr / 100),
        EarlyStopping(patience=8, min_delta=1e-5, restore_best_weights=True),
        ModelCheckpoint('./cardiac_remodeling_phase1_best.hdf5', save_best_only=True,
                        monitor='val_loss', verbose=1),
        ModelCheckpoint('./cardiac_remodeling_phase1_last.hdf5', verbose=0),
        CSVLogger('./training_phase1.log', append=False),
        TensorBoard(log_dir='./logs/phase1', write_graph=False),
    ]

    print("\nFase 1 training gestart...")
    model.fit(
        train_seq,
        epochs=30,
        callbacks=callbacks_phase1,
        validation_data=valid_seq,
        verbose=1
    )

    # --- Fase 2 (optioneel): Fine-tune bovenste lagen ---
    if args.fine_tune:
        print(f"\n=== FASE 2: Fine-tuning (bovenste {args.fine_tune_layers} lagen) ===")
        lr_finetune = 1e-4  # veel lagere learning rate om pre-trained kennis te bewaren

        model = unfreeze_top_layers(model, n_layers=args.fine_tune_layers)
        model.compile(optimizer=Adam(lr_finetune), loss=loss, metrics=['accuracy'])

        callbacks_phase2 = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              min_lr=lr_finetune / 100),
            EarlyStopping(patience=8, min_delta=1e-5, restore_best_weights=True),
            ModelCheckpoint('./cardiac_remodeling_finetuned_best.hdf5', save_best_only=True,
                            monitor='val_loss', verbose=1),
            ModelCheckpoint('./cardiac_remodeling_finetuned_last.hdf5', verbose=0),
            CSVLogger('./training_phase2.log', append=False),
            TensorBoard(log_dir='./logs/phase2', write_graph=False),
        ]

        print("\nFase 2 fine-tuning gestart...")
        model.fit(
            train_seq,
            epochs=20,
            callbacks=callbacks_phase2,
            validation_data=valid_seq,
            verbose=1
        )

        model.save('./cardiac_remodeling_final.hdf5')
        print("\nFinaal model opgeslagen: cardiac_remodeling_final.hdf5")
    else:
        model.save('./cardiac_remodeling_phase1.hdf5')
        print("\nModel opgeslagen: cardiac_remodeling_phase1.hdf5")
