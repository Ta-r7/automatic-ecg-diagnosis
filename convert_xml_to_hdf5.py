"""
Converteert GE MUSE XML ECG bestanden naar HDF5 + CSV voor transfer learning.

Gebruik:
    python convert_xml_to_hdf5.py \
        --xml_dir pad/naar/xml_bestanden \
        --labels_csv pad/naar/jouw_labels.csv \
        --output_hdf5 ecgs.hdf5 \
        --output_csv labels_ordered.csv

Labels CSV formaat (jouw invoer):
    Record_ID,label
    110012,1
    110013,0
    ...

De Record_ID in de CSV wordt gekoppeld aan de <Record_ID> tag in de XML.
Output CSV heeft dezelfde volgorde als de HDF5 dataset, zodat
rij 0 in CSV = ECG 0 in HDF5.
"""

import argparse
import base64
import glob
import os
import struct
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd

# Volgorde die het model verwacht
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Model verwacht 4096 samples per lead
TARGET_SAMPLES = 4096


def decode_waveform(waveform_data_text, n_samples):
    """Decodeer base64 → int16 array."""
    # Verwijder witruimte (spaties, newlines)
    clean = ''.join(waveform_data_text.split())
    raw = base64.b64decode(clean)
    # Elke sample = 2 bytes (int16, little-endian)
    samples = struct.unpack(f'<{n_samples}h', raw[:n_samples * 2])
    return np.array(samples, dtype=np.float32)


def parse_amplitude_units_per_bit(text):
    """Verwerkt '4,88' (NL) of '4.88' (EN) naar float."""
    return float(text.replace(',', '.'))


def read_ecg_xml(xml_path):
    """
    Lees een GE MUSE XML en geef (record_id, signaal_12leads) terug.
    signaal shape: (TARGET_SAMPLES, 12)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Record ID
    record_id_elem = root.find('.//Record_ID')
    record_id = record_id_elem.text.strip() if record_id_elem is not None else None

    # Zoek Rhythm waveform (langste opname, 5000 samples bij 500 Hz)
    # Val terug op Median als Rhythm niet beschikbaar is
    rhythm_waveform = None
    median_waveform = None
    for wf in root.findall('.//Waveform'):
        wf_type = wf.findtext('WaveformType', '').strip().upper()
        if wf_type == 'RHYTHM':
            rhythm_waveform = wf
        elif wf_type == 'MEDIAN':
            median_waveform = wf

    waveform = rhythm_waveform if rhythm_waveform is not None else median_waveform
    if waveform is None:
        raise ValueError(f"Geen Waveform gevonden in {xml_path}")

    # Lees alle leads
    leads = {}
    for lead_data in waveform.findall('LeadData'):
        lead_id = lead_data.findtext('LeadID', '').strip()
        n_samples = int(lead_data.findtext('LeadSampleCountTotal', '0'))
        units_per_bit_text = lead_data.findtext('LeadAmplitudeUnitsPerBit', '1')
        units_per_bit = parse_amplitude_units_per_bit(units_per_bit_text)
        waveform_text = lead_data.findtext('WaveFormData', '')

        signal = decode_waveform(waveform_text, n_samples)
        signal_uv = signal * units_per_bit  # naar microvolts
        leads[lead_id] = signal_uv

    # Bereken afgeleide leads als ze ontbreken
    if 'I' in leads and 'II' in leads:
        i = leads['I']
        ii = leads['II']

        if 'III' not in leads:
            leads['III'] = ii - i
        if 'aVR' not in leads:
            leads['aVR'] = -(i + ii) / 2.0
        if 'aVL' not in leads:
            leads['aVL'] = (i - leads['III']) / 2.0
        if 'aVF' not in leads:
            leads['aVF'] = (ii + leads['III']) / 2.0

    # Zet leads samen in de juiste volgorde
    n_available = min(len(leads[l]) for l in LEAD_ORDER if l in leads)
    signal_matrix = np.zeros((n_available, 12), dtype=np.float32)

    for col, lead_name in enumerate(LEAD_ORDER):
        if lead_name in leads:
            signal_matrix[:, col] = leads[lead_name][:n_available]
        else:
            print(f"  Waarschuwing: lead {lead_name} ontbreekt in {xml_path}, vul met nullen")

    # Crop of pad naar TARGET_SAMPLES
    n = signal_matrix.shape[0]
    if n >= TARGET_SAMPLES:
        signal_matrix = signal_matrix[:TARGET_SAMPLES, :]
    else:
        pad = np.zeros((TARGET_SAMPLES - n, 12), dtype=np.float32)
        signal_matrix = np.vstack([signal_matrix, pad])

    return record_id, signal_matrix


def main():
    parser = argparse.ArgumentParser(description='Converteer GE MUSE XML ECGs naar HDF5.')
    parser.add_argument('--xml_dir', type=str, required=True,
                        help='Map met XML bestanden')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='CSV met kolommen: Record_ID,<label_kolom(men)>')
    parser.add_argument('--output_hdf5', type=str, default='ecgs.hdf5',
                        help='Naam van het uitvoer HDF5 bestand')
    parser.add_argument('--output_csv', type=str, default='labels_ordered.csv',
                        help='Naam van de uitvoer labels CSV (zelfde volgorde als HDF5)')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='Naam van de HDF5 dataset. Default: tracings')
    args = parser.parse_args()

    # Lees labels
    labels_df = pd.read_csv(args.labels_csv)
    labels_df['Record_ID'] = labels_df['Record_ID'].astype(str).str.strip()
    labels_dict = {row['Record_ID']: row for _, row in labels_df.iterrows()}

    label_columns = [c for c in labels_df.columns if c != 'Record_ID']
    print(f"Label kolommen: {label_columns}")

    # Zoek alle XML bestanden
    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, '*.xml')))
    if not xml_files:
        xml_files = sorted(glob.glob(os.path.join(args.xml_dir, '*.XML')))
    print(f"Gevonden: {len(xml_files)} XML bestanden")

    # Verwerk XMLs
    ecg_list = []
    label_rows = []
    skipped = 0

    for xml_path in xml_files:
        fname = os.path.basename(xml_path)
        try:
            record_id, signal = read_ecg_xml(xml_path)

            if record_id not in labels_dict:
                print(f"  Overgeslagen: {fname} (Record_ID '{record_id}' niet in labels CSV)")
                skipped += 1
                continue

            ecg_list.append(signal)
            label_rows.append(labels_dict[record_id])
            print(f"  OK: {fname} | Record_ID={record_id} | shape={signal.shape}")

        except Exception as e:
            print(f"  FOUT bij {fname}: {e}")
            skipped += 1

    if not ecg_list:
        print("Geen ECGs verwerkt. Controleer XML map en labels CSV.")
        return

    # Sla op als HDF5
    ecg_array = np.stack(ecg_list, axis=0)  # shape: (N, 4096, 12)
    print(f"\nECG array shape: {ecg_array.shape}")

    with h5py.File(args.output_hdf5, 'w') as f:
        f.create_dataset(args.dataset_name, data=ecg_array, dtype=np.float32)
    print(f"HDF5 opgeslagen: {args.output_hdf5}")

    # Sla labels op in dezelfde volgorde
    output_df = pd.DataFrame(label_rows)[label_columns]
    output_df.to_csv(args.output_csv, index=False)
    print(f"Labels CSV opgeslagen: {args.output_csv}")
    print(f"\nSamenvatting:")
    print(f"  Verwerkt: {len(ecg_list)} ECGs")
    print(f"  Overgeslagen: {skipped}")
    print(f"  ECG shape: {ecg_array.shape}  (N x {TARGET_SAMPLES} samples x 12 leads)")
    print(f"\nGebruik voor transfer learning:")
    print(f"  python transfer_learning.py {args.output_hdf5} {args.output_csv} --pretrained_weights model.hdf5")


if __name__ == '__main__':
    main()
