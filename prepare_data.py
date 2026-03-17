"""
Prepare ECG data from GE MUSE XML files for fine-tuning.

Reads XML files from ecg_xml/ directory, matches them to labels in an Excel file,
extracts 12-lead ECG signals, resamples to 400Hz/4096 samples, and saves as HDF5.

IMPORTANT: Splits are done at the PATIENT level to prevent data leakage.
All ECGs from the same patient stay in the same split (train/val/test).

Usage:
    python prepare_data.py --excel labels.xlsx --xml_dir ecg_xml/ --output_dir prepared_data/

Output:
    prepared_data/
        ecg_tracings.hdf5       # ECG signals, shape (N, 4096, 12)
        labels.csv              # Labels matching HDF5 row order
        train_labels.csv        # Training set labels
        val_labels.csv          # Validation set labels
        test_labels.csv         # Test set labels
        split_info.csv          # Which patient/ECG is in which split
"""

import argparse
import base64
import glob
import os
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample


# Lead order expected by the model (12 leads)
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Target parameters (matching the pre-trained model)
TARGET_FREQ = 400   # Hz
TARGET_SAMPLES = 4096


def decode_waveform(b64_data, n_samples):
    """Decode base64-encoded 16-bit waveform data from GE MUSE XML.

    Parameters
    ----------
    b64_data : str
        Base64-encoded waveform string from XML.
    n_samples : int
        Expected number of samples.

    Returns
    -------
    np.ndarray
        Decoded signal as array of int16 values.
    """
    # Remove whitespace from base64 string
    b64_clean = b64_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
    raw_bytes = base64.b64decode(b64_clean)
    # 16-bit signed little-endian
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(samples) != n_samples:
        print(f"  Warning: expected {n_samples} samples, got {len(samples)}")
    return samples.astype(np.float64)


def parse_muse_xml(xml_path):
    """Parse a GE MUSE ECG XML file and extract waveform data + metadata.

    Parameters
    ----------
    xml_path : str
        Path to the XML file.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - 'record_id': patient record ID
        - 'acquisition_time': time string (HHMMSS)
        - 'acquisition_date': date string (MMDDYY)
        - 'age': patient age
        - 'gender': patient gender
        - 'leads': dict mapping lead name to signal array (physical units, microvolts)
        - 'sample_freq': original sample frequency
        - 'n_samples': number of samples per lead
        Returns None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  XML parse error in {xml_path}: {e}")
        return None

    # Extract patient demographics
    demographics = root.find('.//PatientDemographics')
    if demographics is None:
        print(f"  No PatientDemographics in {xml_path}")
        return None

    record_id = demographics.findtext('Record_ID', '').strip()
    age = demographics.findtext('PatientAge', '').strip()
    gender = demographics.findtext('Gender', '').strip()

    # Extract acquisition info
    test_demo = root.find('.//TestDemographics')
    if test_demo is None:
        print(f"  No TestDemographics in {xml_path}")
        return None

    acq_time = test_demo.findtext('AcquisitionTime', '').strip()
    acq_date = test_demo.findtext('AcquisitionDate', '').strip()

    # Convert time: "13:19:05" -> "131905"
    acq_time_clean = acq_time.replace(':', '')

    # Convert date: "11-07-2020" (DD-MM-YYYY European) -> "MMDDYY"
    # Handle multiple date formats
    acq_date_mmddyy = ''
    if acq_date:
        parts = acq_date.replace('/', '-').split('-')
        if len(parts) == 3:
            dd, mm, yyyy = parts[0], parts[1], parts[2]
            yy = yyyy[-2:]  # last 2 digits of year
            acq_date_mmddyy = f"{mm}{dd}{yy}"

    # Find the Rhythm waveform (not Median) - it has the full 10s recording
    rhythm_waveform = None
    for waveform in root.findall('.//Waveform'):
        wf_type = waveform.findtext('WaveformType', '')
        if wf_type == 'Rhythm':
            rhythm_waveform = waveform
            break

    if rhythm_waveform is None:
        print(f"  No Rhythm waveform found in {xml_path}")
        return None

    # Get sample rate
    sample_base = int(rhythm_waveform.findtext('SampleBase', '500'))
    sample_exp = int(rhythm_waveform.findtext('SampleExponent', '0'))
    sample_freq = sample_base * (10 ** sample_exp)

    # Parse each lead
    leads = {}
    for lead_data in rhythm_waveform.findall('.//LeadData'):
        lead_id = lead_data.findtext('LeadID', '').strip()
        n_samples = int(lead_data.findtext('LeadSampleCountTotal', '0'))
        waveform_data = lead_data.findtext('WaveFormData', '')

        # Get amplitude conversion factor
        # LeadAmplitudeUnitsPerBit can use comma as decimal separator (European)
        amp_str = lead_data.findtext('LeadAmplitudeUnitsPerBit', '1').replace(',', '.')
        amp_per_bit = float(amp_str)  # microvolts per bit

        if not waveform_data or n_samples == 0:
            continue

        # Decode and convert to microvolts
        signal = decode_waveform(waveform_data, n_samples)
        signal_uv = signal * amp_per_bit  # now in microvolts

        leads[lead_id] = signal_uv

    if not leads:
        print(f"  No lead data extracted from {xml_path}")
        return None

    # Get n_samples from first lead
    first_lead = next(iter(leads.values()))

    return {
        'record_id': record_id,
        'acquisition_time': acq_time_clean,
        'acquisition_date': acq_date_mmddyy,
        'age': age,
        'gender': gender,
        'leads': leads,
        'sample_freq': sample_freq,
        'n_samples': len(first_lead),
        'xml_path': xml_path,
    }


def derive_12_leads(leads_dict):
    """Derive the full 12-lead ECG from 8 stored leads.

    GE MUSE stores 8 leads (I, II, V1-V6). The remaining 4 are derived:
        III  = II - I
        aVR  = -(I + II) / 2
        aVL  = I - II / 2
        aVF  = II - I / 2

    Parameters
    ----------
    leads_dict : dict
        Mapping lead name -> signal array for 8 stored leads.

    Returns
    -------
    np.ndarray
        Shape (n_samples, 12) with leads in standard order.
    """
    lead_I = leads_dict['I']
    lead_II = leads_dict['II']

    # Derive augmented leads
    lead_III = lead_II - lead_I
    lead_aVR = -(lead_I + lead_II) / 2.0
    lead_aVL = lead_I - lead_II / 2.0
    lead_aVF = lead_II - lead_I / 2.0

    all_leads = {
        'I': lead_I,
        'II': lead_II,
        'III': lead_III,
        'aVR': lead_aVR,
        'aVL': lead_aVL,
        'aVF': lead_aVF,
        'V1': leads_dict['V1'],
        'V2': leads_dict['V2'],
        'V3': leads_dict['V3'],
        'V4': leads_dict['V4'],
        'V5': leads_dict['V5'],
        'V6': leads_dict['V6'],
    }

    n_samples = len(lead_I)
    ecg_matrix = np.zeros((n_samples, 12))
    for i, lead_name in enumerate(LEAD_ORDER):
        ecg_matrix[:, i] = all_leads[lead_name]

    return ecg_matrix


def resample_and_pad(ecg_matrix, orig_freq, target_freq=TARGET_FREQ,
                     target_samples=TARGET_SAMPLES):
    """Resample ECG signal and zero-pad to target length.

    Parameters
    ----------
    ecg_matrix : np.ndarray
        Shape (n_samples, 12).
    orig_freq : int
        Original sampling frequency in Hz.
    target_freq : int
        Target sampling frequency in Hz.
    target_samples : int
        Target number of samples (with padding).

    Returns
    -------
    np.ndarray
        Shape (target_samples, 12), resampled and padded.
    """
    n_orig = ecg_matrix.shape[0]
    duration_s = n_orig / orig_freq
    n_resampled = int(round(duration_s * target_freq))

    # Resample each lead
    resampled = resample(ecg_matrix, n_resampled, axis=0)

    # Pad symmetrically with zeros to reach target_samples
    if n_resampled >= target_samples:
        # If already long enough, take center portion
        start = (n_resampled - target_samples) // 2
        return resampled[start:start + target_samples, :]
    else:
        pad_total = target_samples - n_resampled
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(resampled, ((pad_left, pad_right), (0, 0)), mode='constant')


def convert_to_model_units(ecg_matrix_uv):
    """Convert ECG from microvolts to model input units.

    The pre-trained model expects signals at a scale of 1e-4 V (= 100 uV).
    So we divide microvolts by 100.

    Parameters
    ----------
    ecg_matrix_uv : np.ndarray
        ECG data in microvolts.

    Returns
    -------
    np.ndarray
        ECG data in model units (1 unit = 100 uV = 0.1 mV).
    """
    return ecg_matrix_uv / 100.0


def construct_record_id(record_id, acq_time, acq_date):
    """Construct the full Record_ID matching the Excel format.

    Excel format: {patientID}_{HHMMSS}_{MMDDYY}

    Parameters
    ----------
    record_id : str
        Patient record ID from XML.
    acq_time : str
        Acquisition time as HHMMSS.
    acq_date : str
        Acquisition date as MMDDYY.

    Returns
    -------
    str
        Full Record_ID string.
    """
    return f"{record_id}_{acq_time}_{acq_date}"


def extract_patient_id(record_id_full):
    """Extract patient ID from full Record_ID.

    The patient ID is the first part before the first underscore.

    Parameters
    ----------
    record_id_full : str
        Full Record_ID like '110005_082739_121623'.

    Returns
    -------
    str
        Patient ID like '110005'.
    """
    return record_id_full.split('_')[0]


def patient_level_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                        random_state=42):
    """Split data at the patient level to prevent data leakage.

    All ECGs from the same patient are assigned to the same split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'patient_id' and 'label' columns.
    train_ratio : float
        Fraction for training set.
    val_ratio : float
        Fraction for validation set.
    test_ratio : float
        Fraction for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (train_indices, val_indices, test_indices)
        Integer indices into df.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    rng = np.random.RandomState(random_state)

    # Get unique patients and shuffle
    patients = df['patient_id'].unique()
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = int(round(n_patients * train_ratio))
    n_val = int(round(n_patients * val_ratio))

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_idx = df.index[df['patient_id'].isin(train_patients)].tolist()
    val_idx = df.index[df['patient_id'].isin(val_patients)].tolist()
    test_idx = df.index[df['patient_id'].isin(test_patients)].tolist()

    print(f"\nPatient-level split (seed={random_state}):")
    print(f"  Train: {len(train_patients)} patients, {len(train_idx)} ECGs")
    print(f"  Val:   {len(val_patients)} patients, {len(val_idx)} ECGs")
    print(f"  Test:  {len(test_patients)} patients, {len(test_idx)} ECGs")

    # Check label distribution per split
    for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        labels = df.loc[idx, 'label']
        pos = labels.sum()
        neg = len(labels) - pos
        print(f"  {name} labels: {int(pos)} positive, {int(neg)} negative "
              f"({100*pos/len(labels):.1f}% positive)")

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ECG data from GE MUSE XML for fine-tuning.')
    parser.add_argument('--excel', type=str, default='labels.xlsx',
                        help='Path to Excel file with Record_ID and label columns')
    parser.add_argument('--xml_dir', type=str, default='ecg_xml',
                        help='Directory containing MUSE XML files')
    parser.add_argument('--output_dir', type=str, default='prepared_data',
                        help='Output directory (default: prepared_data/)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split (default: 42)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Load Excel labels ---
    print("=" * 60)
    print("Step 1: Loading labels from Excel")
    print("=" * 60)

    if args.excel.endswith('.xlsx') or args.excel.endswith('.xls'):
        labels_df = pd.read_excel(args.excel)
    else:
        # Auto-detect separator (comma or semicolon)
        labels_df = pd.read_csv(args.excel, sep=None, engine='python')

    # Ensure Record_ID is string
    labels_df['Record_ID'] = labels_df['Record_ID'].astype(str).str.strip()
    print(f"Loaded {len(labels_df)} records from {args.excel}")
    print(f"Columns: {list(labels_df.columns)}")
    print(f"Label distribution: {labels_df['label'].value_counts().to_dict()}")

    # Extract patient IDs for leakage prevention
    labels_df['patient_id'] = labels_df['Record_ID'].apply(extract_patient_id)
    n_unique_patients = labels_df['patient_id'].nunique()
    n_ecgs = len(labels_df)
    multi_ecg = labels_df.groupby('patient_id').size()
    multi_ecg_patients = multi_ecg[multi_ecg > 1]
    print(f"\nUnique patients: {n_unique_patients}")
    print(f"Total ECGs: {n_ecgs}")
    if len(multi_ecg_patients) > 0:
        print(f"Patients with multiple ECGs: {len(multi_ecg_patients)}")
        for pid, count in multi_ecg_patients.items():
            matching = labels_df[labels_df['patient_id'] == pid]['Record_ID'].tolist()
            print(f"  Patient {pid}: {count} ECGs -> {matching}")

    # Build lookup: Record_ID -> row index
    label_lookup = dict(zip(labels_df['Record_ID'], labels_df.index))

    # --- Step 2: Parse XML files ---
    print("\n" + "=" * 60)
    print("Step 2: Parsing XML files")
    print("=" * 60)

    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, '*.xml')))
    if not xml_files:
        # Also try recursive search
        xml_files = sorted(glob.glob(os.path.join(args.xml_dir, '**', '*.xml'),
                                     recursive=True))
    print(f"Found {len(xml_files)} XML files in {args.xml_dir}")

    if len(xml_files) == 0:
        print("ERROR: No XML files found. Check the --xml_dir path.")
        return

    # Parse each XML and try to match with Excel
    parsed_ecgs = []
    matched = 0
    unmatched_xml = []

    for i, xml_path in enumerate(xml_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Parsing {i+1}/{len(xml_files)}...")

        result = parse_muse_xml(xml_path)
        if result is None:
            continue

        # Construct Record_ID to match Excel
        constructed_id = construct_record_id(
            result['record_id'],
            result['acquisition_time'],
            result['acquisition_date']
        )

        # Try to match with Excel
        if constructed_id in label_lookup:
            idx = label_lookup[constructed_id]
            label = labels_df.loc[idx, 'label']
            patient_id = labels_df.loc[idx, 'patient_id']
            result['full_record_id'] = constructed_id
            result['label'] = label
            result['patient_id'] = patient_id
            parsed_ecgs.append(result)
            matched += 1
        else:
            # Try matching by filename (without extension)
            fname = os.path.splitext(os.path.basename(xml_path))[0]
            if fname in label_lookup:
                idx = label_lookup[fname]
                label = labels_df.loc[idx, 'label']
                patient_id = labels_df.loc[idx, 'patient_id']
                result['full_record_id'] = fname
                result['label'] = label
                result['patient_id'] = patient_id
                parsed_ecgs.append(result)
                matched += 1
            else:
                unmatched_xml.append({
                    'xml_path': xml_path,
                    'constructed_id': constructed_id,
                    'filename': fname,
                    'record_id': result['record_id'],
                })

    print(f"\nMatched: {matched}/{len(xml_files)} XML files to Excel labels")

    if unmatched_xml:
        print(f"Unmatched: {len(unmatched_xml)} XML files")
        for u in unmatched_xml[:5]:
            print(f"  {u['xml_path']} -> constructed ID: {u['constructed_id']}")
        if len(unmatched_xml) > 5:
            print(f"  ... and {len(unmatched_xml) - 5} more")

    # Check which Excel records were NOT found in XMLs
    matched_ids = {e['full_record_id'] for e in parsed_ecgs}
    missing_from_xml = labels_df[~labels_df['Record_ID'].isin(matched_ids)]
    if len(missing_from_xml) > 0:
        print(f"\nExcel records without matching XML: {len(missing_from_xml)}")
        for _, row in missing_from_xml.head(5).iterrows():
            print(f"  {row['Record_ID']}")

    if not parsed_ecgs:
        print("\nERROR: No ECGs matched. Check Record_ID format or XML directory.")
        print("Expected Excel Record_ID format: patientID_HHMMSS_MMDDYY")
        print("XML provides: Record_ID + AcquisitionTime + AcquisitionDate")
        return

    # --- Step 3: Process waveforms ---
    print("\n" + "=" * 60)
    print("Step 3: Processing waveforms (12-lead extraction, resampling)")
    print("=" * 60)

    n_ecgs = len(parsed_ecgs)
    ecg_data = np.zeros((n_ecgs, TARGET_SAMPLES, 12), dtype=np.float32)
    record_ids = []
    patient_ids = []
    ecg_labels = []

    for i, ecg in enumerate(parsed_ecgs):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{n_ecgs}: {ecg['full_record_id']}")

        # Check required leads are present
        required_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        missing_leads = [l for l in required_leads if l not in ecg['leads']]
        if missing_leads:
            print(f"  WARNING: Missing leads {missing_leads} in {ecg['full_record_id']}, "
                  f"skipping")
            continue

        # Derive 12 leads from 8 stored
        ecg_12lead = derive_12_leads(ecg['leads'])

        # Resample from original freq to 400Hz and pad to 4096
        ecg_resampled = resample_and_pad(ecg_12lead, ecg['sample_freq'])

        # Convert from microvolts to model units (1 unit = 100 uV)
        ecg_model = convert_to_model_units(ecg_resampled)

        ecg_data[i] = ecg_model.astype(np.float32)
        record_ids.append(ecg['full_record_id'])
        patient_ids.append(ecg['patient_id'])
        ecg_labels.append(ecg['label'])

    # Trim if some ECGs were skipped
    actual_n = len(record_ids)
    ecg_data = ecg_data[:actual_n]

    print(f"\nSuccessfully processed {actual_n} ECGs")
    print(f"ECG data shape: {ecg_data.shape}")

    # --- Step 4: Patient-level split ---
    print("\n" + "=" * 60)
    print("Step 4: Patient-level train/val/test split")
    print("=" * 60)

    info_df = pd.DataFrame({
        'Record_ID': record_ids,
        'patient_id': patient_ids,
        'label': ecg_labels,
    })

    train_idx, val_idx, test_idx = patient_level_split(
        info_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    # --- Step 5: Save outputs ---
    print("\n" + "=" * 60)
    print("Step 5: Saving output files")
    print("=" * 60)

    # Save full HDF5 (all data, ordered: train, val, test)
    all_indices = train_idx + val_idx + test_idx
    ecg_ordered = ecg_data[all_indices]

    hdf5_path = os.path.join(args.output_dir, 'ecg_tracings.hdf5')
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('tracings', data=ecg_ordered, dtype='float32',
                         compression='gzip', compression_opts=4)
    print(f"Saved HDF5: {hdf5_path} (shape: {ecg_ordered.shape})")

    # Save labels CSV (matching HDF5 row order)
    labels_ordered = info_df.loc[all_indices, 'label'].values
    labels_csv = pd.DataFrame({'label': labels_ordered})
    labels_csv_path = os.path.join(args.output_dir, 'labels.csv')
    labels_csv.to_csv(labels_csv_path, index=False)
    print(f"Saved labels: {labels_csv_path}")

    # Save per-split label CSVs (for use with ECGSequence)
    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)

    train_labels = pd.DataFrame({'label': labels_ordered[:n_train]})
    val_labels = pd.DataFrame({'label': labels_ordered[n_train:n_train + n_val]})
    test_labels = pd.DataFrame({'label': labels_ordered[n_train + n_val:]})

    train_labels.to_csv(os.path.join(args.output_dir, 'train_labels.csv'), index=False)
    val_labels.to_csv(os.path.join(args.output_dir, 'val_labels.csv'), index=False)
    test_labels.to_csv(os.path.join(args.output_dir, 'test_labels.csv'), index=False)

    # Save split info for traceability
    info_df['split'] = ''
    info_df.loc[train_idx, 'split'] = 'train'
    info_df.loc[val_idx, 'split'] = 'val'
    info_df.loc[test_idx, 'split'] = 'test'
    split_info_path = os.path.join(args.output_dir, 'split_info.csv')
    info_df.to_csv(split_info_path, index=False)
    print(f"Saved split info: {split_info_path}")

    # Print summary for fine-tuning command
    print("\n" + "=" * 60)
    print("DONE! Summary:")
    print("=" * 60)
    print(f"  Total ECGs:      {actual_n}")
    print(f"  Train ECGs:      {n_train} (rows 0-{n_train-1} in HDF5)")
    print(f"  Validation ECGs: {n_val} (rows {n_train}-{n_train+n_val-1} in HDF5)")
    print(f"  Test ECGs:       {n_test} (rows {n_train+n_val}-{actual_n-1} in HDF5)")
    print(f"\n  HDF5 shape: {ecg_ordered.shape}")
    print(f"  Lead order: {LEAD_ORDER}")

    val_split_fraction = n_val / (n_train + n_val) if (n_train + n_val) > 0 else 0
    print(f"\nTo fine-tune (train+val only), run:")
    print(f"  python finetune_cardiac_remodeling.py \\")
    print(f"    {hdf5_path} {labels_csv_path} \\")
    print(f"    --pretrained_model model.hdf5 \\")
    print(f"    --n_classes 1 \\")
    print(f"    --val_split {val_split_fraction:.4f} \\")
    print(f"    --epochs 50 --batch_size 32 --lr 0.0001")


if __name__ == '__main__':
    main()
