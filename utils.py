import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config

# Utility functions for ECG data loading, beat segmentation, and evaluation.

# MIT-BIH beat annotation symbols.
# Normal symbols follow the AAMI class N grouping (normal, LBBB, RBBB, atrial
# escape, nodal/junctional escape); every other beat symbol is treated as an
# anomaly.
MIT_BIH_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r",
    "F", "e", "j", "n", "E", "/", "f", "Q", "?"
}
MIT_BIH_NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}


def normalize_signal(signal):
    """Min-max normalize an ECG signal to the [0, 1] range."""
    signal = np.asarray(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)


def load_mit_bih_record(record_path, record_name):
    """
    Load a single MIT-BIH record (signal and annotations).

    Args:
        record_path: Path to the MIT-BIH database directory
        record_name: Record number as string (e.g., '100', '101')

    Returns:
        Dictionary with signal, annotations, and metadata (or None on error).
    """
    try:
        signal, fields = wfdb.rdsamp(str(Path(record_path) / record_name))
        annotation = wfdb.rdann(str(Path(record_path) / record_name), 'atr')

        return {
            'record': record_name,
            'signal': signal,
            'fs': fields['fs'],
            'sig_name': fields['sig_name'],
            'n_sig': fields['n_sig'],
            'n_samples': fields.get('sig_len', len(signal)),
            'anno_sample': annotation.sample,
            'anno_symbol': annotation.symbol,
        }
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None


def read_record_names(record_path, drop_paced=None):
    """
    Read the list of record names from the MIT-BIH RECORDS file.

    Args:
        record_path: Path to the MIT-BIH database directory
        drop_paced: If True, paced records are excluded (default: config.DROP_PACED)

    Returns:
        List of record name strings
    """
    if drop_paced is None:
        drop_paced = config.DROP_PACED

    records_file = Path(record_path) / 'RECORDS'

    if not records_file.exists():
        raise FileNotFoundError(f"RECORDS file not found at {records_file}")

    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines() if line.strip()]

    if drop_paced:
        record_names = [r for r in record_names if r not in config.PACED_RECORDS]

    return record_names


def load_all_mit_bih_records(record_path, drop_paced=None):
    """
    Load all MIT-BIH records from a directory.

    Args:
        record_path: Path to the MIT-BIH database directory
        drop_paced: If True, paced records are excluded (default: config.DROP_PACED)

    Returns:
        List of dictionaries with all records
    """
    record_names = read_record_names(record_path, drop_paced=drop_paced)

    records = []
    for record_name in record_names:
        print(f"Loading record {record_name}...")
        record = load_mit_bih_record(record_path, record_name)
        if record is not None:
            records.append(record)

    return records


def find_r_peaks(signal, fs=None):
    """
    Detect R-peaks in an ECG signal using simple peak detection.

    The model pipeline segments beats from the dataset's R-peak *annotations*
    instead; this helper is kept for illustration (see notebook 01).

    Args:
        signal: ECG signal (1D array)
        fs: Sampling rate in Hz (default: config.FS)

    Returns:
        Array of R-peak indices
    """
    if fs is None:
        fs = config.FS

    signal = np.asarray(signal)

    # Positive peaks, at least ~0.4 s apart, above 30% of the signal maximum.
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(signal, distance=min_distance, height=np.max(signal) * 0.3)

    return peaks


def is_mit_bih_anomaly(symbol, normal_symbols=None):
    """
    Return True if a MIT-BIH beat annotation should be treated as anomalous.
    Non-beat annotations should be filtered out before calling this function.
    """
    if normal_symbols is None:
        normal_symbols = MIT_BIH_NORMAL_SYMBOLS

    return symbol not in normal_symbols


def segment_record_by_annotations(record, channel=0, half_window=None, normalize=True):
    """
    Segment one MIT-BIH record using the official beat annotation samples.

    Returns a DataFrame with one row per valid beat. The label is numeric:
    0 = normal, 1 = anomaly.
    """
    if half_window is None:
        half_window = config.HALF_WINDOW

    signal = np.asarray(record['signal'])[:, channel]
    rows = []

    for sample, symbol in zip(record['anno_sample'], record['anno_symbol']):
        if symbol not in MIT_BIH_BEAT_SYMBOLS:
            continue

        start = sample - half_window
        end = sample + half_window

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]
        if normalize:
            beat = normalize_signal(beat)

        is_anomaly = is_mit_bih_anomaly(symbol)

        rows.append({
            'record': record['record'],
            'sample': int(sample),
            'symbol': symbol,
            'beat_index': len(rows),
            'signal': beat,
            'label': int(is_anomaly),
            'label_name': 'anomaly' if is_anomaly else 'normal',
        })

    return pd.DataFrame(rows)


def create_beat_dataset(records, normalize=True):
    """
    Create a dataset of individual beats from MIT-BIH records.

    Each beat is segmented around its annotated R-peak
    (see segment_record_by_annotations).

    Args:
        records: List of record dictionaries (from load_mit_bih_record).
        normalize: Whether to min-max normalize each beat.

    Returns:
        DataFrame with one row per beat (beats, labels, metadata); empty if none.
    """
    datasets = [
        segment_record_by_annotations(record, normalize=normalize)
        for record in records
    ]
    datasets = [df for df in datasets if not df.empty]
    if not datasets:
        return pd.DataFrame()
    return pd.concat(datasets, ignore_index=True)


def split_by_record(X, y, metadata, test_size=None, random_state=None):
    """
    Split a beat dataset by record to avoid train/test leakage.

    Beats from the same ECG record are kept entirely in either train or test.
    """
    if test_size is None:
        test_size = config.TEST_SPLIT
    if random_state is None:
        random_state = config.RANDOM_SEED

    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("X, y, and metadata must contain the same number of beats.")
    if 'record' not in metadata.columns:
        raise ValueError("metadata must contain a 'record' column.")

    records = np.array(sorted(metadata['record'].unique()))
    if len(records) < 2:
        raise ValueError("At least two records are needed for a train/test split by record.")

    rng = np.random.default_rng(random_state)
    rng.shuffle(records)

    n_test = max(1, int(round(len(records) * test_size)))
    n_test = min(n_test, len(records) - 1)
    test_records = set(records[:n_test])

    is_test = metadata['record'].isin(test_records).to_numpy()
    is_train = ~is_test

    return {
        'X_train': X[is_train],
        'X_test': X[is_test],
        'y_train': y[is_train],
        'y_test': y[is_test],
        'metadata_train': metadata.loc[is_train].reset_index(drop=True),
        'metadata_test': metadata.loc[is_test].reset_index(drop=True),
        'train_records': sorted(set(records) - test_records),
        'test_records': sorted(test_records),
    }


def split_by_record_lists(X, y, metadata, train_records, test_records):
    """
    Split a beat dataset using explicit lists of train and test record names.

    Beats are assigned to train or test according to which record they belong to,
    following a fixed inter-patient partition (e.g. the De Chazal DS1/DS2 split in
    config.DE_CHAZAL_DS1 / DE_CHAZAL_DS2). Records not in either list are dropped.

    This complements split_by_record, which assigns records to train/test at
    random. Use this function when a reproducible, standard partition is needed.
    """
    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("X, y, and metadata must contain the same number of beats.")
    if 'record' not in metadata.columns:
        raise ValueError("metadata must contain a 'record' column.")

    train_records = {str(r) for r in train_records}
    test_records = {str(r) for r in test_records}

    overlap = train_records & test_records
    if overlap:
        raise ValueError(f"Records appear in both train and test: {sorted(overlap)}")

    record_col = metadata['record'].astype(str)
    is_train = record_col.isin(train_records).to_numpy()
    is_test = record_col.isin(test_records).to_numpy()

    return {
        'X_train': X[is_train],
        'X_test': X[is_test],
        'y_train': y[is_train],
        'y_test': y[is_test],
        'metadata_train': metadata.loc[is_train].reset_index(drop=True),
        'metadata_test': metadata.loc[is_test].reset_index(drop=True),
        'train_records': sorted(train_records),
        'test_records': sorted(test_records),
    }


def evaluate_anomaly_detection(y_true, y_pred, scores=None):
    """
    Compute common metrics for binary anomaly detection.

    Labels follow the project convention:
    0 = normal, 1 = anomaly.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if scores is not None and len(np.unique(y_true)) == 2:
        scores = np.asarray(scores)
        results['roc_auc'] = roc_auc_score(y_true, scores)
        results['pr_auc'] = average_precision_score(y_true, scores)
    else:
        results['roc_auc'] = np.nan
        results['pr_auc'] = np.nan

    return results
