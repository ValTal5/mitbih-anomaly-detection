import os
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

# Utility functions for ECG data preprocessing and visualization

# MIT-BIH beat annotations.
# Normal symbols follow the common AAMI grouping used in many ECG studies.
MIT_BIH_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r",
    "F", "e", "j", "n", "E", "/", "f", "Q", "?"
}
MIT_BIH_NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}

def load_ecg_data(path):
    """
    Load ECG data from a CSV file or directory.
    Returns a pandas DataFrame.
    """
    if os.path.isdir(path):
        # Load all CSV files in the directory and concatenate
        dfs = []
        for file in os.listdir(path):
            if file.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(path, file)))
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No CSV files found in the directory.")
    elif os.path.isfile(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Invalid path: {path}")

def normalize_signal(signal):
    """
    Normalize an ECG signal between 0 and 1.
    """
    signal = np.asarray(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

def plot_ecg(signal, title="ECG Signal", figsize=(12, 4)):
    """
    Plot an ECG signal using matplotlib.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.plot(signal, lw=1)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def load_mit_bih_record(record_path, record_name):
    """
    Load a single MIT-BIH record (signal and annotations).
    
    Args:
        record_path: Path to the MIT-BIH database directory
        record_name: Record number as string (e.g., '100', '101')
    
    Returns:
        Dictionary with signal, annotations, and metadata
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

def extract_ecg_features(signal, sampling_rate):
    """
    Extract basic features from ECG signal.
    
    Args:
        signal: ECG signal (1D array)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary with computed features
    """
    signal = np.asarray(signal)
    
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'rms': np.sqrt(np.mean(signal ** 2)),
        'duration_sec': len(signal) / sampling_rate,
    }
    
    return features

def find_r_peaks(signal, fs=None):
    """
    Detect R-peaks in ECG signal using peak detection.
    
    Args:
        signal: ECG signal (1D array)
        fs: Sampling rate in Hz (default: config.FS)
    
    Returns:
        Array of R-peak indices
    """
    if fs is None:
        fs = config.FS
    
    signal = np.asarray(signal)
    
    # Find peaks with adaptive threshold
    # Look for positive peaks, minimum distance between peaks (~0.4 seconds)
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(signal, distance=min_distance, height=np.max(signal) * 0.3)
    
    return peaks

def segment_signal_by_beats(signal, r_peaks, fs=None, half_window=None):
    """
    Segment ECG signal into individual beats around R-peaks.
    
    Args:
        signal: ECG signal (1D array)
        r_peaks: Array of R-peak indices
        fs: Sampling rate in Hz (default: config.FS)
        half_window: Half-window size in samples (default: config.HALF_WINDOW)
    
    Returns:
        List of beat segments, list of valid indices
    """
    if fs is None:
        fs = config.FS
    if half_window is None:
        half_window = config.HALF_WINDOW
    
    signal = np.asarray(signal)
    beats = []
    valid_indices = []
    
    for i, r_peak in enumerate(r_peaks):
        start = r_peak - half_window
        end = r_peak + half_window
        
        # Skip if window goes out of bounds
        if start < 0 or end > len(signal):
            continue
        
        beat = signal[start:end]
        beats.append(beat)
        valid_indices.append(i)
    
    return beats, valid_indices

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

def create_beat_dataset(records, fs=None, normalize=True, use_annotations=True):
    """
    Create dataset of individual beats from MIT-BIH records.
    
    Args:
        records: List of record dictionaries
        fs: Sampling rate in Hz (default: config.FS)
        normalize: Whether to normalize beats
        use_annotations: If True, use MIT-BIH annotation samples as beat centers.
    
    Returns:
        DataFrame with beats, labels, and metadata
    """
    if fs is None:
        fs = config.FS
    
    if use_annotations:
        datasets = [
            segment_record_by_annotations(record, normalize=normalize)
            for record in records
        ]
        datasets = [df for df in datasets if not df.empty]
        if not datasets:
            return pd.DataFrame()
        return pd.concat(datasets, ignore_index=True)

    dataset = []
    
    for record in records:
        signal = record['signal'][:, 0]  # First channel
        
        # Find R-peaks
        r_peaks = find_r_peaks(signal, fs)
        
        # Segment into beats
        beats, valid_indices = segment_signal_by_beats(signal, r_peaks, fs)
        
        anno_samples = np.asarray(record['anno_sample'])
        anno_symbols = np.asarray(record['anno_symbol'])

        for beat_idx, (beat, peak_idx) in enumerate(zip(beats, valid_indices)):
            r_peak = r_peaks[peak_idx]

            distances = np.abs(anno_samples - r_peak)
            nearest_idx = int(np.argmin(distances))
            nearest_symbol = anno_symbols[nearest_idx]

            if distances[nearest_idx] >= config.HALF_WINDOW:
                continue
            if nearest_symbol not in MIT_BIH_BEAT_SYMBOLS:
                continue

            is_anomaly = is_mit_bih_anomaly(nearest_symbol)
            
            if normalize:
                beat = normalize_signal(beat)
            
            dataset.append({
                'record': record['record'],
                'sample': int(r_peak),
                'symbol': nearest_symbol,
                'beat_index': beat_idx,
                'signal': beat,
                'label': int(is_anomaly),
                'label_name': 'anomaly' if is_anomaly else 'normal',
            })
    
    return pd.DataFrame(dataset)

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
