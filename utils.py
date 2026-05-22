import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path

# Utility functions for ECG data preprocessing and visualization

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
            'n_samples': fields['n_samples'],
            'anno_sample': annotation.sample,
            'anno_symbol': annotation.symbol,
        }
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None

def load_all_mit_bih_records(record_path):
    """
    Load all MIT-BIH records from a directory.
    
    Args:
        record_path: Path to the MIT-BIH database directory
    
    Returns:
        List of dictionaries with all records
    """
    records_file = Path(record_path) / 'RECORDS'
    
    if not records_file.exists():
        raise FileNotFoundError(f"RECORDS file not found at {records_file}")
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines()]
    
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
