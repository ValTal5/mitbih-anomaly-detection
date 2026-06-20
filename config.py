"""
Configuration settings for ECG Anomaly Detection project.
Contains constants for signal processing, segmentation, and model training.
"""

import os

# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

# Sampling rate (Hz) - MIT-BIH standard
FS = 360

# ============================================================================
# SEGMENTATION
# ============================================================================

# Half-window size in samples (250 ms at 360 Hz)
HALF_WINDOW = 90

# Full beat length in samples (500 ms)
BEAT_LEN = 2 * HALF_WINDOW

# ============================================================================
# DATA PATHS
# ============================================================================

# Path to MIT-BIH database.
# Override this on another machine with the MIT_BIH_PATH environment variable.
MIT_BIH_PATH = os.getenv(
    "MIT_BIH_PATH",
    r"C:\Users\Admin\Downloads\mit-bih-arrhythmia-database-1.0.0",
)

# Paced records (pacemaker). Their beats have an artificial morphology driven
# by the device, not by intrinsic cardiac activity, so they are excluded
# following the De Chazal et al. (2004) standard.
# Dropping them also removes the only records containing the '/' (paced) and
# 'f' (fusion of paced and normal) beat symbols.
PACED_RECORDS = {"102", "104", "107", "217"}

# If True, paced records are skipped when loading the database.
DROP_PACED = True

# Data directories
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"

# ============================================================================
# MODEL TRAINING
# ============================================================================

# Train/test split
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2

# Anomaly contamination rate
CONTAMINATION_RATE = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# ARIMA/ARMA PARAMETERS
# ============================================================================

ARIMA_P = 1
ARIMA_D = 0
ARIMA_Q = 1
ARIMA_THRESHOLD_PERCENTILE = 95

# ============================================================================
# NEURAL NETWORK PARAMETERS
# ============================================================================

# Batch size
BATCH_SIZE = 32

# Epochs
EPOCHS = 100

# Learning rate
LEARNING_RATE = 1e-3

# Validation split
VALIDATION_SPLIT = 0.2
