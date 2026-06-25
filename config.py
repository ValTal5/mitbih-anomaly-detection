"""
Configuration settings for the ECG Anomaly Detection project.

Holds constants for signal processing, beat segmentation, the dataset, and the
train/test split. Model hyperparameters (batch size, epochs, learning rate) are
not kept here: they are passed explicitly in each notebook, because different
experiments use different values.
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
# DATA
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

# De Chazal et al. (2004) inter-patient split (paced records already excluded).
# DS1 is used for training/validation, DS2 for testing. Keeping records disjoint
# between the two sets avoids patient-level leakage.
DE_CHAZAL_DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DE_CHAZAL_DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]

# ============================================================================
# EXPERIMENT
# ============================================================================

# Fraction of records assigned to the test set by the random `split_by_record`.
# (The full-dataset experiments use the explicit De Chazal DS1/DS2 lists via
# `split_by_record_lists` instead.)
TEST_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# ARMA BASELINE
# ============================================================================

ARIMA_P = 1
ARIMA_D = 0
ARIMA_Q = 1
