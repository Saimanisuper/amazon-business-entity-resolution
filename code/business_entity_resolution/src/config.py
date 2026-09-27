import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
STUDENT_RESOURCE_DIR = BASE_DIR / "student_resource"
DATASET_DIR = STUDENT_RESOURCE_DIR / "dataset"
TRAIN_DATA_DIR = DATASET_DIR / "train"
TEST_DATA_DIR = DATASET_DIR / "test"

# Fallback to Resources/ directory if dataset/ is not found
RESOURCES_DIR = BASE_DIR / "Resources"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Training files
TRAIN_SOURCE1 = TRAIN_DATA_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DATA_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DATA_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DATA_DIR / "train_ground_truth.tsv"

# Fallback training paths
if not TRAIN_SOURCE1.exists() and (RESOURCES_DIR / "train1.txt").exists():
    TRAIN_SOURCE1 = RESOURCES_DIR / "train1.txt"
    TRAIN_SOURCE2 = RESOURCES_DIR / "train2.txt"
    TRAIN_SOURCE3 = RESOURCES_DIR / "train3.txt"
    TRAIN_GROUND_TRUTH = RESOURCES_DIR / "traingt.txt"

# Test files
TEST_SOURCE1 = TEST_DATA_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DATA_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DATA_DIR / "test_source3.tsv"

# Fallback test paths
if not TEST_SOURCE1.exists() and (RESOURCES_DIR / "test1.txt").exists():
    TEST_SOURCE1 = RESOURCES_DIR / "test1.txt"
    TEST_SOURCE2 = RESOURCES_DIR / "test2.txt"
    TEST_SOURCE3 = RESOURCES_DIR / "test3.txt"

# Model and Pipeline hyperparameters
BLOCKING_TOP_K = 3           # Focused top candidates per S1 entity for compact file size and high precision
MAX_TRAIN_PAIRS = 500000     # Manageable pair count for training
VAL_SAMPLE_SIZE = 50000      # Validation sample size for threshold tuning
BETA = 0.5                   # F_beta weight
DEFAULT_THRESHOLD = 0.85     # Precision-heavy threshold to eliminate false merges and keep file small
NUM_WORKERS = max(1, os.cpu_count() - 1 if os.cpu_count() else 4)
