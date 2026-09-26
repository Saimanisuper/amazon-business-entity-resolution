# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Overview & Architecture

The pipeline resolves multi-source business records into unified entities across heterogeneous and noisy data sources (`Source 1`, `Source 2`, and `Source 3`).

### Multi-Stage Pipeline:
1. **Preprocessing & Text Normalization:**
   - Strips legal corporate suffixes (*Inc, Corp, LLC, Pvt Ltd, Sarl, SCI, SA*).
   - Unifies accents and diacritics via Unicode `NFKD` decomposition (generalizes zero-shot to French test entities).
   - Normalizes address abbreviations (*st -> street, rd -> road, r. -> rue, etc.*).
2. **Multi-Index Blocking (Candidate Generation):**
   - High-recall inverted token index with dynamic stop-word pruning.
   - Restricts search space dynamically by country while supporting open-set country additions (`France`).
   - Generates top candidates per Source 1 query record to populate `output/candidate_pairs.tsv`.
3. **Pairwise Feature Engineering:**
   - Name similarity: Jaccard token similarity, Sørensen-Dice, character 3-gram overlap, SequenceMatcher ratio, exact match, first-token match.
   - Address similarity: Token Jaccard, character n-grams, numeric overlap (street numbers, postal codes).
   - Compound interactions: Combined token set Jaccard, geometric mean similarities.
4. **Matching & Threshold Calibration:**
   - Gradient Boosted Decision Tree (XGBoost) trained on balanced candidate pairs.
   - Direct optimization of the official macro $F_{0.5}$ metric on a held-out validation split.
   - Precision-biased thresholding ($\tau^* \ge 0.70$) that penalizes false merges to protect singleton accuracy ($1.0 \to 0.0$ penalty).
   - Produces the final scored submission `output/matching_results.tsv`.

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── requirements.txt           # Pinned python dependencies
├── README.md                  # Reproduction instructions
└── src/
    ├── __init__.py
    ├── config.py              # Path definitions & hyperparameter settings
    ├── preprocess.py          # Multilingual text cleaning & normalization
    ├── blocking.py            # High-throughput candidate generation engine
    ├── features.py            # Pairwise feature extraction
    ├── model.py               # XGBoost matching model & threshold optimizer
    ├── evaluate.py            # Official macro F_0.5 evaluation implementation
    └── pipeline.py            # End-to-end training and inference orchestrator
```

---

## 3. Installation & Setup

Ensure Python 3.8+ is installed:

```bash
pip install -r requirements.txt
```

---

## 4. How to Reproduce

### A. Run Rapid Benchmark / Validation Mode
To run an end-to-end test on a representative subset of records:
```bash
python -m src.pipeline --mode sample --sample-size 10000
```

### B. Run Full-Scale Training and Test Inference
To train the model on the full training set and generate final predictions for all test entities:
```bash
python -m src.pipeline --mode full
```

This generates:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

### C. Validate Submission Format
From the `student_resource/` directory, run the official validation script:
```bash
python utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```
A successful validation outputs `PASS` (exit code 0).
