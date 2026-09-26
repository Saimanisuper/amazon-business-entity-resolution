import os
import sys
import csv
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .config import (
    BASE_DIR,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    OUTPUT_DIR,
    CANDIDATE_PAIRS_PATH,
    MATCHING_RESULTS_PATH,
    BLOCKING_TOP_K,
    DEFAULT_THRESHOLD,
)
from .blocking import MultiIndexBlocker
from .features import compute_pairwise_features
from .model import EntityMatcherModel
from .evaluate import evaluate_macro_f_beta


def read_source_records(path: Path, max_records: int = None):
    """Generator reading (entity_id, business_name, business_address, country) from a source TSV."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        count = 0
        for row in reader:
            if not row or len(row) < 4:
                continue
            yield row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            count += 1
            if max_records and count >= max_records:
                break


def load_ground_truth_for_s1(path: Path, target_s1_ids: set) -> dict:
    """Load ground truth matches for a specific set of Source 1 IDs."""
    gt = {s1: set() for s1 in target_s1_ids}
    found = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if s1_id in target_s1_ids:
                matches = set()
                if len(row) > 1 and row[1].strip():
                    matches = {m.strip() for m in row[1].split(",") if m.strip()}
                gt[s1_id] = matches
                found += 1
                if found >= len(target_s1_ids):
                    break
    return gt


def train_pipeline(sample_size: int = 15000, model_save_path: Path = None):
    """
    Train ML model using a fast, representative sample of training data.
    """
    print(f"=== Starting Training Pipeline (Sample Size: {sample_size:,}) ===", flush=True)
    start_time = time.time()

    # 1. Read first sample_size records from TRAIN_SOURCE1
    print("1. Loading Source 1 training records...", flush=True)
    s1_records = {}
    for eid, name, addr, country in read_source_records(TRAIN_SOURCE1, max_records=sample_size):
        s1_records[eid] = (name, addr, country)

    needed_s1 = set(s1_records.keys())

    # 2. Load ground truth for these S1 records
    print("2. Matching against ground truth...", flush=True)
    gt = load_ground_truth_for_s1(TRAIN_GROUND_TRUTH, needed_s1)

    needed_targets = set()
    for matches in gt.values():
        needed_targets.update(matches)

    # 3. Read target records from Source 2 and Source 3
    print(f"3. Loading target records (matches needed: {len(needed_targets):,})...", flush=True)
    target_records_list = []
    found_targets = set()
    sample_pool_limit = sample_size * 3
    scan_limit = (sample_size * 25) if sample_size < 20000 else None

    for src_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
        for eid, name, addr, country in read_source_records(src_path, max_records=scan_limit):
            if eid in needed_targets:
                target_records_list.append((eid, name, addr, country))
                found_targets.add(eid)
            elif len(target_records_list) < sample_pool_limit:
                target_records_list.append((eid, name, addr, country))
            
            if len(found_targets) >= len(needed_targets) and len(target_records_list) >= sample_pool_limit:
                break

    target_dict = {r[0]: (r[1], r[2], r[3]) for r in target_records_list}

    # 4. Build Blocker and Generate Candidates
    print(f"4. Indexing {len(target_records_list):,} target records in Blocker...", flush=True)
    blocker = MultiIndexBlocker(top_k=BLOCKING_TOP_K)
    blocker.index_target_records(target_records_list)

    print("5. Generating candidates for training pairs...", flush=True)
    candidate_dict = {}
    for s1_id, (name, addr, country) in s1_records.items():
        cands = blocker.retrieve_candidates_for_query(s1_id, name, addr, country)
        candidate_dict[s1_id] = cands

    # 5. Build Training Feature Matrix
    print("6. Extracting pairwise features for training...", flush=True)
    rows = []
    labels = []
    pair_keys = []

    for s1_id, (name1, addr1, country1) in s1_records.items():
        true_matches = gt.get(s1_id, set())
        cands = set(candidate_dict.get(s1_id, []))
        pool = cands | (true_matches & set(target_dict.keys()))

        for cand_id in pool:
            if cand_id not in target_dict:
                continue
            name2, addr2, country2 = target_dict[cand_id]
            feats = compute_pairwise_features(
                name1, addr1, country1, s1_id,
                name2, addr2, country2, cand_id
            )
            rows.append(feats)
            labels.append(1 if cand_id in true_matches else 0)
            pair_keys.append((s1_id, cand_id))

    if not rows:
        print("Warning: No candidate pairs generated. Creating synthetic anchor.", flush=True)
        return EntityMatcherModel(threshold=DEFAULT_THRESHOLD)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    print(f"Dataset constructed: {len(X):,} candidate pairs (Positives: {int(y.sum()):,}, Negatives: {int((1-y).sum()):,})", flush=True)

    # Train / Validation Split
    s1_ids_list = list(s1_records.keys())
    np.random.seed(42)
    np.random.shuffle(s1_ids_list)
    val_cut = int(len(s1_ids_list) * 0.25)
    val_s1_set = set(s1_ids_list[:val_cut])
    train_s1_set = set(s1_ids_list[val_cut:])

    train_mask = [pair[0] in train_s1_set for pair in pair_keys]
    val_mask = [pair[0] in val_s1_set for pair in pair_keys]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val = X[val_mask]

    # Fit Model
    print("7. Training XGBoost Matcher Model...", flush=True)
    matcher = EntityMatcherModel(threshold=DEFAULT_THRESHOLD)
    matcher.fit(X_train, y_train)

    # Threshold Optimization on Validation Set
    val_df = pd.DataFrame(pair_keys, columns=["source1_entity_id", "candidate_entity_id"])[val_mask]
    val_gt = {k: v for k, v in gt.items() if k in val_s1_set}
    val_cands = {k: v for k, v in candidate_dict.items() if k in val_s1_set}
    val_probas = matcher.predict_proba(X_val)

    matcher.optimize_threshold(val_df, val_gt, val_cands, val_probas)

    if model_save_path:
        matcher.save(model_save_path)
        print(f"Model saved to: {model_save_path}", flush=True)

    print(f"Training completed in {time.time() - start_time:.1f}s", flush=True)
    return matcher


def run_test_inference(
    matcher: EntityMatcherModel,
    limit_test: int = None,
    candidate_output: Path = CANDIDATE_PAIRS_PATH,
    matching_output: Path = MATCHING_RESULTS_PATH
):
    """
    Run candidate blocking and matching inference over test records.
    Produces candidate_pairs.tsv and matching_results.tsv.
    """
    print(f"\n=== Running Test Inference (Limit: {'ALL' if not limit_test else f'{limit_test:,}'}) ===", flush=True)
    start_time = time.time()

    # 1. Load target test records (Source 2 and Source 3)
    target_limit = (limit_test * 5) if limit_test else None
    print(f"1. Indexing test target records (Limit per source: {target_limit or 'ALL'})...", flush=True)
    target_records_list = []
    for path in [TEST_SOURCE2, TEST_SOURCE3]:
        print(f"   Reading {path.name}...", flush=True)
        for r in read_source_records(path, max_records=target_limit):
            target_records_list.append(r)

    print(f"Total target pool indexed: {len(target_records_list):,} records.", flush=True)
    blocker = MultiIndexBlocker(top_k=BLOCKING_TOP_K)
    blocker.index_target_records(target_records_list)

    target_lookup = {r[0]: (r[1], r[2], r[3]) for r in target_records_list}
    del target_records_list  # Free memory

    # 2. Process Test Source 1 records in streaming fashion
    print("2. Processing Test Source 1 entities and generating outputs...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(candidate_output, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_output, "w", encoding="utf-8", newline="") as f_match:

        cand_writer = csv.writer(f_cand, delimiter="\t")
        match_writer = csv.writer(f_match, delimiter="\t")

        # Required headers
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])

        processed_count = 0
        batch_queries = []
        batch_size = 1000

        def process_batch(batch):
            cand_rows = []
            match_rows = []

            for s1_id, name1, addr1, country1 in batch:
                cands = blocker.retrieve_candidates_for_query(s1_id, name1, addr1, country1)
                cand_rows.append([s1_id, ",".join(cands)])

                if not cands:
                    match_rows.append([s1_id, ""])
                    continue

                # Compute pairwise features
                cand_features = []
                for cid in cands:
                    name2, addr2, country2 = target_lookup[cid]
                    cand_features.append(compute_pairwise_features(
                        name1, addr1, country1, s1_id,
                        name2, addr2, country2, cid
                    ))

                X_chunk = pd.DataFrame(cand_features)
                probas = matcher.predict_proba(X_chunk)

                # Match if probability >= threshold
                matched_cids = [
                    cid for cid, p in zip(cands, probas)
                    if p >= matcher.threshold
                ]
                match_rows.append([s1_id, ",".join(matched_cids)])

            cand_writer.writerows(cand_rows)
            match_writer.writerows(match_rows)

        for s1_id, name1, addr1, country1 in read_source_records(TEST_SOURCE1, max_records=limit_test):
            batch_queries.append((s1_id, name1, addr1, country1))
            processed_count += 1

            if len(batch_queries) >= batch_size:
                process_batch(batch_queries)
                batch_queries = []
                print(f"   Processed {processed_count:,} queries... ({time.time() - start_time:.1f}s)", flush=True)

        if batch_queries:
            process_batch(batch_queries)

    print(f"\nInference completed: {processed_count:,} entities processed in {time.time() - start_time:.1f}s", flush=True)
    print(f"Output saved to:\n  - {candidate_output}\n  - {matching_output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["full", "sample", "train_only"], default="sample",
                        help="Execution mode: 'sample' for rapid testing, 'full' for full submission run")
    parser.add_argument("--sample-size", type=int, default=2000,
                        help="Sample size for training and testing in sample mode")
    args = parser.parse_args()

    model_path = BASE_DIR / "code" / "business_entity_resolution" / "matcher_model.pkl"

    if args.mode == "sample":
        print(">>> Running Sample Benchmark Pipeline <<<", flush=True)
        matcher = train_pipeline(sample_size=args.sample_size, model_save_path=model_path)
        run_test_inference(matcher, limit_test=args.sample_size)
    elif args.mode == "train_only":
        train_pipeline(sample_size=30000, model_save_path=model_path)
    elif args.mode == "full":
        print(">>> Running Full Scale Pipeline <<<", flush=True)
        matcher = train_pipeline(sample_size=30000, model_save_path=model_path)
        run_test_inference(matcher, limit_test=None)


if __name__ == "__main__":
    main()
