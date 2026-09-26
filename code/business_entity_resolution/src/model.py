import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
import xgboost as xgb
from .evaluate import evaluate_macro_f_beta


class EntityMatcherModel:
    """
    ML Matching Model using XGBoost and F_0.5 threshold optimization.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit XGBoost model on feature matrix."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities of being a true match."""
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def optimize_threshold(
        self,
        val_df: pd.DataFrame,
        val_ground_truth: Dict[str, Set[str]],
        val_candidates: Dict[str, List[str]],
        candidate_probas: np.ndarray,
        threshold_range: List[float] = None
    ) -> float:
        """
        Find optimal decision threshold that maximizes macro F_0.5 on validation set.
        val_df must contain ['source1_entity_id', 'candidate_entity_id'].
        """
        if threshold_range is None:
            threshold_range = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        # Map (s1_id, cand_id) -> proba
        pair_to_prob = {}
        for (s1, cand), prob in zip(zip(val_df['source1_entity_id'], val_df['candidate_entity_id']), candidate_probas):
            pair_to_prob[(s1, cand)] = prob

        best_score = -1.0
        best_thresh = self.threshold

        print("\n--- Tuning Decision Threshold for Macro F_0.5 ---")
        for th in threshold_range:
            preds = {}
            for s1_id in val_ground_truth.keys():
                cands = val_candidates.get(s1_id, [])
                matched = {c for c in cands if pair_to_prob.get((s1_id, c), 0.0) >= th}
                preds[s1_id] = matched

            eval_res = evaluate_macro_f_beta(preds, val_ground_truth, beta=0.5)
            score = eval_res['macro_f_beta']
            print(f"Threshold: {th:.2f} -> Macro F_0.5: {score:.5f} | Singleton Acc: {eval_res['singleton_accuracy']:.4f}")

            if score > best_score:
                best_score = score
                best_thresh = th

        print(f"Optimal Threshold Selected: {best_thresh:.2f} (Macro F_0.5: {best_score:.5f})\n")
        self.threshold = best_thresh
        return best_thresh

    def save(self, filepath: str):
        """Save model and metadata to disk."""
        data = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_names": self.feature_names
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load model and metadata from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.threshold = data["threshold"]
        self.feature_names = data["feature_names"]
