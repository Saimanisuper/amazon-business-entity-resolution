from typing import Dict, Set, List


def compute_f_beta_entity(
    predicted_matches: Set[str],
    true_matches: Set[str],
    beta: float = 0.5
) -> float:
    """
    Calculate F_beta score for a single Source 1 entity.
    Special rules:
    - If true_matches is empty:
        - returns 1.0 if predicted_matches is also empty (singleton correctly identified)
        - returns 0.0 if predicted_matches has any entities (false merge on singleton)
    - If true_matches is non-empty:
        - returns 0.0 if predicted_matches is empty
        - calculates precision and recall, returns F_beta
    """
    # Singleton case
    if not true_matches:
        return 1.0 if not predicted_matches else 0.0
    
    # Non-singleton but nothing predicted
    if not predicted_matches:
        return 0.0
    
    tp = len(predicted_matches & true_matches)
    if tp == 0:
        return 0.0
    
    precision = tp / len(predicted_matches)
    recall = tp / len(true_matches)
    
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def evaluate_macro_f_beta(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Compute macro-average F_beta across all entities in ground_truth.
    Returns:
        dict with keys: 'macro_f_beta', 'singleton_accuracy', 'evaluated_entities'
    """
    total_score = 0.0
    singleton_count = 0
    singleton_correct = 0
    non_singleton_count = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_f_beta_entity(pred_set, true_set, beta=beta)
        total_score += score
        
        if not true_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            
    num_entities = len(ground_truth)
    macro_score = total_score / num_entities if num_entities > 0 else 0.0
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 1.0
    
    return {
        "macro_f_beta": macro_score,
        "singleton_accuracy": singleton_acc,
        "singleton_count": singleton_count,
        "non_singleton_count": non_singleton_count,
        "total_entities": num_entities,
    }
