import math
from difflib import SequenceMatcher
from .preprocess import clean_name, clean_address, extract_numbers, get_name_tokens


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def dice_coefficient(set1: set, set2: set) -> float:
    """Compute Sørensen-Dice coefficient between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    total = len(set1) + len(set2)
    return (2.0 * intersection) / total if total > 0 else 0.0


def char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    if not text:
        return set()
    s = f" {text} "
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def sequence_ratio(s1: str, s2: str) -> float:
    """Fast similarity ratio using SequenceMatcher."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # Quick length check
    if abs(len(s1) - len(s2)) / max(len(s1), len(s2)) > 0.8:
        return 0.0
    return SequenceMatcher(None, s1, s2).quick_ratio()


def compute_pairwise_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    s1_id: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
    cand_id: str
) -> dict:
    """
    Extract pairwise similarity features between a Source 1 record and a Candidate record.
    Returns a dictionary of numerical feature values.
    """
    clean_n1 = clean_name(s1_name)
    clean_n2 = clean_name(cand_name)
    clean_a1 = clean_address(s1_addr)
    clean_a2 = clean_address(cand_addr)
    
    # Token sets
    toks_n1 = set(clean_n1.split())
    toks_n2 = set(clean_n2.split())
    toks_a1 = set(clean_a1.split())
    toks_a2 = set(clean_a2.split())
    
    # Numbers (postal codes, street numbers)
    nums1 = extract_numbers(s1_addr)
    nums2 = extract_numbers(cand_addr)
    
    # 1. Name similarities
    name_jaccard = jaccard_similarity(toks_n1, toks_n2)
    name_dice = dice_coefficient(toks_n1, toks_n2)
    name_exact = 1.0 if (clean_n1 and clean_n1 == clean_n2) else 0.0
    name_char_jaccard = jaccard_similarity(char_ngrams(clean_n1), char_ngrams(clean_n2))
    name_seq_ratio = sequence_ratio(clean_n1, clean_n2)
    
    len_n1 = len(clean_n1)
    len_n2 = len(clean_n2)
    name_len_diff = abs(len_n1 - len_n2)
    name_len_ratio = min(len_n1, len_n2) / max(len_n1, len_n2) if max(len_n1, len_n2) > 0 else 1.0
    
    # First token match
    t1_first = clean_n1.split()[0] if clean_n1 else ""
    t2_first = clean_n2.split()[0] if clean_n2 else ""
    name_first_match = 1.0 if (t1_first and t1_first == t2_first) else 0.0
    
    # 2. Address similarities
    addr_jaccard = jaccard_similarity(toks_a1, toks_a2)
    addr_dice = dice_coefficient(toks_a1, toks_a2)
    addr_char_jaccard = jaccard_similarity(char_ngrams(clean_a1), char_ngrams(clean_a2))
    addr_seq_ratio = sequence_ratio(clean_a1, clean_a2)
    
    len_a1 = len(clean_a1)
    len_a2 = len(clean_a2)
    addr_len_diff = abs(len_a1 - len_a2)
    addr_len_ratio = min(len_a1, len_a2) / max(len_a1, len_a2) if max(len_a1, len_a2) > 0 else 1.0
    
    # Number overlap (postal codes, street numbers)
    shared_nums = len(nums1 & nums2)
    num_jaccard = jaccard_similarity(nums1, nums2)
    has_num_match = 1.0 if (nums1 and nums2 and shared_nums > 0) else 0.0
    
    # 3. Combined features
    all_toks1 = toks_n1 | toks_a1
    all_toks2 = toks_n2 | toks_a2
    combined_jaccard = jaccard_similarity(all_toks1, all_toks2)
    
    # Harmonic mean of name and address similarity
    eps = 1e-6
    geom_mean_sim = math.sqrt(max(0.0, name_jaccard * addr_jaccard))
    
    # Target source indicator (S2 vs S3)
    target_source_num = 2.0 if cand_id.startswith("S2-") else 3.0
    
    # Country check
    same_country = 1.0 if (s1_country == cand_country) else 0.0
    
    return {
        "name_jaccard": name_jaccard,
        "name_dice": name_dice,
        "name_char_jaccard": name_char_jaccard,
        "name_seq_ratio": name_seq_ratio,
        "name_exact": name_exact,
        "name_first_match": name_first_match,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_jaccard": addr_jaccard,
        "addr_dice": addr_dice,
        "addr_char_jaccard": addr_char_jaccard,
        "addr_seq_ratio": addr_seq_ratio,
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": addr_len_ratio,
        "shared_nums": shared_nums,
        "num_jaccard": num_jaccard,
        "has_num_match": has_num_match,
        "combined_jaccard": combined_jaccard,
        "geom_mean_sim": geom_mean_sim,
        "target_source_num": target_source_num,
        "same_country": same_country,
    }
