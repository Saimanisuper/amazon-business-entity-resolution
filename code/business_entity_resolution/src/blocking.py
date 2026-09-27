import collections
from typing import Dict, List, Set, Tuple
from .preprocess import clean_name, clean_address, extract_numbers, get_name_tokens

# Frequent generic words across business names to ignore during initial token indexing
COMMON_BUSINESS_STOPWORDS = {
    'the', 'and', 'for', 'of', 'in', 'at', 'by', 'to', 'on', 'with',
    'enterprise', 'enterprises', 'service', 'services', 'solutions', 'solution',
    'group', 'holdings', 'global', 'center', 'centre', 'shop', 'store',
    'mart', 'market', 'hotel', 'restaurant', 'cafe', 'bar', 'dhaba',
    'associates', 'trading', 'agency', 'agencies', 'industries', 'industry',
    'international', 'national', 'tech', 'technologies', 'consulting'
}


class MultiIndexBlocker:
    """
    High-throughput, memory-efficient candidate generator (blocking engine).
    Combines:
    1. Country partition
    2. Core name token inverted index
    3. Trigram / Prefix blocking key
    4. Top-K candidate ranking with overlap scoring
    """

    def __init__(self, top_k: int = 15, max_token_frequency: int = 5000):
        self.top_k = top_k
        self.max_token_frequency = max_token_frequency
        # index: (country, token) -> list of entity_ids
        self.inverted_index = collections.defaultdict(list)
        # target_data: entity_id -> (clean_name, clean_addr, nums)
        self.target_data = {}

    def index_target_records(self, target_records: List[Tuple[str, str, str, str]]):
        """
        Build single-pass inverted index over target records with on-the-fly frequency capping.
        """
        for eid, name, addr, country in target_records:
            c_name = clean_name(name)
            c_addr = clean_address(addr)
            nums = extract_numbers(addr)
            self.target_data[eid] = (c_name, c_addr, nums, country)

            tokens = set(get_name_tokens(c_name)) - COMMON_BUSINESS_STOPWORDS
            indexed_any = False
            for t in tokens:
                key = (country, t)
                if len(self.inverted_index[key]) < self.max_token_frequency:
                    self.inverted_index[key].append(eid)
                    indexed_any = True

            # If no selective token was indexed, use 4-character prefix
            if not indexed_any and len(c_name) >= 3:
                prefix = c_name[:4].strip()
                if prefix:
                    key = (country, f"__pref__{prefix}")
                    if len(self.inverted_index[key]) < self.max_token_frequency:
                        self.inverted_index[key].append(eid)

    def retrieve_candidates_for_query(
        self,
        s1_id: str,
        name: str,
        addr: str,
        country: str
    ) -> List[str]:
        """
        Retrieve and rank top-K candidate target entity_ids for a single Source 1 query.
        """
        c_name = clean_name(name)
        c_addr = clean_address(addr)
        nums1 = extract_numbers(addr)
        tokens1 = set(get_name_tokens(c_name)) - COMMON_BUSINESS_STOPWORDS

        candidate_scores = collections.defaultdict(float)

        # Match via selective tokens
        matched_tokens = False
        for t in tokens1:
            key = (country, t)
            matches = self.inverted_index.get(key, [])
            if matches:
                matched_tokens = True
                # Weight by token rarity
                weight = 1.0 + (len(t) * 0.2)
                for eid in matches:
                    candidate_scores[eid] += weight

        # If no candidates found via tokens, fallback to prefix key
        if not candidate_scores and len(c_name) >= 3:
            prefix = c_name[:4].strip()
            key = (country, f"__pref__{prefix}")
            for eid in self.inverted_index.get(key, []):
                candidate_scores[eid] += 1.0

        if not candidate_scores:
            return []

        # Re-rank candidates by quick overlap score
        ranked_candidates = []
        for eid, base_score in candidate_scores.items():
            t_name, t_addr, t_nums, _ = self.target_data[eid]
            # Fast verification boosts
            score = base_score
            if c_name and c_name == t_name:
                score += 5.0
            if nums1 and t_nums and (nums1 & t_nums):
                score += 2.0
            ranked_candidates.append((score, eid))

        ranked_candidates.sort(key=lambda x: x[0], reverse=True)
        return [eid for _, eid in ranked_candidates[:self.top_k]]
