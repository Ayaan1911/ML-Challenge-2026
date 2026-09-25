"""
src/features.py
Pairwise feature extraction module (27+ features).
Assigned to: RA (Feature Engineering Lead)
AM Contract: pair_features(rec1: dict, rec2: dict) -> dict
"""
import re

def jaccard_similarity(s1: str, s2: str) -> float:
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def pair_features(rec1: dict, rec2: dict) -> dict:
    """
    Computes pairwise similarity features between query rec1 and candidate rec2.
    """
    n1, n2 = str(rec1.get('business_name', '')), str(rec2.get('business_name', ''))
    a1, a2 = str(rec1.get('business_address', '')), str(rec2.get('business_address', ''))
    
    # Core baseline features
    return {
        'name_jaccard': jaccard_similarity(n1, n2),
        'addr_jaccard': jaccard_similarity(a1, a2),
        'country_match': 1.0 if rec1.get('country') == rec2.get('country') else 0.0,
        'exact_name_match': 1.0 if n1 and n1 == n2 else 0.0
    }
