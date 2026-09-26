"""
src/normalize.py - High-speed, zero-backtracking NormRecord creation.
Owner: MA (Adapter for AM & RA pipeline)
"""
import re
from typing import Any, Dict, List

RE_CLEAN = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_LEGAL = re.compile(r"\b(pvt ltd|private limited|ltd|limited|llc|inc|incorporated|corp|corporation|co|company|gmbh|sarl|sas|sa|bv|plc)\b", re.IGNORECASE)

def clean_text(s: Any) -> str:
    if not s or not isinstance(s, str):
        return ""
    s_clean = RE_CLEAN.sub(" ", s.lower())
    return RE_SPACES.sub(" ", s_clean).strip()

def normalize_record(record: dict) -> dict:
    raw_id = str(record.get("entity_id", ""))
    name_norm = clean_text(record.get("business_name") or record.get("name", ""))
    addr_norm = clean_text(record.get("business_address") or record.get("address", ""))
    country_raw = str(record.get("country", "")).strip()
    country_norm = clean_text(country_raw)

    # 1. Fast DBA split
    dba_name = ""
    name_core = name_norm
    if " dba " in name_norm:
        parts = name_norm.split(" dba ", 1)
        name_core, dba_name = parts[0].strip(), parts[1].strip()

    # 2. Fast Legal Suffix
    legal_suffix = ""
    match = RE_LEGAL.search(name_core)
    if match:
        legal_suffix = match.group(1).lower()
        name_core = RE_LEGAL.sub(" ", name_core)
        name_core = RE_SPACES.sub(" ", name_core).strip()

    # 3. Numbers & Postcode
    numbers = RE_NUMBERS.findall(addr_norm)
    postcode = ""
    for n in reversed(numbers):
        if len(n) in (5, 6):
            postcode = n
            break

    numbers_core = [n for n in numbers if n != postcode]

    return {
        "entity_id": raw_id,
        "country": country_raw.upper(),           # Legacy Phase 1 key
        "country_norm": country_norm,             # RA's schema key
        "business_name": name_norm,               # Legacy Phase 1 key
        "business_address": addr_norm,            # Legacy Phase 1 key
        "name_norm": name_norm,
        "name_core": name_core,
        "legal_suffix": legal_suffix,
        "dba_name": dba_name,
        "addr_norm": addr_norm,
        "addr_core": addr_norm,
        "postcode": postcode,
        "numbers": numbers_core if numbers_core else numbers,
        "landmark": "",
    }
