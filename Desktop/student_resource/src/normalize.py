import re

LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|pvt|gmbh|sa|sas)\b", 
    re.IGNORECASE
)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Clean punctuation first so "Inc." becomes "inc "
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    # 2. Strip legal suffixes
    text = LEGAL_SUFFIXES.sub(" ", text)
    # 3. Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()

def normalize_record(record: dict) -> dict:
    return {
        "entity_id": record.get("entity_id", ""),
        "business_name": normalize_text(record.get("business_name", "")),
        "business_address": normalize_text(record.get("business_address", "")),
        "country": str(record.get("country", "")).strip().upper()
    }
