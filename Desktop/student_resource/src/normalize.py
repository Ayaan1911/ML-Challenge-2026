"""
src/normalize.py
Record normalization module.
Assigned to: MA (Data Normalization Lead)
AM Contract: normalize_record(record: dict) -> dict
"""
import re

LEGAL_SUFFIXES = re.compile(r'(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|pvt|gmbh|sa|sas)', re.IGNORECASE)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = LEGAL_SUFFIXES.sub('', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_record(record: dict) -> dict:
    """
    Normalizes a single entity record.
    Works for any country (US, India, France, etc.).
    """
    return {
        'entity_id': record.get('entity_id', ''),
        'business_name': normalize_text(record.get('business_name', '')),
        'business_address': normalize_text(record.get('business_address', '')),
        'country': str(record.get('country', '')).strip().upper()
    }
