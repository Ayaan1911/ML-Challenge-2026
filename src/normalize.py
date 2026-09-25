# STUB - replaced by teammate branch; do not extend here.
"""src/normalize.py - owner AM. Stdlib only: re, unicodedata, typing, math."""
import math
import re
import unicodedata

_FR = ("sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop")
_LEGAL = {"inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
          "co": "co", "company": "co", "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc",
          "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pvte": "pvt", "opc": "opc",
          **{w: w for w in _FR}}
_ADDR = {"street": "st", "str": "st", "road": "rd", "avenue": "av", "ave": "av", "avn": "av",
         "boulevard": "bd", "blvd": "bd", "boul": "bd", "lane": "ln", "drive": "dr", "court": "ct",
         "place": "pl", "plz": "pl", "plaza": "pl", "highway": "hwy", "route": "rte", "suite": "ste",
         "apartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg", "bldng": "bldg",
         "number": "no", "num": "no", "north": "n", "south": "s", "east": "e", "west": "w",
         "nr": "near", "opposite": "opp", "crs": "cross", "mn": "main", "sec": "sector",
         "chemin": "ch", "impasse": "imp", "allee": "all", "faubourg": "fbg",
         "bengaluru": "bangalore", "blr": "bangalore", "bombay": "mumbai", "madras": "chennai",
         "calcutta": "kolkata", "gurugram": "gurgaon"}
_DBA = re.compile(r"\b(?:d/b/a|d\.b\.a\.?|dba|doing business as|trading as|t/a|a\.k\.a\.?|aka)\b", re.I)
_CUE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|pres de|face a)\b\s*([^,]*)", re.I)


def _s(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def normalize_country(country) -> str:
    return _s(country).strip().lower()


def clean_text(text) -> str:
    t = unicodedata.normalize("NFKD", _s(text))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("&", " and ").replace("@", " at ")
    t = "".join(c if c.isalnum() or c == " " else " " for c in t)
    return " ".join(t.split())


def split_dba(name, country: str = "") -> tuple[str, str]:
    parts = _DBA.split(_s(name), maxsplit=1)
    return clean_text(parts[0]), clean_text(parts[1]) if len(parts) > 1 else ""


def split_legal_suffix(name_clean: str, country: str = "") -> tuple[str, str]:
    words = _s(name_clean).split()
    core, prefix, suffix = list(words), [], []
    if core and core[0] == "the":
        core.pop(0)
    while core and core[0] in _FR:
        prefix.append(core.pop(0))
    while core and core[-1] in _LEGAL:
        suffix.insert(0, _LEGAL[core.pop()])
    suffix = prefix + suffix
    if not core:
        return " ".join(words), ""
    return " ".join(core), " ".join(suffix)


def normalize_name(name, country: str = "") -> str:
    raw = _DBA.sub(" dba ", _s(name))
    return " ".join(_LEGAL.get(w, w) for w in clean_text(raw).split())


def normalize_address(address, country: str = "") -> str:
    return " ".join(_ADDR.get(w, w) for w in clean_text(address).split() if w != "cedex")


def extract_postcode(address, country: str = "") -> str:
    runs = re.findall(r"(?<!\d)\d{5,6}(?!\d)", _s(address))
    return runs[-1] if runs else ""


def extract_numbers(address, country: str = "") -> list[str]:
    pc = extract_postcode(address, country)
    out = []
    for r in re.findall(r"\d+", _s(address)):
        n = r.lstrip("0") or "0"
        if r != pc and n not in out:
            out.append(n)
    return out


def extract_landmark(address, country: str = "") -> str:
    return " | ".join(filter(None, (clean_text(m) for m in _CUE.findall(_s(address)))))


def normalize_record(record: dict) -> dict:
    record = record if isinstance(record, dict) else {}
    country = normalize_country(record.get("country"))
    name, addr = record.get("business_name"), record.get("business_address")
    primary, trade = split_dba(name, country)
    core, suffix = split_legal_suffix(primary, country)
    addr_norm = normalize_address(addr, country)
    postcode = extract_postcode(addr, country)
    landmark = normalize_address(extract_landmark(addr, country), country)
    drop = {postcode, "near", "opp", *landmark.split()}
    return {
        "entity_id": _s(record.get("entity_id")),
        "country_norm": country,
        "name_norm": normalize_name(name, country),
        "name_core": " ".join(_LEGAL.get(w, w) for w in core.split()),
        "legal_suffix": suffix,
        "dba_name": split_legal_suffix(trade, country)[0] if trade else "",
        "addr_norm": addr_norm,
        "addr_core": " ".join(w for w in addr_norm.split() if w not in drop),
        "postcode": postcode,
        "numbers": extract_numbers(addr, country),
        "landmark": landmark,
    }
