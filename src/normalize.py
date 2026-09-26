"""src/normalize.py - NormRecord creation (TEAM_GUIDE Section 5.2), noise-hardened in Pass 7.

Owner AM; Pass 7 rewrite by MA from the Section 5.2 spec, keeping AM's precompiled-regex style.
Imports: stdlib, plus anyascii (ISC license, approved by MA in Pass 7) for transliteration.

Noise handled (docs/noise_audit.md): accents on any Latin letter; Indic scripts (Devanagari, Bengali, Gujarati,
Gurmukhi, Tamil, Telugu, Kannada, Malayalam) transliterated to Latin, their legal forms mapped to pvt / ltd /
llp; URL protocols, www., TLDs and a leading @ or # in names; trailing phone or registration numbers (6-12 digits)
in names; junk tokens (N/A, <NULL>, ##, --, empty brackets); doubled words in names; French legal forms and
street words.
"""
import math
import re
import unicodedata

from anyascii import anyascii

# ---------- tables (Section 5.3 / 5.4, plus Pass 7 additions) ----------

_FR_FORMS = ("sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop")
_LEGAL = {"inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
          "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "ltd": "ltd", "limited": "ltd",
          "pvt": "pvt", "private": "pvt", "pvte": "pvt", "opc": "opc", **{w: w for w in _FR_FORMS},
          # transliterated Indic legal forms (anyascii output, counted on train S2/S3)
          "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivrr": "pvt",
          "limitet": "ltd", "limirrd": "ltd", "limtid": "ltd", "elelpi": "llp"}
_LEGAL_PHRASES = re.compile(r"\b(?:l l c|l l p|pra li)\b")
_LEGAL_PHRASE_MAP = {"l l c": "llc", "l l p": "llp", "pra li": "pvt ltd"}

_ADDR = {"street": "st", "str": "st", "road": "rd", "avenue": "av", "ave": "av", "avn": "av",
         "boulevard": "bd", "blvd": "bd", "boul": "bd", "lane": "ln", "drive": "dr", "court": "ct",
         "place": "pl", "plz": "pl", "plaza": "pl", "highway": "hwy", "route": "rte", "suite": "ste",
         "apartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg", "bldng": "bldg",
         "number": "no", "num": "no", "north": "n", "south": "s", "east": "e", "west": "w",
         "nr": "near", "opposite": "opp", "crs": "cross", "mn": "main", "sec": "sector",
         "chemin": "ch", "impasse": "imp", "allee": "all", "faubourg": "fbg", "square": "sq",
         "bengaluru": "bangalore", "blr": "bangalore", "bombay": "mumbai", "madras": "chennai",
         "calcutta": "kolkata", "gurugram": "gurgaon"}
_ADDR_FR = {"r": "rue"}  # a standalone "r" is "rue" only in French addresses
_ADDR_PHRASES = re.compile(r"\b(?:mahatma gandhi|new delhi|delhi|nd)\b")
_ADDR_PHRASE_MAP = {"mahatma gandhi": "mg", "new delhi": "new delhi", "delhi": "new delhi", "nd": "new delhi"}

# ---------- precompiled regexes ----------

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|trading as|t/a|a\.k\.a\.?|aka)\b", re.I)
_JUNK = re.compile(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|#{2,}|-{2,}|[\(\[\{]\s*\.*\s*[\)\]\}]", re.I)
_URL = re.compile(r"(?i)https?://|\bwww\.|\.(?:co\.in|com|org|net|in|fr|biz|info|us|co|io)\b|^[\s@#]+")
_TRAIL_NUM = re.compile(r"(?:^|[\s\-#])\d{6,12}\s*$")
_CUE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|pres de|face a)\b\.?"
                  r"\s*([^,]*)", re.I)
_POSTCODE = {"india": re.compile(r"(?<!\d)([1-9]\d{2}) ?(\d{3})(?!\d)"),
             "us": re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)"),
             "fr": re.compile(r"(?<!\d)(\d{5})(?!\d)"),
             "any": re.compile(r"(?<!\d)(\d{5,6})(?!\d)")}
_DIGITS = re.compile(r"\d+")


_KIND_CACHE: dict[str, str] = {}


def _s(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return x if isinstance(x, str) else str(x)


def _fold(text) -> str:
    """ASCII transliteration: accents stripped, Indic and other scripts romanized (anyascii)."""
    t = _s(text)
    return t if t.isascii() else anyascii(t)


def _kind(country: str) -> str:
    c = country if country.islower() or not country else normalize_country(country)
    if "india" in c:
        return "india"
    if c in ("us", "usa", "united states", "united states of america"):
        return "us"
    if c in ("france", "fr"):
        return "fr"
    return "any"


# ---------- Section 5.2 functions ----------

def normalize_country(country) -> str:
    return _s(country).strip().lower()


def _clean_folded(t: str) -> str:
    """clean_text for text that is already ASCII-folded."""
    return " ".join(_NON_ALNUM.sub(" ", t.lower().replace("&", " and ").replace("@", " at ")).split())


def clean_text(text) -> str:
    return _clean_folded(_fold(text))


def _landmarks(folded: str) -> list[str]:
    fl = folded.lower()
    if not ("near" in fl or "nr" in fl or "opp" in fl or "behind" in fl or "beside" in fl or "next to" in fl
            or "adjacent" in fl or "front of" in fl or "pres de" in fl or "face a" in fl):
        return []
    return [_clean_folded(m) for m in _CUE.findall(folded)]


def _dejunk(t: str) -> str:
    if "ull" in t or "ULL" in t or "/" in t or "##" in t or "--" in t or "(" in t or "[" in t or "{" in t:
        return _JUNK.sub(" ", t)
    return t


def _name_pre(name) -> str:
    """Raw name -> folded name without junk tokens, URL parts and a trailing phone/registration number."""
    t = _dejunk(_fold(name))
    if "." in t or "/" in t or "@" in t or "#" in t or "www" in t or "WWW" in t:
        t = _URL.sub(" ", t)
    return _TRAIL_NUM.sub("", t) if t[-1:].isdigit() or t[-2:-1].isdigit() else t


def _words(cleaned: str) -> list[str]:
    """Legal phrases joined, doubled words collapsed, 'et' between two words -> 'and', legal canonical forms."""
    if "l l " in cleaned or "pra li" in cleaned:
        cleaned = _LEGAL_PHRASES.sub(lambda m: _LEGAL_PHRASE_MAP[m.group(0)], cleaned)
    out: list[str] = []
    prev = None
    for w in cleaned.split():
        if w != prev:
            out.append(_LEGAL.get(w, w))
            prev = w
    if "et" in out:  # French 'et' between two words -> 'and' (Table 5.3)
        out = ["and" if w == "et" and 0 < i < len(out) - 1 else w for i, w in enumerate(out)]
    return out


def _maybe_dba(t: str) -> bool:
    lt = t.lower()
    return ("dba" in lt or "d/" in lt or "d." in lt or "d /" in lt or "aka" in lt or "a.k" in lt or "t/a" in lt
            or "trading" in lt or "doing" in lt)


def split_dba(name, country: str = "") -> tuple[str, str]:
    parts = _DBA.split(_name_pre(name), maxsplit=1)
    return clean_text(parts[0]), clean_text(parts[1]) if len(parts) > 1 else ""


def split_legal_suffix(name_clean: str, country: str = "") -> tuple[str, str]:
    words = _s(name_clean).split()
    core, prefix, suffix = list(words), [], []
    if core and core[0] == "the":
        core.pop(0)
    while core and core[0] in _FR_FORMS:
        prefix.append(core.pop(0))
    while core and core[-1] in _LEGAL:
        suffix.insert(0, _LEGAL[core.pop()])
    if not core:
        return " ".join(words), ""
    return " ".join(core), " ".join(prefix + suffix)


def normalize_name(name, country: str = "") -> str:
    return " ".join(_words(clean_text(_DBA.sub(" dba ", _name_pre(name)))))


def _addr_tokens(folded: str, kind: str) -> list[str]:
    """Folded, de-junked address -> canonical address words (Table 5.4), 'cedex' dropped."""
    t = " ".join(_NON_ALNUM.sub(" ", folded.lower().replace("&", " and ").replace("@", " at ")).split())
    if "delhi" in t or "gandhi" in t or "nd" in t:
        t = _ADDR_PHRASES.sub(lambda m: _ADDR_PHRASE_MAP[m.group(0)], t)
    get = _ADDR.get
    if kind == "fr":
        return [_ADDR_FR.get(w, get(w, w)) for w in t.split() if w != "cedex"]
    return [get(w, w) for w in t.split() if w != "cedex"]


def normalize_address(address, country: str = "") -> str:
    return " ".join(_addr_tokens(_dejunk(_fold(address)), _kind(country)))


def _postcode_span(folded: str, country: str, kind: str | None = None) -> tuple[str, tuple[int, int] | None]:
    rx = _POSTCODE[kind or _kind(country)]
    last = None
    for last in rx.finditer(folded):
        pass
    if last is None:
        return "", None
    return "".join(g for g in last.groups() if g), last.span(1) if last.lastindex == 1 else (last.start(1), last.end(2))


def extract_postcode(address, country: str = "") -> str:
    return _postcode_span(_fold(address), country)[0]


def _numbers(folded: str, span) -> list[str]:
    if span:
        folded = folded[:span[0]] + " " + folded[span[1]:]
    out: list[str] = []
    for r in _DIGITS.findall(folded):
        n = r.lstrip("0") or "0"
        if n not in out:
            out.append(n)
    return out


def extract_numbers(address, country: str = "") -> list[str]:
    folded = _fold(address)
    return _numbers(folded, _postcode_span(folded, country)[1])


def extract_landmark(address, country: str = "") -> str:
    return " | ".join(filter(None, (clean_text(m) for m in _CUE.findall(_fold(address)))))


def normalize_record(record: dict) -> dict:
    record = record if isinstance(record, dict) else {}
    country = normalize_country(record.get("country"))
    # name: one folded pass shared by name_norm, the DBA split and the legal split
    pre = _name_pre(record.get("business_name"))
    parts = _DBA.split(pre, maxsplit=1) if _maybe_dba(pre) else [pre]
    primary = " ".join(_words(_clean_folded(parts[0])))
    core, suffix = split_legal_suffix(primary, country)
    if len(parts) > 1:
        trade_words = " ".join(_words(_clean_folded(parts[1])))
        trade = split_legal_suffix(trade_words, country)[0]
        name_norm = f"{primary} dba {trade_words}".strip()
    else:
        trade, name_norm = "", primary
    # address: postcode and landmark from the folded raw text; one tokenization, addr_core derived from it
    kind = _KIND_CACHE.get(country) or _KIND_CACHE.setdefault(country, _kind(country))
    folded = _dejunk(_fold(record.get("business_address")))
    postcode, span = _postcode_span(folded, country, kind)
    landmarks = _landmarks(folded) if folded else []
    addr = _addr_tokens(folded, kind)
    if landmarks:
        rest = folded if not span else folded[:span[0]] + " " + folded[span[1]:]
        core_toks = _addr_tokens(_CUE.sub(",", rest), kind)
    elif span:
        pc_toks = _NON_ALNUM.sub(" ", folded[span[0]:span[1]].lower()).split()  # '560 041' -> two tokens
        core_toks, n = list(addr), len(pc_toks)
        for i in range(len(addr) - n, -1, -1):  # drop the last occurrence of the postcode tokens
            if addr[i:i + n] == pc_toks:
                del core_toks[i:i + n]
                break
    else:
        core_toks = addr
    return {
        "entity_id": _s(record.get("entity_id")),
        "country_norm": country,
        "name_norm": name_norm,
        "name_core": core,
        "legal_suffix": suffix,
        "dba_name": trade,
        "addr_norm": " ".join(addr),
        "addr_core": " ".join(core_toks),
        "postcode": postcode,
        "numbers": _numbers(folded, span),
        "landmark": " ".join(_addr_tokens(" | ".join(filter(None, landmarks)), kind)) if landmarks else "",
    }
