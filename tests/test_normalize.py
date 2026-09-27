"""src/normalize.py: the four Section 5.4 examples exactly, one test per Pass 7 noise rule, and speed."""
import random
import time

import pytest

from src.normalize import (clean_text, extract_landmark, extract_numbers, extract_postcode, normalize_address,
                           normalize_country, normalize_name, normalize_record, split_dba, split_legal_suffix)


def rec(name, addr="", country="US"):
    return normalize_record({"entity_id": "S2-1", "business_name": name, "business_address": addr, "country": country})


# ---------- Section 5.4 worked examples ----------

SEC54 = [
    ({"entity_id": "S1-00001", "business_name": "Sharma & Sons Pvt. Ltd.",
      "business_address": "No. 12, 4th Cross, Near SBI ATM, Jayanagar, Bengaluru 560041", "country": "India"},
     {"entity_id": "S1-00001", "country_norm": "india", "name_norm": "sharma and sons pvt ltd",
      "name_core": "sharma and sons", "legal_suffix": "pvt ltd", "dba_name": "",
      "addr_norm": "no 12 4th cross near sbi atm jayanagar bangalore 560041",
      "addr_core": "no 12 4th cross jayanagar bangalore", "postcode": "560041", "numbers": ["12", "4"],
      "landmark": "sbi atm"}),
    ({"entity_id": "S2-00047", "business_name": "ACME Corporation d/b/a Acme Tools",
      "business_address": "1200 N. Main St., Suite 4, Springfield, IL 62704-1234", "country": "US"},
     {"entity_id": "S2-00047", "country_norm": "us", "name_norm": "acme corp dba acme tools", "name_core": "acme",
      "legal_suffix": "corp", "dba_name": "acme tools", "addr_norm": "1200 n main st ste 4 springfield il 62704 1234",
      "addr_core": "1200 n main st ste 4 springfield il 1234", "postcode": "62704", "numbers": ["1200", "4", "1234"],
      "landmark": ""}),
    ({"entity_id": "S3-00812", "business_name": "SARL Boulangerie Dupont",
      "business_address": "12 bis Av. des Champs-Élysées, 75008 Paris", "country": "France"},
     {"entity_id": "S3-00812", "country_norm": "france", "name_norm": "sarl boulangerie dupont",
      "name_core": "boulangerie dupont", "legal_suffix": "sarl", "dba_name": "",
      "addr_norm": "12 bis av des champs elysees 75008 paris", "addr_core": "12 bis av des champs elysees paris",
      "postcode": "75008", "numbers": ["12"], "landmark": ""}),
    ({"entity_id": "S2-99999", "business_name": None, "business_address": float("nan"), "country": "Germany"},
     {"entity_id": "S2-99999", "country_norm": "germany", "name_norm": "", "name_core": "", "legal_suffix": "",
      "dba_name": "", "addr_norm": "", "addr_core": "", "postcode": "", "numbers": [], "landmark": ""}),
]


@pytest.mark.parametrize("inp,want", SEC54)
def test_section_5_4_examples(inp, want):
    got = normalize_record(inp)
    assert got == want
    assert list(got) == list(want)  # exact key order (Table 5.2)


# ---------- accents and transliteration ----------

def test_accents_stripped_on_any_latin_letter():
    assert clean_text("Food & Frèrês Çomite Ànciens Œuvre") == "food and freres comite anciens oeuvre"
    assert rec("YBL ÇOMITE EURL", country="France")["name_core"] == "ybl comite"


def test_devanagari_transliterated_with_legal_forms():
    r = rec("अरिहंत टेक्नोलॉजी स्टोर्स प्राइवेट लिमिटेड", country="India")
    assert r["name_core"] == "arihmt teknoloji stors" and r["legal_suffix"] == "pvt ltd"
    assert rec("ईस्ट सनराइज सिस्टम्स प्रा. लि.", country="India")["legal_suffix"] == "pvt ltd"
    assert rec("अर्बन राज एक्सपोर्ट्स एलएलपी", country="India")["legal_suffix"] == "llp"


def test_telugu_and_other_indic_scripts_transliterated():
    r = rec("నార్త్ కన్‌స్ట్రక్షన్స్ ప్రైవేట్ లిమిటెడ్", "Hyderabad-500 062", "India")
    assert r["name_core"] == "nart knstrksns" and r["legal_suffix"] == "pvt ltd"
    assert r["postcode"] == "500062" and r["addr_core"] == "hyderabad"
    for name in ("প্রিমিয়ার কনসালটেন্সি প্রাইভেট লিমিটেড", "நியூ இன்ஜினியரிங் புரொவிஷன் பிரைவேட் லிமிடெட்",
                 "ಲಕ್ಷ್ಮಿ ಹಾಸ್ಪಿಟಾಲಿಟಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"):
        r = rec(name, country="India")
        assert r["name_core"].isascii() and r["legal_suffix"] == "pvt ltd"


# ---------- French ----------

def test_french_legal_forms_prefix_and_suffix():
    assert split_legal_suffix("eurl lamitie maternelle") == ("lamitie maternelle", "eurl")
    for form in ("sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop"):
        assert rec(f"Levriers Union {form.upper()}", country="France")["legal_suffix"] == form


def test_french_et_between_words_becomes_and():
    assert rec("Dupont et Fils SAS", country="France")["name_core"] == "dupont and fils"
    assert normalize_name("Et Cetera") == "et cetera"


def test_french_street_words():
    got = normalize_address("41 Allée du Kurun, 8 Impasse Carnot, 2 Sq. Clemenceau, Faubourg St Martin, "
                            "Chemin Vert, Route de Paris, 12 Boulevard Voltaire, 3 Place Nette, 9 Avenue Foch, "
                            "1 R du Coutoum, 5 Rue Lacroix", "France")
    assert got == ("41 all du kurun 8 imp carnot 2 sq clemenceau fbg st martin ch vert rte de paris 12 bd voltaire "
                   "3 pl nette 9 av foch 1 rue du coutoum 5 rue lacroix")
    assert normalize_address("1 R K Puram", "India") == "1 r k puram"  # 'r' -> 'rue' only in France


# ---------- URLs, trailing numbers, junk, doubled words ----------

@pytest.mark.parametrize("name", ["https://www.urbangrill.com", "@urbangrill.co.in", "#urbangrill.net",
                                  "www.urbangrill.org", "urbangrill.in"])
def test_url_parts_removed(name):
    assert rec(name)["name_core"] == "urbangrill"


@pytest.mark.parametrize("name", ["Urban Grill - 9876543210", "Urban Grill #20210045", "Urban Grill 123456789012",
                                  "Urban Grill – 987654"])
def test_trailing_phone_or_registration_number_removed(name):
    assert rec(name)["name_core"] == "urban grill"


def test_short_trailing_numbers_kept():
    assert rec("Studio 54")["name_core"] == "studio 54"
    assert rec("Route 66 Diner 12345")["name_core"] == "route 66 diner 12345"  # 5 digits: not a phone number


def test_junk_tokens_and_doubled_words():
    r = rec("Sharma Sharma Traders Traders N/A", "N/A, <NULL>, ## 12 -- MG Road [..] ()", "India")
    assert r["name_norm"] == "sharma traders"
    assert r["addr_norm"] == "12 mg rd"
    assert rec("Acme <NULL> Tools", "6669 Agusta Ln, <NULL>, Safford")["addr_norm"] == "6669 agusta ln safford"


def test_legal_phrases_and_address_phrases():
    r = rec("Acme L.L.C.", "Mahatma Gandhi Road, Delhi 110 001", "India")
    assert (r["name_core"], r["legal_suffix"]) == ("acme", "llc")
    assert r["addr_core"] == "mg rd new delhi" and r["postcode"] == "110001"


# ---------- postcodes, numbers, landmarks ----------

def test_postcodes_by_country():
    assert extract_postcode("Jayanagar 560 041", "India") == "560041"
    assert extract_postcode("Jayanagar 060041", "India") == ""  # India PIN never starts with 0
    assert extract_postcode("Springfield IL 62704-1234", "US") == "62704"
    assert extract_postcode("75008 Paris", "France") == "75008"
    assert extract_postcode("10115 Berlin, Box 123456", "Germany") == "123456"  # generic: last 5-6 digit run
    assert extract_numbers("No. 004, 2nd Floor, 1er etage 12345", "US") == ["4", "2", "1"]


def test_landmarks():
    assert extract_landmark("12 MG Rd, Opp. City Mall, Near Bus Stand") == "city mall | bus stand"
    assert extract_landmark("3 rue X, près de la gare", "France") == "la gare"
    r = rec("X", "12 MG Rd, Opp. City Mall, Bangalore", "India")
    assert r["landmark"] == "city mall" and r["addr_core"] == "12 mg rd bangalore"


def test_dba_split_on_raw_name():
    assert split_dba("ACME Corp d.b.a. Acme Tools") == ("acme corp", "acme tools")
    assert split_dba("Plume trading as Plumes Sportive") == ("plume", "plumes sportive")
    assert split_dba("Acme") == ("acme", "")


# ---------- robustness and speed ----------

@pytest.mark.parametrize("bad", [None, float("nan"), "", 12345, "   "])
def test_never_raises(bad):
    r = normalize_record({"entity_id": bad, "business_name": bad, "business_address": bad, "country": bad})
    assert all(isinstance(v, (str, list)) for v in r.values())
    assert normalize_country(bad) == normalize_country(bad)


def test_speed():
    rnd = random.Random(0)
    names = ["Sharma & Sons Pvt. Ltd.", "ACME Corporation d/b/a Acme Tools", "SARL Boulangerie Dupont",
             "अरिहंत टेक्नोलॉजी स्टोर्स प्राइवेट लिमिटेड", "urbangrill.com", "Food & Frèrês SARL"]
    addrs = ["No. 12, 4th Cross, Near SBI ATM, Jayanagar, Bengaluru 560041", "12 bis Av. des Champs-Élysées, 75008 Paris",
             "1200 N. Main St., Suite 4, Springfield, IL 62704-1234", "N/A, <NULL>"]
    recs = [{"entity_id": f"S2-{i}", "business_name": rnd.choice(names), "business_address": rnd.choice(addrs),
             "country": rnd.choice(["India", "US", "France"])} for i in range(40_000)]
    t = time.perf_counter()
    for r in recs:
        normalize_record(r)
    # ~35k/s idle on an i5-12500H; 20k/s failed twice while other jobs loaded the machine. The floor only has to
    # catch order-of-magnitude regressions (e.g. regexes compiled per call), not measure throughput.
    assert 40_000 / (time.perf_counter() - t) > 5_000
