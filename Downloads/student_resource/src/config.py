from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
DATA_SPLITS_DIR = PROJECT_ROOT / 'data_splits'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SUBMISSION_DIR = PROJECT_ROOT / 'submission'
MODELS_DIR = PROJECT_ROOT / 'models'

DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K_CANDIDATES = 25
BETA = 0.5

S1_ID_COL = 'source1_entity_id'
MATCHED_COL = 'matched_entity_ids'
CANDIDATE_COL = 'candidate_entity_ids'
DELIM = '	'
ID_SEP = ','

COMMON_BUSINESS_WORDS = {
    'the', 'and', 'inc', 'llc', 'ltd', 'corp', 'corporation', 'co', 'company', 
    'services', 'service', 'group', 'holdings', 'enterprises', 'pvt', 'limited',
    'center', 'centre', 'store', 'shop', 'mart', 'restaurant', 'cafe', 'hotel',
    'inn', 'auto', 'food', 'market', 'solutions', 'management', 'international'
}
