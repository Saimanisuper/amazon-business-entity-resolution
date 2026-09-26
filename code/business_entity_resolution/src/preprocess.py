import re
import unicodedata

# Common legal and corporate suffixes across US, India, and France
LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b('
    # US & General
    r'incorporated|corporation|inc|corp|company|co|llc|limited liability company|lp|llp|'
    # India
    r'private limited|pvt ltd|pvt\.?|private|ltd\.?|limited|'
    # France
    r'sarl|sci|sas|sasa|eurl|snc|gie|societe anonyme|societe'
    r')\b',
    re.IGNORECASE
)

# Address abbreviations dictionary
STREET_ABBREVIATIONS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave?\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\br\.\b': 'rue',
    r'\brue\b': 'rue',
    r'\bbd\b': 'boulevard',
}


def strip_accents(text: str) -> str:
    """Normalize unicode characters (e.g., French accents é, è -> e)."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFKD', text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_name(name: str, remove_suffix: bool = True) -> str:
    """
    Clean and canonicalize a business name.
    1. Strip accents & lowercase
    2. Replace symbols (& -> and)
    3. Remove legal corporate suffixes
    4. Remove non-alphanumeric noise
    """
    if not name or not isinstance(name, str):
        return ""
    
    text = strip_accents(name).lower()
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[\'\"«»<>\[\]\(\)\{\}]', ' ', text)
    
    if remove_suffix:
        text = LEGAL_SUFFIX_PATTERN.sub(' ', text)
    
    # Keep alphanumeric words
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())


def clean_address(addr: str) -> str:
    """
    Clean and canonicalize a business address.
    1. Strip accents & lowercase
    2. Expand common street abbreviations
    3. Standardize whitespace and remove excessive punctuation
    """
    if not addr or not isinstance(addr, str):
        return ""
    
    text = strip_accents(addr).lower()
    text = re.sub(r'[,;:\'\"«»<>\[\]\(\)\{\}\-]', ' ', text)
    
    for pattern, replacement in STREET_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
        
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())


def extract_numbers(text: str) -> set:
    """Extract numeric sequences (postal codes, street numbers, door numbers)."""
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))


def get_name_tokens(name: str) -> list:
    """Return clean alphanumeric tokens of length >= 2."""
    cleaned = clean_name(name)
    return [t for t in cleaned.split() if len(t) >= 2]
