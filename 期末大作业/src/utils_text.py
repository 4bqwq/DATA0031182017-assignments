import re

def normalize_text(text: str) -> str:
    """
    Basic text normalization:
    - Lowercase
    - Remove extra whitespace
    - (Add more cleaning logic here later)
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
