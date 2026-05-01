"""Any utility functions should go here"""

import re

def remove_reasoning(text: str) -> str:
    return re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()