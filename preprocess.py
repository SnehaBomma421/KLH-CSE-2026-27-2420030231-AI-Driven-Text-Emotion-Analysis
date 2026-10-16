import re

# Common contractions for expansion
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'m": " am",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
    "cant": "cannot",
    "wont": "will not",
    "dont": "do not",
    "didnt": "did not",
    "wasnt": "was not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "arent": "are not",
    "isnt": "is not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "im": "I am",
    "youre": "you are",
    "theyre": "they are",
    "weare": "we are",
    "hes": "he is",
    "shes": "she is",
    "its": "it is",
    "whats": "what is",
    "theres": "there is",
}

# Slang abbreviations mapping to standard words
SLANG_MAP = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "y": "why",
    "omg": "oh my god",
    "idk": "I do not know",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "lol": "laugh out loud",
    "wtf": "what the fuck",
    "ppl": "people",
    "rn": "right now",
    "sry": "sorry",
    "gr8": "great",
    "b4": "before",
    
    # Mappings for commonly collapsed representations
    "soo": "so",
    "loove": "love",
    "aree": "are",
    "haate": "hate",
}

# English letters that commonly appear doubled in spelling (e.g. good, feel, happy)
ALLOWED_DOUBLES = {'e', 'o', 'l', 's', 't', 'p', 'f', 'd', 'g', 'm', 'n', 'r', 'c', 'b'}

def clean_repeated_chars(text: str) -> str:
    """
    Collapses characters that are repeated 3 or more times consecutive.
    If the character is in ALLOWED_DOUBLES, collapses to 2 characters (e.g., 'sooooo' -> 'soo', 'loooove' -> 'loove').
    Otherwise, collapses to 1 character (e.g., 'haaaate' -> 'hate', 'HAPPYYYY' -> 'HAPPY').
    """
    if not text:
        return text
    
    result = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        run_len = 1
        while i + run_len < n and text[i + run_len] == char:
            run_len += 1
        
        # If run length is 3 or more, reduce it
        if run_len >= 3:
            char_lower = char.lower()
            if char_lower in ALLOWED_DOUBLES:
                result.append(char * 2)
            else:
                result.append(char)
        else:
            result.append(char * run_len)
            
        i += run_len
    return "".join(result)

def clean_repeated_punctuation(text: str) -> str:
    """
    Collapses excessive repeated punctuation (3 or more) to a maximum of 2,
    preserving emotional cues without creating extreme patterns.
    """
    if not text:
        return text
    text = re.sub(r'!{3,}', '!!', text)
    text = re.sub(r'\?{3,}', '??', text)
    text = re.sub(r'\.{4,}', '...', text)
    return text

def case_preserved_replace(text: str, dictionary: dict) -> str:
    """
    Replaces keys in the dictionary with their values case-sensitively
    using word boundary matching.
    """
    for key, value in dictionary.items():
        # Match word bounds case-insensitively.
        # Use a lookahead/lookbehind or word boundary to find exact word matches.
        # We need to handle words with apostrophes inside the dictionary correctly.
        pattern = r'\b' + re.escape(key) + r'\b'
        
        def replace_match(match):
            matched_text = match.group(0)
            if matched_text.isupper():
                return value.upper()
            if matched_text[0].isupper():
                return value.capitalize()
            return value
        
        text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    return text

def normalize_text(text: str) -> str:
    """
    The complete preprocessing pipeline.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Basic space collapsing
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Collapse repeated punctuation (e.g. !!! -> !!)
    text = clean_repeated_punctuation(text)
    
    # 3. Collapse character repetitions of 3+ (e.g. haaaate -> hate, sooooo -> soo)
    text = clean_repeated_chars(text)
    
    # 4. Expand contractions
    text = case_preserved_replace(text, CONTRACTIONS)
    
    # 5. Expand slang and standard normalized words (e.g. soo -> so, u -> you)
    text = case_preserved_replace(text, SLANG_MAP)
    
    # 6. Final whitespace clean
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    # Quick visual tests
    test_cases = [
        "I HAAAAATE this sooo much!!!",
        "HAPPYYYY day!",
        "sooooo happy rn",
        "NOOOOO!!! u r areeeee annoying",
        "ugh!!!! i cant believe this",
        "OMG this is amazing",
        "u r good for nothing",
    ]
    print("Normalizing tests:")
    for tc in test_cases:
        print(f"Original:  {tc}")
        print(f"Normalized: {normalize_text(tc)}")
        print("-" * 40)
