# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System
# 
# Description: Pure-Python string manipulation utilities for TTS chunking.
# ==============================================================================

import re

ABBREVIATIONS = {"Engr", "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "Asst", "Assoc", "Atty", "Hon"}
_SENTENCE_END_RE = re.compile(r'[.!?] |\n')

def get_last_valid_split(buffer: str) -> int:
    """
    Finds the last valid punctuation mark to split a sentence for TTS.
    Bulletproofed against markdown formatting and common abbreviations.
    """
    matches = list(_SENTENCE_END_RE.finditer(buffer))
    if not matches:
        return -1

    for m in matches:
        pre = buffer[:m.start()].strip()
        if not pre:
            continue
        
        preceding_words = pre.split()
        if not preceding_words:
            continue
        
        # Extract the raw word just before the period
        raw_word = preceding_words[-1]
        
        # Strip ALL markdown (*), quotes, and punctuation. 
        # If raw_word was "**D.", clean_word becomes "D".
        clean_word = "".join(filter(str.isalpha, raw_word))
        
        if not clean_word:
            continue
        
        # 1. Check if it's a middle initial (any single letter, ignoring case)
        is_initial = len(clean_word) == 1
        
        # 2. Check if it's a known abbreviation (using .title() to catch 'Engr', 'Dr', etc.)
        is_abbrev = clean_word.title() in ABBREVIATIONS
        
        # If it's an initial or abbreviation, SKIP this match and keep looking
        if is_initial or is_abbrev:
            continue
        
        # If we reach here, it's a true valid end of a sentence
        return m.end()

    return -1

def format_spoken_text(text: str) -> str:
    """Expands academic and professional abbreviations for the TTS engine."""
    return (text.replace("Engr.", "Engineer")
                .replace("Dr.", "Doctor")
                .replace("Asst.", "Assistant")
                .replace("Assoc.", "Associate")
                .replace("Prof.", "Professor")
                .replace("Ma.", "Maria")
                .replace("Atty.", "Attorney"))