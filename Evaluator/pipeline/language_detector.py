from lingua import Language, LanguageDetectorBuilder
import re
import os



# Build detector (lingua)
detector = LanguageDetectorBuilder.from_all_languages().build()

# ---- Helper function to calculate English word ratio ----
def english_word_ratio(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    english_vocab = set(["the", "and", "is", "to", "of", "you", "this", "that", "have", "it"])  # fallback without NLTK
    if not tokens:
        return 0.0
    english_count = sum(1 for token in tokens if token in english_vocab)
    return english_count / len(tokens)


    
# ---- Language detection function ----
def detect_language(text):

    # Use lingua for language detection
    if len(text.strip().split()) < 3:
        return "unknown", 0.0 
    
    detected = detector.detect_language_of(text)
    eng_ratio = english_word_ratio(text)
    
    return detected.name.lower(), eng_ratio



    