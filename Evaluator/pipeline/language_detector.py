from lingua import Language, LanguageDetectorBuilder
import re

# Build detector
detector = LanguageDetectorBuilder.from_all_languages().build()

def english_word_ratio(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    english_vocab = set(["the", "and", "is", "to", "of", "you", "this", "that", "have", "it"])  # fallback without NLTK
    if not tokens:
        return 0.0
    english_count = sum(1 for token in tokens if token in english_vocab)
    return english_count / len(tokens)

def detect_language(text):
    if len(text.split()) < 3:
        return "unknown", 0.0, 0.0

    detected = detector.detect_language_of(text)
    confidence = detector.compute_language_confidence(text, Language.ENGLISH)
    eng_ratio = english_word_ratio(text)
    
    return detected.name.lower(), confidence, eng_ratio