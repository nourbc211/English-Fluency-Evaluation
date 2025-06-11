# ---- Language Detection Module ----
from lingua import Language, LanguageDetectorBuilder

# Build detector (lingua)
detector = LanguageDetectorBuilder.from_all_languages().build()

def english_language_score(text):
    """
    Returns the confidence score of English and the top-2 languages with their scores.
    """
    if len(text.strip().split()) < 3:
        return 0.0, []

    scores = detector.compute_language_confidence_values(text)
    score_dict = {lang.name.lower(): value for lang, value in scores}
    top_langs = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    english_score = score_dict.get("english", 0.0)
    return round(english_score, 2), top_langs[:2]  # (english_score, top_2_languages)
    
# ---- Language detection function ----
def detect_language(text):

    # Use lingua for language detection
    if len(text.strip().split()) < 3:
        return "unknown", 0.0 
    
    detected = detector.detect_language_of(text)
    lang_confidence = english_language_score(text)[0]
    
    return detected.name.lower(), lang_confidence



    