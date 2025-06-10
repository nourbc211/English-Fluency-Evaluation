from langdetect import detect, DetectorFactory
import re

# Make langdetect deterministic
DetectorFactory.seed = 0

# --- Helper function to calculate English word ratio ---
def english_word_ratio(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    english_vocab = set(["the", "and", "is", "to", "of", "you", "this", "that", "have", "it"])
    if not tokens:
        return 0.0
    english_count = sum(1 for token in tokens if token in english_vocab)
    return english_count / len(tokens)

# --- Language detection function using langdetect ---
def detect_language(text):
    if len(text.strip().split()) < 3:
        return "unknown", 0.0

    try:
        detected_lang = detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        detected_lang = "unknown"

    eng_ratio = english_word_ratio(text)
    return detected_lang, eng_ratio

# --- Test samples ---
test_cases = {
    "Short English": "Hi there!",
    "Casual English": "Hey, how are you doing today?",
    "Formal English": "It is with great pleasure that I inform you of your selection.",
    "French": "Bonjour, comment allez-vous aujourd'hui?",
    "Spanish": "Hola, ¿cómo estás?",
    "German": "Guten Tag, wie geht es Ihnen?",
    "Arabic": "مرحبا كيف حالك",
    "Chinese": "你好，你今天怎么样？",
    "Mixed": "Hello, je m'appelle Nour and I live à Paris.",
    "Code": "if (x > 0) { return true; }",
    "Very Short": "Ok",
    "Empty": ""
}

# --- Run tests ---
print("🧪 LangDetect Testing\n")
for name, sample in test_cases.items():
    lang, ratio = detect_language(sample)
    print(f"{name:<15} | Detected: {lang:<8} | English ratio: {ratio:.2f} | Text: {sample}")