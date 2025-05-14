# Language detector script
# language_detector.py
import fasttext
import os

MODEL_PATH = "Evaluator/lid.176.ftz"

# Download the model if it doesn't exist yet
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading FastText language ID model...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        MODEL_PATH
    )

# Load the model
lang_model = fasttext.load_model(MODEL_PATH)

# Detection model method
def detect_language(text, threshold=0.90):
    """
    Returns (lang_code, confidence, is_english_bool)
    """
    label, conf = lang_model.predict(text)
    lang = label[0].replace("__label__", "")
    confidence = float(conf[0])  # force cast to avoid numpy issues
    return lang, confidence, lang == "en" and confidence >= threshold