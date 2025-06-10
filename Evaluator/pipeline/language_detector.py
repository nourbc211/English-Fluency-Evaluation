from lingua import Language, LanguageDetectorBuilder
import re
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os


# API key for OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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


client = OpenAI()
# ---- Alternative approach using LLMs ----

def llm_english_detector(text):
    if len(text.split()) < 3:
        return "unknown", 0.0

    prompt = f"""
    You are a language detection assistant. Given this transcript:
    \"{text}\"
    Determine if the main language is English. 
    Respond with only one word: "Yes" or "No".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "yes"
    except Exception as e:
        print(f"Error in LLM language detection: {e}")
        return False

    
# ---- Language detection function ----
def detect_language(text, use_llm=False):

    # Use lingua for language detection
    if len(text.strip().split()) < 3:
        return "unknown", 0.0 

    if use_llm:
        is_english = llm_english_detector(text)
        eng_ratio = english_word_ratio(text)
        return ("en",eng_ratio) if is_english else ("unknown", 0.0)
    
    
    detected = detector.detect_language_of(text)
    eng_ratio = english_word_ratio(text)
    
    return detected.name.lower(), eng_ratio



    