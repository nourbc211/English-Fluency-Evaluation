# This script tests the language detection functionality of the pipeline.
from ..language_detector import detect_language  

# Sample test transcripts (generated with gpt)
examples = {
    # ✅ English examples
    "Short English": "Hi there!",
    "Casual English": "Hey, what’s up? Just chilling today.",
    "Formal English": "The purpose of this study is to investigate language detection accuracy.",
    "Technical English": "Machine learning models require labeled data for supervised training.",
    "English Email": "Dear John, I hope this message finds you well. Best regards, Mary",
    
    # 🇫🇷 French examples
    "Short French": "Salut !",
    "Casual French": "Ça va ? Je suis en train de regarder un film.",
    "Formal French": "L’objectif de cette étude est d’analyser la détection de la langue.",
    "Mixed French": "Je suis allé au store pour acheter du pain and some milk.",
    "French Email": "Cher Monsieur, je vous écris concernant votre dernière facture.",

    # 🔄 Mixed or edge cases
    "Very Short": "Hi",
    "Empty": "",
    "Code Mixed": "Bonjour, I would like un café s’il vous plaît.",
    "Unrelated Chunks": "Laptop souris baguette airplane bonjour",
    "Arabic": "هذا اختبار لاكتشاف اللغة باستخدام نص باللغة العربية.",
    "Spanish": "Hola, estoy aprendiendo francés y inglés al mismo tiempo."
}

print("🧪 Running language detection tests:\n")

for name, text in examples.items():
    # testing with both lingua and LLM
    lang, ratio = detect_language(text, use_llm=False)  
    print(f"Without LLM : {name:<15} | Detected: {lang:<8} | English ratio: {ratio:.2f}")
    lang_llm, ratio_llm = detect_language(text, use_llm=True)
    print(f"With LLM    : {name:<15} | Detected: {lang_llm:<8} | English ratio: {ratio_llm:.2f}\n")