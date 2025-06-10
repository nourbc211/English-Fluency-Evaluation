from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

try:
    models = client.models.list()
    for m in models.data:
        print(m.id)
except Exception as e:
    print(f"❌ Error: {e}")