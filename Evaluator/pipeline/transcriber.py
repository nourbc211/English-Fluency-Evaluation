# In this script we will go through all the audios of our data and generate an english 
# transcription of them that will be saved in a folder having the same structure than audio_files

# Imports

import os
import json
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# -----------------------------
# Whisper Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

model_id = "openai/whisper-tiny.en" # slihtly faster than whisper-base
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

# -----------------------------
# Transcribe files keeping the same folder structure and filename format as the 
# audio_files folder

audio_base_dir = 'Evaluator/input/segments'
transcript_base_dir = 'Evaluator/input/transcripts'
def transcribe_with_structure(audio_path, audio_base_dir=audio_base_dir, transcript_base_dir=transcript_base_dir):
    # Determine relative subpath and make transcript path
    rel_path = os.path.relpath(audio_path, audio_base_dir)
    rel_dir = os.path.dirname(rel_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Create same subdirectory in transcripts/
    transcript_dir = os.path.join(transcript_base_dir, rel_dir)
    os.makedirs(transcript_dir, exist_ok=True)

    # Save file with _transcript.json suffix
    transcript_file = os.path.join(transcript_dir, base_name + '_transcript.json')
    
    # Skip if already exists
    if os.path.exists(transcript_file):
        print("Skipping: {transcript_file} already exists")
        return

    try:
        print(f"Transcribing: {audio_path}")
        result = pipe(audio_path)
        text = result["text"]
        with open(transcript_file, 'w') as f:
            json.dump({"text": text}, f)
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")



# -----------------------------
# Transcribe all files
def transcribe_all_audios(audio_base_dir=audio_base_dir, transcript_base_dir=transcript_base_dir):
    for subdir, _, files in os.walk(audio_base_dir):
        for file in files:
            audio_path = os.path.join(subdir, file)
            transcribe_with_structure(audio_path, audio_base_dir, transcript_base_dir)
    print("All transcripts saved.")



