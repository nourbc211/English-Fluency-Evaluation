from segmenter import segment_audio
from feature_extractor import generate_feature_file
from predictor import load_model, predict_and_aggregate
from transcriber import transcribe_all_audios
from live_recording import record_audio, save_audio
import shutil
import os



"""
# Record audio via mic
recording, sr = record_audio(duration=10)
if recording is not None:
    recorded_path = save_audio(recording, sr, filename="live_input.wav")
    audio_files = [recorded_path] 
else:
    audio_files = []
"""

#Paths
audio_path = "Evaluator/input/audios/hugo.wav"
segment_dir = "Evaluator/input/segments"
transcript_dir = "Evaluator/input/transcripts"
lang_flags_path = "Evaluator/output/audio_features_lang_flags.txt"

# ------ Cleaning ------
def clear_input_folders():
    """
    Method to clean input folders before running the evaluation
    """
    folders_to_clear = [segment_dir, transcript_dir]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🧹 Cleared: {folder}")
        os.makedirs(folder)

# Cleaning the input folders
clear_input_folders()

# segment the audio and save in the segment_dir
segment_paths = segment_audio(audio_path, segment_dir)

# generating the transcripts
transcribe_all_audios(audio_base_dir=segment_dir, transcript_base_dir=transcript_dir)

# Feature extraction
X = generate_feature_file(audio_dir=segment_dir, transcript_dir=transcript_dir)

# Language flags
with open(lang_flags_path) as f:
    lang_flags = [int(line.strip()) for line in f]

# Model prediction and aggregation
model, top_features = load_model()
final_label, segment_labels = predict_and_aggregate(X, model, top_features, lang_flags)


# Final result
print(f"\n---- Final predicted fluency level ----: {final_label}\n")