import streamlit as st
import os
import shutil
from pipeline.segmenter import segment_audio
from pipeline.feature_extractor import generate_feature_file
from pipeline.predictor import load_model, predict_and_aggregate
from pipeline.transcriber import transcribe_all_audios
import soundfile as sf
#import sounddevice as sd
import numpy as np
import pandas as pd
import io

# Centered, banner-style logo using Streamlit columns
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image("Evaluator/logo_sm.png", width = 300, use_container_width=True)
st.write("") 
# title
st.title("Speech Fluency Evaluator")

# Option to record audio in-browser
st.subheader("Record Audio (Browser)")
try:
    from audio_recorder_streamlit import audio_recorder
    audio_bytes = audio_recorder(text="Click to record (min. 5 seconds)", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="microphone")
except ImportError:
    audio_bytes = None
    st.info("To enable in-browser recording, install streamlit-audio-recorder: pip install audio_recorder_streamlit")

# Ensure directory exists
os.makedirs("input/audios", exist_ok=True)

recorded_audio_path = None

if audio_bytes is not None and len(audio_bytes) > 0:
    recorded_audio_path = os.path.join("input/audios", "recorded_audio.wav")
    with open(recorded_audio_path, "wb") as f:
        f.write(audio_bytes)

    # Check audio duration
    audio_buffer = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_buffer)
    duration = len(data) / samplerate

    if duration < 5:
        st.audio(recorded_audio_path)
        st.error("⏱️ Your recording is too short! Please record at least 5 seconds.")
        st.stop()

    st.audio(recorded_audio_path)

# Upload audio file
st.subheader("Or Upload a WAV audio file")
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

# Choose which audio to process
audio_path = None
if recorded_audio_path:
    audio_path = recorded_audio_path
elif audio_file is not None:
    audio_path = os.path.join("input/audios", audio_file.name)
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    st.audio(audio_path)

if audio_path:
    # Clear previous segments and transcripts
    for folder in ["input/segments", "input/transcripts"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Segment audio
    segment_paths = segment_audio(audio_path, "input/segments")
    st.write(f"Segmented into {len(segment_paths)} segments.")

    # Transcribe segments
    transcribe_all_audios(audio_base_dir="input/segments", transcript_base_dir="input/transcripts")
    st.write("Transcription complete.")

    # Feature extraction
    X = generate_feature_file(audio_dir="input/segments", transcript_dir="input/transcripts")
    st.write("Feature extraction complete.")

    # Prediction
    model, top_features = load_model()
    final_label, segment_labels = predict_and_aggregate(X, segment_paths, model, top_features)

    st.success(f"Predicted Fluency Level: {final_label}")
