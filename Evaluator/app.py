import streamlit as st
import os
import shutil
import io
import soundfile as sf
import torch
import glob
import json
torch.classes.__path__ = []  # 🛠️ workaround to avoid Streamlit crash

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# App modules
from pipeline.segmenter import segment_audio
from pipeline.feature_extractor import generate_feature_file
from pipeline.predictor import load_model, predict_and_aggregate
from pipeline.transcriber import transcribe_all_audios
from pipeline.language_detector import detect_language


# dummy variable to avoid bugs
model_loaded = False

# ---- Streamlit layout ----
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image("Evaluator/logo_sm.png", width=300, use_container_width=True)
st.write("")
st.title("Speech Fluency Evaluator")

# ---- Audio recording ----
st.subheader("Record Audio (Browser)")
try:
    from audio_recorder_streamlit import audio_recorder
    audio_bytes = audio_recorder(
        text="Click to record (min. 5 seconds)",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone"
    )
except ImportError:
    audio_bytes = None
    st.info("To enable in-browser recording, install: pip install audio_recorder_streamlit")

# Path of the input folder
base_input_paths = "Evaluator/input"

# audios path
os.makedirs(base_input_paths + "/audios", exist_ok=True)
recorded_audio_path = None

# Store audio
if audio_bytes is not None and len(audio_bytes) > 0:
    recorded_audio_path = os.path.join(base_input_paths + "/audios", "recorded_audio.wav")
    with open(recorded_audio_path, "wb") as f:
        f.write(audio_bytes)

    # Check duration
    audio_buffer = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_buffer)
    duration = len(data) / samplerate
    st.audio(recorded_audio_path)
    if duration < 5:
        st.error("⏱️ Your recording is too short! Please record at least 5 seconds.")
        st.stop()

# ---- File upload ----
st.subheader("Or Upload a WAV audio file")
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

audio_path = recorded_audio_path if recorded_audio_path else (
    os.path.join(base_input_paths + "/audios", audio_file.name) if audio_file else None
)

if audio_file and not recorded_audio_path:
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    st.audio(audio_path)

# ---- If we have an audio, proceed ----
if audio_path:
    # ---- Clear segments/transcripts ----
    for folder in [base_input_paths + "/segments", base_input_paths + "/transcripts"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # ---- Segment audio ----
    segment_paths = segment_audio(audio_path, base_input_paths + "/segments")
    st.write(f"📊 Segmented into {len(segment_paths)} segments.")

    # ---- Transcribe segments ----
    transcribe_all_audios()
    st.write("📝 Transcription complete.")

    # -- Language detection ----
    # Aggregating transcripts for language detection
    transcripts = []
    base_transcript_path = "Evaluator/input/transcripts"
    for transcript_file in glob.glob(os.path.join(base_transcript_path, "*.json")):
        with open(transcript_file, "r") as f:
            data = json.load(f)
            transcripts.append(data["text"])
    
    # Detect language of the full transcript
    language, english_ratio = detect_language(" ".join(transcripts))
   
   # If language is not English, we skip the evaluation and final label is automatically set to Low
    if language != "english":
        st.warning(f"⚠️ Detected language is '{language}'. Evaluation will be skipped.")
        st.success("🧠 Predicted Fluency Level: Low")
        model_loaded = True
        st.stop()  # Stop further processing

    # If language is English, we proceed with the evaluation
    st.success(f"✅ Detected language is '{language}'. Proceeding with evaluation...")

    # ---- Feature extraction ----
    X = generate_feature_file()
    st.write("🔬 Feature extraction complete.")

    # Model prediction and aggregation
    model, top_features = load_model()
    model_loaded = True
    final_label, segment_labels = predict_and_aggregate(X, model, top_features)


    st.success(f"🧠 Predicted Fluency Level: {final_label}")


# -----Feature explanation graph-----
if model_loaded : 

    full_feature_list = (
    [f"MFCC_{i}" for i in range(20)] +              # 20 MFCCs
    ["RMSE", "ZCR", "Spectral_Flux", "Pitch_Mean", "Pitch_Std", "Speech_Rate",
    "Pause_Duration", "Num_Pauses", "Pause_Ratio", "Pause_too_long", "Avg_Sentence_Length",
    "Vocab_Richness", "Jitter", "Shimmer", "HNR"]  # 5 pronunciation features
)
    df = pd.DataFrame(X, columns=full_feature_list)  # list from original extractor
    X_selected = df[top_features]

    # Column names
    feature_names = top_features

    # Get the mean feature values across segments
    mean_features = X_selected.mean(axis=0)

    # Create DataFrame
    df_features = pd.DataFrame({
        "Feature": feature_names,
        "Value": mean_features
    })

    # ---- Plot feature values ----
    st.subheader("🧪 Feature Values of the Evaluated Audio")

    # Bar chart
    st.bar_chart(df_features.set_index("Feature"))

    # show data table as well
    with st.expander("Show raw feature values"):
        st.dataframe(df_features)

