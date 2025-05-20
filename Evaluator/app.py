import streamlit as st
import os
import shutil
import io
import soundfile as sf
import torch
torch.classes.__path__ = []  # 🛠️ workaround to avoid Streamlit crash

# App modules
from pipeline.segmenter import segment_audio
from pipeline.feature_extractor import generate_feature_file
from pipeline.predictor import load_model, predict_and_aggregate
from pipeline.transcriber import transcribe_all_audios

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

    # ---- Feature extraction ----
    X = generate_feature_file()
    st.write("🔬 Feature extraction complete.")

    # ---- Load language flags ----
    lang_flags_path = "Evaluator/output/audio_features_lang_flags.txt"
    with open(lang_flags_path) as f:
        lang_flags = [int(line.strip()) for line in f]

    # Model prediction and aggregation
    model, top_features = load_model()
    final_label, segment_labels = predict_and_aggregate(X, model, top_features, lang_flags)
    st.success(f"🧠 Predicted Fluency Level: {final_label}")