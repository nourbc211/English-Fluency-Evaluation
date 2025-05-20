# Avoid crash from torch introspection
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
import streamlit as st
import os
import shutil
import io
import torch
import torchaudio
import soundfile as sf

# App modules
from pipeline.segmenter import segment_audio
from pipeline.feature_extractor import generate_feature_file
from pipeline.predictor import load_model, predict_and_aggregate
from pipeline.transcriber import transcribe_all_audios
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

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

os.makedirs("input/audios", exist_ok=True)
recorded_audio_path = None

if audio_bytes is not None and len(audio_bytes) > 0:
    recorded_audio_path = os.path.join("input/audios", "recorded_audio.wav")
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
    os.path.join("input/audios", audio_file.name) if audio_file else None
)

if audio_file and not recorded_audio_path:
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    st.audio(audio_path)

# ---- If we have an audio, proceed ----
if audio_path:
    # ---- Whisper language detection (before segmenting) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-tiny"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    input_features = processor.feature_extractor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").input_features.to(device)
    lang_probs = model.detect_language(input_features)
    detected_lang = max(lang_probs, key=lang_probs.get)
    confidence = lang_probs[detected_lang]

    st.write(f"🌍 Detected language: **{detected_lang}** (Confidence: {confidence:.2f})")

    if detected_lang != "en" or confidence < 0.80:
        st.error("❌ This audio is not confidently in English — predicted fluency level is automatically set to **Low**.")
        st.success("Predicted Fluency Level: Low")
        st.stop()

    # ---- Clear segments/transcripts ----
    for folder in ["input/segments", "input/transcripts"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # ---- Segment audio ----
    segment_paths = segment_audio(audio_path, "input/segments")
    st.write(f"📊 Segmented into {len(segment_paths)} segments.")

    # ---- Transcribe segments ----
    transcribe_all_audios(audio_base_dir="input/segments", transcript_base_dir="input/transcripts")
    st.write("📝 Transcription complete.")

    # ---- Feature extraction ----
    X = generate_feature_file(audio_dir="input/segments", transcript_dir="input/transcripts")
    st.write("🔬 Feature extraction complete.")

    # ---- Load model + predict ----
    model, top_features = load_model()
    final_label, segment_labels = predict_and_aggregate(X, segment_paths, model, top_features)
    st.success(f"🧠 Predicted Fluency Level: {final_label}")