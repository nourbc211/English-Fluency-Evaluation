# We here replicate the work we did in our Feature Extraction Notebook.

# Imports
import os
import numpy as np
import librosa
import parselmouth
import spacy
import json

# Directories paths
segments_dir = 'Evaluator/input/segments/'
transcripts_dir = 'Evaluator/input/transcripts'

# --- Transcript loader ---
def load_transcript(audio_path, audio_base_dir=segments_dir, transcript_base_dir=transcripts_dir):
    rel_path = os.path.relpath(audio_path, audio_base_dir)  # relative to base
    rel_dir = os.path.dirname(rel_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Final path: transcripts/.../filename.wav_transcript.json
    transcript_path = os.path.join(transcript_base_dir, rel_dir, base_name + "_transcript.json")

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found at: {transcript_path}")

    with open(transcript_path, "r") as f:
        return json.load(f)["text"]

# --- Text feature extraction ---

nlp = spacy.load("en_core_web_sm")


def extract_text_features(text):
    """
    We extract the average sentence length and vocabulary richness from the text.
    """
    doc = nlp(text)
    avg_sentence_length = np.mean([len(sent) for sent in doc.sents]) if doc.sents else 0
    vocab_richness = len(set(token.text for token in doc if token.is_alpha)) / len(doc) if len(doc) > 0 else 0
    return [avg_sentence_length, vocab_richness]


# --- Acoustic features (e.g., MFCCs, ZCR, Spectral Flux, RMSE) ---
def extract_acoustic_features(file_path):
    import librosa
    import numpy as np
    y, sr = librosa.load(file_path, sr=None)
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    duration = librosa.get_duration(y=y, sr=sr)
    speech_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / duration

    return np.hstack([mfcc, rmse, zcr, spectral_flux, pitch_mean, pitch_std, speech_rate])

# --- Pause-related features ---
def extract_pause_features(y, sr):
    """
    We extract pause-related features from the audio file : total pause time, number of pauses, pause ratio, and number of pauses longer than 1 second.
    """
    # TODO : Check if we can remove / improve pause_too_long
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    for i in range(1, len(intervals)):
        prev_end = intervals[i - 1][1]
        current_start = intervals[i][0]
        pause_duration = (current_start - prev_end) / sr
        pauses.append(pause_duration)
    num_pauses = len(pauses)
    total_pause_time = sum(pauses)
    pause_ratio = total_pause_time / (len(y) / sr)
    too_long = sum(p > 1.0 for p in pauses)
    return [total_pause_time, num_pauses, pause_ratio, too_long]

# --- Pronunciation features using Parselmouth (Praat) ---
def extract_pronunciation_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # Pronunciation metrics
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        return [jitter, shimmer, hnr]

    except Exception as e:
        print(f"Error extracting pronunciation from {file_path}: {e}")
        return [0.0, 0.0, 0.0]  # fallback values

# --- Unified feature extraction function ---
def extract_all_features(file_path, audio_base_dir, transcript_base_dir):
    """
    We put together all the features
    """
    y, sr = librosa.load(file_path, sr=None)

    # Load transcript
    try:
        text = load_transcript(file_path, audio_base_dir, transcript_base_dir)
    except Exception as e:
        print(f"Could not load transcript for {file_path}: {e}")
        text = ""

    # Extract all feature types
    acoustic = extract_acoustic_features(file_path)
    pauses = extract_pause_features(y, sr)
    text_features = extract_text_features(text) if text else [0, 0]
    pronunciation = extract_pronunciation_features(file_path)

    features = np.hstack([acoustic, pauses, text_features, pronunciation])
    return features

# We add a function to generate the file with all the features
def generate_feature_file(audio_dir=segments_dir, transcript_dir=transcripts_dir, output_path="Evaluator/output/audio_features.npy"):
    feature_list = []
    file_list = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            file_path = os.path.join(root, file)
            features = extract_all_features(file_path, audio_dir, transcript_dir)
            feature_list.append(features)
            file_list.append(file_path)

    features_array = np.array(feature_list)
    
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features_array)
    print(f"✅ Saved features for {len(file_list)} audio files to {output_path}")
    return features_array 

