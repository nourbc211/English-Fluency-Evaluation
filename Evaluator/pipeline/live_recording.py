# ---- Live recording module for audio input via microphone ----
# This module allows users to record audio directly from their microphone and save it for further processing.
import os
import sounddevice as sd
import soundfile as sf

DATA_DIR = "input/audios"

def record_audio(duration=30, samplerate=44100):
    """Record audio for a specified duration."""
    try:
        # Record audio
        print('Start talking\n')
        recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        return recording, samplerate
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        return None, None

def save_audio(recording, samplerate, filename='recorded_audio.wav'):
    """Save the recorded audio to a file."""
    recordings_dir = os.path.join(DATA_DIR)
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    filepath = os.path.join(recordings_dir, filename)
    sf.write(filepath, recording, samplerate)
    return filepath