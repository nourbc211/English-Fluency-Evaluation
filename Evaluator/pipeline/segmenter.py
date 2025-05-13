# imports
import os
import librosa
import soundfile as sf

# Segmentation method
def segment_audio(input_path, output_dir, segment_duration=5.0):
    """
    Method that segments an audio file into smaller segments of a specified duration and saves them as separate files.
    Args:
        input_path (str): Path to the input audio file.
        output_dir (str): Directory to save the segmented audio files.
        segment_duration (float): Duration of each segment in seconds : We trained our model with audios of 5seconds, so we split
        it that way here too.
    Returns: the list of paths to the segmented audio files
    """
    os.makedirs(output_dir, exist_ok=True)
    print(input_path)
    y, sr = librosa.load(input_path, sr=None)
    segment_samples = int(segment_duration * sr)

    segments = []
    for i in range(0, len(y), segment_samples):
        segment = y[i:i+segment_samples]
        if len(segment) < segment_samples:
            break
        segment_path = os.path.join(output_dir, f"segment_{i//segment_samples}.wav")
        sf.write(segment_path, segment, sr)
        segments.append(segment_path)
    return segments