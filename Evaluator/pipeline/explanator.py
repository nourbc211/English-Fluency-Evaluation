#----- Script containing explanatory methods for the features-----

FEATURE_DESCRIPTIONS = {
    # Acoustic Features
    "MFCC": "These capture timbral and phonetic information of speech. Important for pronunciation quality.",
    "RMSE": "Root Mean Square Energy — reflects loudness. Higher = louder speech.",
    "ZCR": "Zero Crossing Rate — how often the signal crosses the zero line. Higher values may indicate noisier or more energetic speech.",
    "Spectral_Flux": "Measures how quickly the power spectrum of a signal is changing — higher flux indicates more dynamic speech.",
    "Pitch_Mean": "Average pitch. Too high or low could signal tension or discomfort.",
    "Pitch_Std": "Pitch variation — expressive speakers tend to vary pitch more.",

    # Temporal / Rhythm
    "Speech_Rate": "Words per second. Higher rates typically indicate fluency.",
    "Pause_Duration": "Average length of pauses. Long pauses suggest hesitation.",
    "Num_Pauses": "Number of pauses — too many can indicate disfluency.",
    "Pause_Ratio": "Ratio of silence to speech — higher ratio may indicate slower or more hesitant speech.",
    "Pause_too_long": "Binary indicator of whether very long pauses were present (>1 sec).",

    # Lexical / Textual
    "Avg_Sentence_Length": "Average number of words per sentence. Too short = simple structure; too long = complex grammar.",
    "Vocab_Richness": "Diversity of vocabulary — high richness is associated with advanced speakers.",

    # Pronunciation
    "Jitter": "Variation in pitch period — higher jitter suggests vocal instability.",
    "Shimmer": "Amplitude variation — high shimmer can indicate unsteady voice.",
    "HNR": "Harmonics-to-Noise Ratio — higher values = clearer voice with less breathiness.",
}

