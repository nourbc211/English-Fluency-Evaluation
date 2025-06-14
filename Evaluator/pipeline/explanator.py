#----- Script containing explanations of the features descriptive statistics -----

import numpy as np
import pandas as pd


# ----- Feature Descriptions -----
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

# ----- Descriptive Statistics -----
full_feature_path = "Evaluator/training_data/full_feat.npy"
full_labels_path = "Evaluator/training_data/full_labels.npy"

def compute_stats_by_class(features_path=full_feature_path, labels_path=full_labels_path):
    """
    Compute descriptive statistics for each class in the dataset.
    Returns a dictionary of {class_label: {mean, std, min, max}} with dicts as values.
    """
    X = np.load(features_path)
    y = np.load(labels_path)

    # Define the feature names manually (same order as X)
    feature_names = (
        [f"MFCC_{i}" for i in range(20)] + 
        ["RMSE", "ZCR", "Spectral_Flux", "Pitch_Mean", "Pitch_Std", "Speech_Rate",
         "Pause_Duration", "Num_Pauses", "Pause_Ratio", "Pause_too_long", "Avg_Sentence_Length",
         "Vocab_Richness", "Jitter", "Shimmer", "HNR"]
    )

    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y

    stats = {}
    for class_label in sorted(df['label'].unique()):
        group = df[df['label'] == class_label].drop(columns=["label"])
        stats[class_label] = {
            "mean": group.mean().to_dict(),
            "std": group.std().to_dict(),
            "min": group.min().to_dict(),
            "max": group.max().to_dict(),
        }

    return stats

def analyze_feature_results(feature_values, label, stats):
    """
    Analyze feature values against class statistics.
    Args:
        feature_values (dataframe): Feature values to analyze.
        stats (dict): Class statistics dictionary.
    Returns:
        dict: Analysis results for each class.
    """

    # Map the label to its correponding numeric value
    label_mapping = {"Low": 0, "Intermediate": 1, "High": 2}
    features_to_dict = feature_values.set_index("Feature").to_dict()["Value"]
    current_stats = stats[label_mapping[label]]
    results = []
    for feature in features_to_dict.keys():
        # Extract feature values and calculate z-score
        actual_value = features_to_dict[feature]
        mean = current_stats["mean"][feature]
        std = current_stats["std"][feature]
        min_val = current_stats["min"][feature]
        max_val = current_stats["max"][feature]
        z_score = (actual_value - mean) / std if std != 0 else 0.0
        
        # Qualitative assessment
        if actual_value < min_val:
            assessment = "⬇️ Below class range"
        elif actual_value > max_val:
            assessment = "⬆️ Above class range"
        elif abs(z_score) < 0.5:
            assessment = "✅ Close to class mean"
        elif z_score > 0:
            assessment = "➕ Slightly above average"
        else:
            assessment = "➖ Slightly below average"

        results.append({
            "Feature": feature,
            "Value": round(actual_value, 2),
            "Z-score": round(z_score, 2),
            "Assessment": assessment
        })

    return pd.DataFrame(results)
