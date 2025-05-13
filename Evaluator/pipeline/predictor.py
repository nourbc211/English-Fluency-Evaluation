# imports
import joblib
import json
import numpy as np
import pandas as pd


# This list must exactly match the output of extract_all_features()
full_feature_list = (
    [f"MFCC_{i}" for i in range(20)] +              # 20 MFCCs
    ["RMSE", "ZCR", "Spectral_Flux", "Pitch_Mean", "Pitch_Std", "Speech_Rate",
    "Pause_Duration", "Num_Pauses", "Pause_Ratio", "Pause_too_long", "Avg_Sentence_Length",
    "Vocab_Richness", "Jitter", "Shimmer", "HNR"]  # 5 pronunciation features
)

def load_model():
    """
    Method to load the model and the top features that were used.
    """
    model = joblib.load("model/xgb_model.pkl")
    with open("model/top_features.json") as f:
        top_features = json.load(f)
    return model, top_features

def predict_and_aggregate(X, segment_paths, model, top_features):
    """
    Predicting and aggregating the results.
    """
    df = pd.DataFrame(X, columns=full_feature_list)  # list from original extractor
    X_selected = df[top_features]

    preds = model.predict(X_selected)
    labels = np.array(['Low', 'Intermediate', 'High'])[preds]

    majority = pd.Series(labels).value_counts().idxmax()
    return majority, labels