
#  English Fluency Evaluator – SemantikMatch

This project provides an end-to-end modular pipeline to assess English fluency from spoken audio files. It was developed as part of the CRP (Company Research Project) at ESSEC Business School for SemantikMatch. The tool classifies speakers into three fluency levels — **Low**, **Intermediate**, or **High** — using a trained machine learning model and interpretable features.

---

## 🚀 Key Features

- ✅ Predicts fluency from `.wav` audio input (the audio must last at least 5 seconds)
- 🧠 Uses a trained **XGBoost classifier**
- 🔍 Extracts rich acoustic, prosodic, phonetic, and textual features
- 🔤 Transcribes speech using **Whisper**
- 🧾 Segments and aligns speech for finer analysis
- 🌐 Detects non-English audio and auto-assigns a “Low” label
- 📊 Outputs detailed explanations via feature comparison
- 🧪 Deployable as a [Streamlit app](https://semantikmatch-fluency-evaluator.streamlit.app) through the app.py script.

---

## 📁 Project Structure

```
Evaluator/
├── app.py                   # Streamlit frontend
├── segmenter.py             # Audio segmentation
├── transcriber.py           # Whisper-based transcription
├── language_detector.py     # Language detection
├── feature_extractor.py     # Feature computation
├── predictor.py             # Prediction + aggregation of the results by segments to give the final label
├── explain_results.py       # Explains results by comparing feature scores (only to facilitate interpretability)
├── model/
│   ├── xgboost_model.pkl    # Trained model
│   └── selected_features.json  # Top features used
├── output/                  # Stores temporary results
└── requirements.txt         # Python dependencies
```

---

## 🔍 Script Overview

| Script                   | Purpose |
|--------------------------|---------|
| `app.py`                 | Streamlit web interface for uploading audios, running predictions, and visualizing results. |
| `segmenter.py`           | Splits the input audio into 5-second segments to stay consistent with the way we trained our model. |
| `transcriber.py`         | Transcribes each audio segment using Whisper and stores the transcript in the input/transcripts folder. |
| `language_detector.py`   | Detects whether the audio is in English based on the combined transcript; assigns "Low" label if not. |
| `feature_extractor.py`   | Extracts a rich set of features (acoustic, lexical, phonetic, prosodic) from both the audio and transcript. |
| `predictor.py`           | Loads the trained model, predicts fluency for each segment, and aggregates segment scores via majority vote. |
| `explain_results.py`     | Visualizes how each feature influenced the predicted label by comparing it to reference means for each class. |
| `model/xgboost_model.pkl`| Pre-trained XGBoost classifier saved after training on the Avalinguo dataset. |
| `model/selected_features.json` | JSON file containing the names of the most significant features kept after feature selection and ablation. |
| `output/`                | Temporary directory used to store intermediate files (transcripts, segment scores, features).|

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/English-Fluency-Evaluator.git
cd English-Fluency-Evaluator

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit App

```bash
streamlit run app.py
```

Once running, open the local URL shown in your terminal (e.g., http://localhost:8501).

Or access the live app at:  
👉 **https://semantikmatch-fluency-evaluator.streamlit.app**

---

## 📥 Input Format

- File type: `.wav`
- Duration: at least 5 seconds
- Sampling rate: 16kHz (automatically handled)

---

## 📤 Output

- Predicted fluency level: **Low**, **Intermediate**, or **High**
- Feature comparison (on the streamlit)

---

## 🧪 How It Works (Pipeline Overview)

1. **Audio Input**: User uploads a `.wav` file.
2. **Segmentation** (`segmenter.py`): The audio is split into 5-second overlapping segments.
3. **Transcription** (`transcriber.py`): Each segment is transcribed using Whisper.
4. **Language Detection** (`language_detector.py`): Language is identified. If not English → label is “Low”.
5. **Feature Extraction** (`feature_extractor.py`): All acoustic, prosodic, phonetic, and lexical features are computed.
6. **Prediction** (`predictor.py`): Each segment is classified using the saved XGBoost model.
7. **Aggregation**: Final label = majority vote over segments.
8. **Explanation** (`explain_results.py`): Visual display of features and comparison with class-level norms.

---

## 🔧 Integration & Extension

SemantikMatch can integrate this module into its existing platform by:

1. **Embedding model inference scripts** (`predictor.py`, `feature_extractor.py`, etc.) into backend services.
2. **Using saved model** (`model/xgboost_model.pkl`) and feature set (`selected_features.json`) for consistent prediction.
3. **Containerizing the whole app** with Docker or wrapping it in a RESTful API via FastAPI or Flask for full production deployment.
4. **Customizing thresholds** and feature interpretations via the config files and JSON mappings.

---

## 👨‍🔧 Handover & Maintenance Plan

- The full pipeline is modular and well-commented for easy onboarding.
- The model and feature files are saved and reusable for future predictions.
- Code ownership will be transferred to the SemantikMatch tech team.
- Future iterations may include accent-specific calibration or GPT scoring.

---

## 🤝 Authors

- Nour Ben Cherif – ESSEC Business School  
- Hadi Hijazi – ESSEC Business School  

---

## 📄 License & Usage

This project was developed as a prototype under academic supervision. Please contact the SemantikMatch team for any licensing or commercial use.

---
