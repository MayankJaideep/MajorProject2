# 🚀 Quick Start Guide - Outcome Prediction System

## Installation

The required dependencies have already been added to `requirements.txt`. If you haven't installed them yet:

```bash
cd /Users/mayankjaideep/Desktop/ai-driven-research-engine-main
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Full Application

```bash
streamlit run 1-Rag/app.py
```

Then ask questions like:
- *"What are my chances of winning a trademark case in Bombay High Court?"*
- *"Predict the outcome for a breach of contract in Delhi HC filed in 2024"*
- *"What is the success probability for a patent dispute in Supreme Court?"*

The agent will automatically:
1. Detect your prediction intent
2. Extract case features
3. Make a prediction
4. Display results with beautiful visualizations

### Option 2: Run the Demonstration Script

```bash
cd /Users/mayankjaideep/Desktop/ai-driven-research-engine-main
python3 1-Rag/demo_prediction.py
```

This will show 4 sample predictions without starting the full Streamlit app.

### Option 3: Use Programmatically

```python
from feature_extractor import FeatureExtractor
from outcome_predictor import OutcomePredictor

# Initialize
extractor = FeatureExtractor()
predictor = OutcomePredictor(model_dir="1-Rag/models")
predictor.load_model()

# Extract features
case_text = "Trademark infringement in Bombay High Court, 2023"
features = extractor.extract_features(case_text)

# Predict
result = predictor.predict(features)
print(f"Outcome: {result['predicted_outcome']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Files Created

### Core Components
- `1-Rag/generate_training_data.py` - Synthetic data generator
- `1-Rag/feature_extractor.py` - Feature extraction module
- `1-Rag/outcome_predictor.py` - ML model trainer/predictor
- `1-Rag/demo_prediction.py` - Demonstration script

### Data & Models
- `1-Rag/data/training_cases.csv` - 250 synthetic training cases
- `1-Rag/models/outcome_model.pkl` - Trained Random Forest model
- `1-Rag/models/feature_encoders.pkl` - Label encoders
- `1-Rag/models/feature_scaler.pkl` - Standard scaler
- `1-Rag/models/outcome_encoder.pkl` - Outcome encoder
- `1-Rag/models/feature_names.pkl` - Feature names

### Modified Files
- `1-Rag/agent.py` - Added `predict_case_outcome` tool
- `1-Rag/app.py` - Enhanced source detection
- `1-Rag/premium-style.css` - Added prediction visualization styles
- `requirements.txt` - Added ML dependencies

## Model Performance

- **Accuracy**: 54% (on synthetic data)
- **Training Set**: 200 cases
- **Test Set**: 50 cases
- **Features**: 6 (court, judge, case_type, legal_domain, year, complexity)
- **Outcomes**: 4 (plaintiff_win, defendant_win, settlement, dismissed)

## Next Steps

1. **Collect Real Data**: Replace synthetic data with actual case outcomes
2. **Retrain Model**: Improve accuracy to 70-80% with real data
3. **Add More Features**: Lawyer experience, case value, etc.
4. **Fine-tune UI**: Customize probability bar colors and animations

## Troubleshooting

**Model not found error?**
```bash
cd /Users/mayankjaideep/Desktop/ai-driven-research-engine-main
python3 1-Rag/outcome_predictor.py --train --data 1-Rag/data/training_cases.csv
```

**Missing dependencies?**
```bash
pip install scikit-learn pandas numpy joblib plotly
```

## Support

For detailed documentation, see [walkthrough.md](file:///Users/mayankjaideep/.gemini/antigravity/brain/a5725012-5a18-4d01-bc92-d77b74548269/walkthrough.md)
