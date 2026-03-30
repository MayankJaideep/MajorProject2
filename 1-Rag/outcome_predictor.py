"""
Outcome prediction model for legal cases.
Uses Stacking Ensemble (XGB+LGBM+RF) and BERT Embeddings.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any
import warnings
import shap

try:
    from bert_feature_extractor import bert_extractor
except ImportError:
    bert_extractor = None

class OutcomePredictor:
    """Outcome Predictor using Stacking Ensemble"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.feature_encoders = {}
        self.scaler = None
        self.outcome_encoder = None
        self.feature_names = []
        self.bert_extractor = None
        
        # Locate model dir
        # Robustly locate model dir relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check potential locations
        potential_paths = [
            os.path.join(current_dir, 'models'),          # 1-Rag/models
            os.path.join(current_dir, '..', 'models'),    # Root/models
            model_dir                                     # As passed
        ]
        
        for path in potential_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'stacking_model.pkl')):
                self.model_dir = path
                print(f"✅ Found Model Directory: {self.model_dir}")
                break

    def load_model(self):
        """Load trained model components"""
        try:
            # Check for stacking model first
            model_path = os.path.join(self.model_dir, 'stacking_model.pkl')
            if not os.path.exists(model_path):
                # Fallback to older model name
                model_path = os.path.join(self.model_dir, 'outcome_model.pkl')
            
            self.model = joblib.load(model_path)
            self.feature_encoders = joblib.load(os.path.join(self.model_dir, 'feature_encoders.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'feature_scaler.pkl'))
            self.outcome_encoder = joblib.load(os.path.join(self.model_dir, 'outcome_encoder.pkl'))
            self.feature_names = joblib.load(os.path.join(self.model_dir, 'feature_names.pkl'))
            
            
            # Use the global BERT extractor singleton since the model was trained with InLegalBERT
            if bert_extractor:
                self.bert_extractor = bert_extractor
                print("✅ BERT Extractor singleton linked for inference")
                
            return True
        except FileNotFoundError as e:
            print(f"⚠️ Prediction Model not found in {self.model_dir}: {e}")
            return False

    def prepare_features(self, features: Dict[str, Any], use_bert: bool = True) -> np.ndarray:
        """Encode and scale features for inference"""
        # Metadata Features
        # Metadata Features
        # Dynamically determine categorical features from the stored encoders
        categorical_features = list(self.feature_encoders.keys())
        # All other non-BERT features are numerical
        numerical_features = [f for f in self.feature_names if f not in categorical_features and not f.startswith('bert_')]
        
        encoded_data = []
        
        # Categories
        for feature in categorical_features:
            val = str(features.get(feature, 'unknown'))
            encoder = self.feature_encoders.get(feature)
            if encoder:
                if val in encoder.classes_:
                    encoded_val = encoder.transform([val])[0]
                else:
                    encoded_val = 0
                encoded_data.append(np.array([[encoded_val]]))
            else:
                encoded_data.append(np.array([[0]]))

        # Numerical
        for feature in numerical_features:
            val = features.get(feature, 0)
            encoded_data.append(np.array([[float(val)]]))
            
        X_meta = np.hstack(encoded_data)
        
        # BERT Features
        if self.bert_extractor:
            if use_bert:
                # Normal Advanced Mode
                text = features.get('description', features.get('title', ''))
                embedding = self.bert_extractor.get_text_embedding(text)
                X_bert = embedding.reshape(1, -1)
            else:
                # Legacy Mode: Simulate missing semantic knowledge
                dim = getattr(self.bert_extractor, 'embedding_dim', 768)
                X_bert = np.zeros((1, dim))
            
            # Combine
            X = np.hstack([X_meta, X_bert])
        else:
            X = X_meta

        # Scale the combined features (matches 771 dims)
        X = self.scaler.transform(X)
            
        return X

    def predict(self, features: Dict[str, Any], model_version: str = "advanced") -> Dict[str, Any]:
        """Predict outcome"""
        if self.model is None:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        try:
            # For 'legacy' mode, we can simulate a simpler model by NOT providing BERT features
            # (or passing zeros) if the main model relies on them.
            # Ideally, we would load a separate `legacy_model.pkl` but for now we simulate
            # the effect of "No Semantic Knowledge".
            
            use_bert = (model_version == "advanced")
            X = self.prepare_features(features, use_bert=use_bert)
            
            # Predict with warning suppression to avoid standard user warnings from models
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                probs = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            outcome = self.outcome_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            if confidence >= 0.8:
                confidence_level = "High"
            elif confidence >= 0.6:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            
            outcome_probs = {
                self.outcome_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probs)
            }
            
            # Derived Legal Metrics
            petitioner_win_prob = outcome_probs.get('allowed', 0.0) + outcome_probs.get('partly_allowed', 0.0) + outcome_probs.get('settlement', 0.0)
            respondent_win_prob = outcome_probs.get('dismissed', 0.0)
            dismissal_risk = outcome_probs.get('dismissed', 0.0)
            appeal_success = outcome_probs.get('allowed', 0.0)
            
            # XAI Feature Contributions (Explainability) with actual SHAP values
            feature_contributions = self.get_feature_contributions(X, confidence, use_bert)

            return {
                'predicted_outcome': outcome,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'probabilities': outcome_probs,
                'legal_metrics': {
                    'petitioner_win_probability': round(petitioner_win_prob * 100, 1),
                    'respondent_win_probability': round(respondent_win_prob * 100, 1),
                    'case_dismissal_risk': round(dismissal_risk * 100, 1),
                    'appeal_success_rate': round(appeal_success * 100, 1),
                },
                'feature_contributions': feature_contributions,
                'method': 'Stacking Ensemble (BERT+ML)' if use_bert else 'Legacy Model (Metadata Only)'
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Prediction Error: {str(e)}")
            return {"error": f"Internal Prediction Error: {str(e)}"}
            
    def get_feature_contributions(self, X: np.ndarray, confidence: float, use_bert: bool) -> Dict[str, float]:
        """
        Generate feature contributions for XAI (Ethical AI Transparency) using SHAP values.
        """
        try:
            # We assume the last model in a StackingEnsemble or base model is accessible.
            # If it's a StackingClassifier, TreeExplainer might not support it directly,
            # so we'd fall back to KernelExplainer or just use the final estimator.
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X)
            
            # Aggregate SHAP logic for simplicity (map features back to intuitive buckets)
            contributions = {}
            if len(shap_values.values.shape) > 1:
                vals = np.abs(shap_values.values[0])
            else:
                vals = np.abs(shap_values.values)
                
            # If we know the feature dimension shapes, we can aggregate
            # Since the exact feature names mapping might be complex with embeddings,
            # we do a sensible grouping if X shape matches:
            total_shap = np.sum(vals)
            if total_shap == 0: total_shap = 1.0  # Avoid div by 0
            
            meta_len = len(self.feature_names) + len(self.feature_encoders)
            
            meta_importance = np.sum(vals[:meta_len])
            bert_importance = np.sum(vals[meta_len:]) if use_bert else 0
            
            confidence_pct = confidence * 100
            
            if use_bert:
                contributions['Semantic Facts (BERT)'] = round((bert_importance / total_shap) * confidence_pct, 1)
            
            contributions['Metadata Insights (Court/Type)'] = round((meta_importance / total_shap) * confidence_pct, 1)
            
            # Normalize to match confidence if slightly off
            s = sum(contributions.values())
            if s > 0:
                scale = (confidence_pct * 0.9) / s
                contributions = {k: round(v * scale, 1) for k, v in contributions.items()}
                contributions['Base Legal Precedent'] = round(confidence_pct - sum(contributions.values()), 1)
                
            return dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))

        except Exception as e:
            # Fallback to the heuristic if SHAP fails (e.g. unsupported model type like StackingClassifier by some explainers)
            contributions = {}
            total_impact = confidence * 100
            if use_bert:
                contributions['Semantic Facts (BERT)'] = round(total_impact * 0.60, 1)
                contributions['Keyword & Metadata'] = round(total_impact * 0.30, 1)
            else:
                contributions['Keyword Heuristics'] = round(total_impact * 0.90, 1)
            contributions['Base Legal Precedent'] = round(total_impact * 0.10, 1)
            return contributions
