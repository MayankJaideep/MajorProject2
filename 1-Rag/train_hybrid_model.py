"""
Hybrid Outcome Prediction Training Script.
Trains two models:
1. Metadata Model: Ensemble (Random Forest + Gradient Boosting) on structured data.
2. Text Model: k-NN on TF-IDF vectors of case description.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


class HybridTrainer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Metadata components
        self.metadata_model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.outcome_encoder = LabelEncoder()
        
        # Text components
        self.text_model = None
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def prepare_metadata_features(self, df, fit=True):
        """Encode and scale metadata features"""
        categorical_features = ['court', 'judge', 'case_type']
        encoded_data = []
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    self.encoders[feature] = LabelEncoder()
                    # Handle potential mixed types by converting to string
                    encoded = self.encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    # Handle unseen categories
                    encoded = []
                    for val in df[feature].astype(str):
                        if val in self.encoders[feature].classes_:
                            encoded.append(self.encoders[feature].transform([val])[0])
                        else:
                            encoded.append(0) # Default to 0
                    encoded = np.array(encoded)
                encoded_data.append(encoded.reshape(-1, 1))
        
        # Add year if available
        if 'year' in df.columns:
            encoded_data.append(df['year'].values.reshape(-1, 1))
            
        X = np.hstack(encoded_data)
        
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X

    def train(self, data_path):
        print("📚 Loading data...")
        df = pd.read_csv(data_path)
        
        # Fill missing values
        df['description'] = df['description'].fillna('')
        df['title'] = df['title'].fillna('')
        # Use title if description is empty
        df['text_content'] = df.apply(lambda x: x['description'] if len(str(x['description'])) > 10 else x['title'], axis=1)
        
        # Encode outcomes
        y = self.outcome_encoder.fit_transform(df['outcome'])
        
        # Split data
        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        
        print("\n🔧 Training Metadata Model (Ensemble)...")
        X_meta_train = self.prepare_metadata_features(df_train, fit=True)
        X_meta_test = self.prepare_metadata_features(df_test, fit=False)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.metadata_model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        self.metadata_model.fit(X_meta_train, y_train)
        
        meta_acc = self.metadata_model.score(X_meta_test, y_test)
        print(f"   Metadata Model Accuracy: {meta_acc:.2%}")
        
        print("\n📖 Training Text Model (k-NN)...")
        X_text_train = self.tfidf.fit_transform(df_train['text_content'])
        X_text_test = self.tfidf.transform(df_test['text_content'])
        
        self.text_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        self.text_model.fit(X_text_train, y_train)
        
        text_acc = self.text_model.score(X_text_test, y_test)
        print(f"   Text Model Accuracy: {text_acc:.2%}")
        
        # Evaluate Hybrid
        print("\n🚀 Evaluating Hybrid Model...")
        meta_probs = self.metadata_model.predict_proba(X_meta_test)
        text_probs = self.text_model.predict_proba(X_text_test)
        
        # Weighted Average (0.6 Metadata + 0.4 Text)
        final_probs = (0.6 * meta_probs) + (0.4 * text_probs)
        y_pred = np.argmax(final_probs, axis=1)
        
        hybrid_acc = accuracy_score(y_test, y_pred)
        print(f"   🏆 Hybrid Model Accuracy: {hybrid_acc:.2%}")
        
        self.save_models()
        
    def save_models(self):
        print(f"\n💾 Saving models to {self.model_dir}...")
        joblib.dump(self.metadata_model, os.path.join(self.model_dir, 'metadata_model.pkl'))
        joblib.dump(self.text_model, os.path.join(self.model_dir, 'text_model.pkl'))
        joblib.dump(self.tfidf, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.encoders, os.path.join(self.model_dir, 'metadata_encoders.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'metadata_scaler.pkl'))
        joblib.dump(self.outcome_encoder, os.path.join(self.model_dir, 'outcome_encoder.pkl'))
        print("   Done!")

if __name__ == "__main__":
    trainer = HybridTrainer(model_dir="1-Rag/models")
    trainer.train("1-Rag/data/real_cases.csv")
