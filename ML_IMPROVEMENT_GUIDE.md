# ML Model Improvement Guide: Achieving 80-90% Accuracy

## Current Status
- **Model**: Random Forest Classifier
- **Current Accuracy**: 54% (on synthetic data)
- **Target Accuracy**: 80-90%
- **Main Issue**: Training on synthetic data instead of real case outcomes

---

## 🎯 Strategy Overview

To achieve 80-90% accuracy, you need to:

1. **Collect Real Data** (Most Important - 40% impact)
2. **Engineer Better Features** (30% impact)
3. **Optimize Model Architecture** (20% impact)
4. **Handle Class Imbalance** (10% impact)

---

## 📊 Step 1: Collect Real Case Data

### **Option A: Web Scraping Indian Kanoon (Recommended)**

Indian Kanoon has thousands of cases with outcomes. Here's how to scrape them:

```python
# scrape_indian_kanoon.py
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

def scrape_cases(num_cases=500):
    """
    Scrape real case data from Indian Kanoon
    """
    cases = []
    api_token = os.getenv("INDIAN_KANOON_API_TOKEN")
    
    # Search for commercial court cases
    search_queries = [
        "commercial court breach of contract",
        "trademark infringement",
        "patent dispute",
        "arbitration award",
        "shareholder dispute"
    ]
    
    for query in search_queries:
        url = "https://api.indiankanoon.org/search/"
        headers = {"Authorization": f"Token {api_token}"}
        params = {
            "formInput": query,
            "pagenum": 0
        }
        
        for page in range(10):  # 10 pages per query
            params["pagenum"] = page
            response = requests.post(url, headers=headers, data=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for doc in data.get('docs', []):
                    case = extract_case_details(doc)
                    if case:
                        cases.append(case)
            
            time.sleep(1)  # Rate limiting
    
    return pd.DataFrame(cases)

def extract_case_details(doc):
    """
    Extract structured data from case document
    """
    # Parse case text to extract:
    # - Court name
    # - Judge name
    # - Case type
    # - Year
    # - Outcome (plaintiff win/defendant win/settlement/dismissed)
    
    text = doc.get('headline', '') + ' ' + doc.get('title', '')
    
    # Use regex or NLP to extract outcome
    outcome = extract_outcome(text)
    
    if outcome:
        return {
            'case_id': doc.get('tid'),
            'court': extract_court(text),
            'judge': extract_judge(text),
            'case_type': extract_case_type(text),
            'year': extract_year(text),
            'outcome': outcome,
            'text': text
        }
    
    return None
```

**Outcome Extraction Patterns:**
```python
def extract_outcome(text):
    """
    Extract case outcome from text using keywords
    """
    text_lower = text.lower()
    
    # Plaintiff win indicators
    if any(word in text_lower for word in [
        'petition allowed', 'appeal allowed', 'suit decreed',
        'plaintiff succeeds', 'in favor of plaintiff',
        'injunction granted', 'damages awarded'
    ]):
        return 'plaintiff_win'
    
    # Defendant win indicators
    elif any(word in text_lower for word in [
        'petition dismissed', 'appeal dismissed', 'suit dismissed',
        'in favor of defendant', 'defendant succeeds'
    ]):
        return 'defendant_win'
    
    # Settlement indicators
    elif any(word in text_lower for word in [
        'settlement', 'compromise', 'consent decree',
        'parties agreed', 'amicable resolution'
    ]):
        return 'settlement'
    
    # Dismissed indicators
    elif any(word in text_lower for word in [
        'dismissed for default', 'dismissed for non-prosecution',
        'withdrawn', 'struck off'
    ]):
        return 'dismissed'
    
    return None
```

---

### **Option B: Manual Data Collection**

If API access is limited, manually collect data:

1. **Visit**: https://indiankanoon.org
2. **Search**: "Commercial Court" + case type
3. **Extract**: Court, Judge, Year, Outcome from each case
4. **Format**: Save to CSV with same structure as synthetic data

**Target**: Collect at least **500-1000 real cases** for good accuracy.

---

### **Option C: Use Existing Datasets**

Check these sources:
- **eCourts India**: https://ecourts.gov.in/ecourts_home/
- **Supreme Court of India**: https://main.sci.gov.in/
- **Legal databases**: Manupatra, SCC Online (paid)

---

## 🔧 Step 2: Feature Engineering

Add more predictive features to improve accuracy:

### **New Features to Add**

```python
# enhanced_feature_extractor.py

class EnhancedFeatureExtractor(FeatureExtractor):
    """
    Extract additional features for better predictions
    """
    
    def extract_enhanced_features(self, text):
        """
        Extract 15+ features instead of 6
        """
        base_features = self.extract_features(text)
        
        # Add new features
        enhanced = {
            **base_features,
            
            # Temporal features
            'case_age': self.calculate_case_age(base_features['year']),
            'is_recent': 1 if base_features['year'] >= 2020 else 0,
            
            # Court hierarchy
            'court_level': self.get_court_level(base_features['court']),
            'is_supreme_court': 1 if 'Supreme' in base_features['court'] else 0,
            
            # Case characteristics
            'case_complexity_score': self.calculate_complexity(text),
            'num_parties': self.count_parties(text),
            'has_precedent': self.check_precedent_citation(text),
            
            # Legal domain
            'is_ip_case': 1 if base_features['case_type'] in ['Trademark', 'Patent', 'Copyright'] else 0,
            'is_contract_case': 1 if 'Contract' in base_features['case_type'] else 0,
            
            # Text features
            'text_length': len(text),
            'num_legal_terms': self.count_legal_terms(text),
            
            # Judge features
            'judge_experience': self.estimate_judge_experience(base_features['judge']),
        }
        
        return enhanced
    
    def get_court_level(self, court):
        """
        Assign hierarchy level to court
        """
        if 'Supreme' in court:
            return 3
        elif 'High' in court:
            return 2
        else:
            return 1
    
    def calculate_complexity(self, text):
        """
        Calculate case complexity based on text analysis
        """
        # Count legal terminology
        legal_terms = [
            'precedent', 'jurisdiction', 'constitutional',
            'statutory', 'interpretation', 'doctrine',
            'ratio decidendi', 'obiter dicta', 'ultra vires'
        ]
        
        term_count = sum(1 for term in legal_terms if term.lower() in text.lower())
        
        # Word count factor
        word_count = len(text.split())
        
        # Complexity score (1-10)
        complexity = min(10, (term_count * 2) + (word_count // 500))
        
        return complexity
    
    def count_parties(self, text):
        """
        Count number of parties involved
        """
        # Look for "vs", "v.", "versus"
        vs_count = text.lower().count(' vs ') + text.lower().count(' v. ')
        return min(vs_count + 1, 10)  # Cap at 10
    
    def check_precedent_citation(self, text):
        """
        Check if case cites precedents
        """
        citation_patterns = ['AIR', 'SCC', 'SCR', 'cited in', 'relying on']
        return 1 if any(pattern in text for pattern in citation_patterns) else 0
    
    def count_legal_terms(self, text):
        """
        Count legal terminology density
        """
        legal_terms = [
            'plaintiff', 'defendant', 'petitioner', 'respondent',
            'appellant', 'court', 'judgment', 'order', 'decree',
            'suit', 'appeal', 'petition', 'application'
        ]
        
        count = sum(text.lower().count(term) for term in legal_terms)
        return min(count, 100)  # Cap at 100
    
    def estimate_judge_experience(self, judge_name):
        """
        Estimate judge experience (simplified)
        """
        # In real implementation, maintain a database of judges
        # For now, use a simple heuristic
        return 5  # Default medium experience
```

---

## 🤖 Step 3: Model Optimization

### **A. Try Different Algorithms**

```python
# advanced_models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        ),
        
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"  Train Accuracy: {train_acc:.2%}")
        print(f"  Test Accuracy: {test_acc:.2%}")
    
    return results
```

---

### **B. Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_random_forest(X_train, y_train):
    """
    Hyperparameter tuning for Random Forest
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.2%}")
    
    return grid_search.best_estimator_
```

---

### **C. Ensemble Methods**

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

def create_ensemble(X_train, y_train):
    """
    Create ensemble of best models
    """
    # Base models
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
    lgbm = LGBMClassifier(n_estimators=200, max_depth=10, random_state=42)
    
    # Voting ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm)
        ],
        voting='soft'  # Use probability voting
    )
    
    voting_clf.fit(X_train, y_train)
    
    return voting_clf
```

---

## 📈 Step 4: Handle Class Imbalance

Your data has 63% plaintiff wins - this imbalance hurts accuracy.

### **Techniques:**

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def balance_dataset(X_train, y_train):
    """
    Balance the dataset using SMOTE
    """
    # SMOTE: Synthetic Minority Over-sampling
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original distribution: {pd.Series(y_train).value_counts()}")
    print(f"Balanced distribution: {pd.Series(y_balanced).value_counts()}")
    
    return X_balanced, y_balanced
```

---

## 🎯 Step 5: Cross-Validation

Use proper cross-validation to get accurate performance estimates:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate_with_cv(model, X, y):
    """
    Evaluate model with stratified k-fold cross-validation
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Std deviation: {scores.std():.2%}")
    
    return scores
```

---

## 📊 Complete Training Pipeline

Here's the complete pipeline to achieve 80-90% accuracy:

```python
# train_improved_model.py

def train_improved_model():
    """
    Complete training pipeline with all improvements
    """
    print("=" * 80)
    print("IMPROVED MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load REAL data (not synthetic)
    print("\n1. Loading real case data...")
    df = pd.read_csv('1-Rag/data/real_cases.csv')  # Replace with real data
    print(f"   Loaded {len(df)} real cases")
    
    # Step 2: Extract enhanced features
    print("\n2. Extracting enhanced features...")
    extractor = EnhancedFeatureExtractor()
    
    features_list = []
    for text in df['description']:
        features = extractor.extract_enhanced_features(text)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Step 3: Prepare data
    print("\n3. Preparing data...")
    predictor = OutcomePredictor(model_dir='1-Rag/models')
    X = predictor.prepare_features(features_df, fit=True)
    y = predictor.outcome_encoder.fit_transform(df['outcome'])
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 5: Balance dataset
    print("\n4. Balancing dataset...")
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Step 6: Train multiple models
    print("\n5. Training multiple models...")
    results = train_multiple_models(
        X_train_balanced, y_train_balanced,
        X_test, y_test
    )
    
    # Step 7: Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['test_accuracy']
    
    print(f"\n6. Best model: {best_model_name}")
    print(f"   Test accuracy: {best_accuracy:.2%}")
    
    # Step 8: Cross-validation
    print("\n7. Cross-validation...")
    cv_scores = evaluate_with_cv(best_model, X, y)
    
    # Step 9: Save model
    predictor.model = best_model
    predictor.save_model()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    return best_model, best_accuracy

if __name__ == "__main__":
    model, accuracy = train_improved_model()
```

---

## 🚀 Quick Start Guide

### **Immediate Actions (This Week)**

1. **Collect 100 real cases manually**:
   - Visit Indian Kanoon
   - Search for commercial court cases
   - Extract court, judge, case type, year, outcome
   - Save to `real_cases.csv`

2. **Retrain with real data**:
   ```bash
   python3 train_improved_model.py
   ```

3. **Expected improvement**: 54% → 65-70%

---

### **Medium-term (This Month)**

1. **Scrape 500-1000 cases** using the scraping script
2. **Add enhanced features** (15+ features instead of 6)
3. **Try XGBoost/LightGBM** instead of Random Forest
4. **Use SMOTE** for class balancing

**Expected accuracy**: 70-80%

---

### **Long-term (Next 3 Months)**

1. **Collect 2000+ real cases**
2. **Implement all advanced features**
3. **Use ensemble methods**
4. **Fine-tune hyperparameters**

**Expected accuracy**: 80-90%

---

## 📊 Expected Accuracy Progression

| Stage | Data Size | Features | Model | Accuracy |
|-------|-----------|----------|-------|----------|
| **Current** | 250 synthetic | 6 basic | Random Forest | 54% |
| **Phase 1** | 100 real | 6 basic | Random Forest | 65-70% |
| **Phase 2** | 500 real | 15 enhanced | XGBoost | 72-78% |
| **Phase 3** | 1000 real | 15 enhanced | Ensemble | 78-85% |
| **Phase 4** | 2000+ real | 20+ enhanced | Tuned Ensemble | 85-90% |

---

## 🎯 Key Takeaways

1. **Real data is crucial**: Synthetic data limits you to ~60% max
2. **More features help**: 15+ features vs 6 can add 10-15% accuracy
3. **Better algorithms matter**: XGBoost/LightGBM often beat Random Forest
4. **Balance your data**: SMOTE can add 5-10% accuracy
5. **Ensemble methods**: Combining models adds 3-5% accuracy

**Bottom line**: To reach 80-90%, you MUST collect real case data. Everything else is optimization on top of that foundation.
