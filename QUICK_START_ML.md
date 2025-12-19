# Quick Start: Improving ML Model to 80-90% Accuracy

## 🎯 Goal
Improve outcome prediction accuracy from **54%** to **80-90%**

## 📋 3-Step Action Plan

### **Step 1: Collect Real Data (Most Important!)**

**Option A: Use the Scraper (Recommended)**
```bash
# Add your Indian Kanoon API token to .env
echo "INDIAN_KANOON_API_TOKEN=your_token_here" >> .env

# Run the scraper to collect 100 cases
cd /Users/mayankjaideep/Desktop/ai-driven-research-engine-main
python3 1-Rag/scrape_indian_kanoon.py
```

**Option B: Manual Collection**
1. Visit https://indiankanoon.org
2. Search for "commercial court" cases
3. Extract: court, judge, case type, year, outcome
4. Save to `1-Rag/data/real_cases.csv`

**Target**: Start with 100 cases, then scale to 500-1000

---

### **Step 2: Use Enhanced Features**

The enhanced feature extractor adds **20+ features** instead of 6:

```bash
# Test the enhanced extractor
python3 1-Rag/enhanced_feature_extractor.py
```

**New features include:**
- Temporal: case age, decade, is_recent
- Court hierarchy: level, is_supreme_court, is_high_court  
- Case characteristics: complexity score, num_parties, has_precedent
- Legal domain: is_ip_case, is_contract_case, is_corporate_case

---

### **Step 3: Retrain with Better Model**

Once you have real data:

```bash
# Install additional ML libraries
pip install xgboost lightgbm imbalanced-learn

# Retrain with enhanced features and real data
# (You'll need to modify outcome_predictor.py to use enhanced features)
```

---

## 📊 Expected Accuracy Progression

| Phase | Data | Features | Model | Accuracy |
|-------|------|----------|-------|----------|
| **Current** | 250 synthetic | 6 | Random Forest | 54% |
| **Phase 1** | 100 real | 6 | Random Forest | 65-70% |
| **Phase 2** | 500 real | 20+ | XGBoost | 75-80% |
| **Phase 3** | 1000+ real | 20+ | Ensemble | 80-90% |

---

## 📁 New Files Created

1. **[ML_IMPROVEMENT_GUIDE.md](file:///Users/mayankjaideep/Desktop/ai-driven-research-engine-main/ML_IMPROVEMENT_GUIDE.md)** - Complete guide with code examples
2. **[enhanced_feature_extractor.py](file:///Users/mayankjaideep/Desktop/ai-driven-research-engine-main/1-Rag/enhanced_feature_extractor.py)** - 20+ feature extraction
3. **[scrape_indian_kanoon.py](file:///Users/mayankjaideep/Desktop/ai-driven-research-engine-main/1-Rag/scrape_indian_kanoon.py)** - Real data collection script

---

## 🚀 Quick Commands

```bash
# 1. Get API token from Indian Kanoon
# Visit: https://indiankanoon.org/api/

# 2. Add to .env file
echo "INDIAN_KANOON_API_TOKEN=your_token" >> .env

# 3. Scrape 100 real cases
python3 1-Rag/scrape_indian_kanoon.py

# 4. Test enhanced features
python3 1-Rag/enhanced_feature_extractor.py

# 5. Install ML libraries
pip install xgboost lightgbm imbalanced-learn
```

---

## 💡 Key Insights

1. **Real data is #1 priority**: Synthetic data caps you at ~60% accuracy
2. **More features = better predictions**: 20+ features vs 6 can add 10-15%
3. **XGBoost > Random Forest**: Usually 5-10% better for this task
4. **Balance your classes**: SMOTE can add 5-10% accuracy
5. **Ensemble methods**: Combining models adds 3-5%

---

## ⚠️ Important Notes

- **Start small**: Collect 100 cases first, then scale up
- **Quality > Quantity**: Ensure outcome labels are accurate
- **Iterate**: Train → Evaluate → Collect more data → Repeat
- **Be patient**: Getting to 80-90% takes time and real data

---

## 📞 Next Steps

1. **This week**: Get Indian Kanoon API token, scrape 100 cases
2. **Next week**: Retrain with real data, expect 65-70% accuracy
3. **This month**: Scale to 500 cases, add enhanced features, reach 75-80%
4. **Long-term**: Collect 1000+ cases, use ensemble methods, achieve 80-90%

**Read the full guide**: [ML_IMPROVEMENT_GUIDE.md](file:///Users/mayankjaideep/Desktop/ai-driven-research-engine-main/ML_IMPROVEMENT_GUIDE.md)
