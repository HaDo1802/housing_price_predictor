# 🎓 Complete MLOps Guide for Beginners

## 📌 ANSWER TO YOUR MAIN QUESTIONS

### Question 1: "Where should we split train/test?"

**ANSWER: Split AFTER cleaning but BEFORE any preprocessing that learns from data.**

```python
# CORRECT ORDER:
1. Load raw data
2. Clean data (remove duplicates, fix types) ← No learning
3. SPLIT into train/val/test               ← SPLIT HERE!
4. Preprocess (fit on train only)          ← Learning happens
5. Train models
```

**Why?**
- Operations that "learn" from data (mean, median, scaling, encoding) must ONLY see training data
- If preprocessing happens before split, test data "leaks" into training
- This causes overly optimistic metrics that don't hold in production

### Question 2: "Why make everything modular?"

**ANSWER: Modularity enables reusability, testability, and prevents bugs.**

**Example: Non-modular (Bad)**
```python
# train.py - 500 lines of spaghetti code
df = pd.read_csv('data.csv')
df.fillna(df.mean(), inplace=True)  # Line 50
# ... 400 more lines ...
scaler.fit(X_train)  # Line 300
# ... 200 more lines ...

# predict.py - Copy-paste nightmare
df = pd.read_csv('new_data.csv')
df.fillna(df.mean(), inplace=True)  # Did we use mean or median?
# ... bug! Different preprocessing ...
```

**Example: Modular (Good)**
```python
# Training
preprocessor = DataPreprocessor()
X_train = preprocessor.fit_transform(X_train)
preprocessor.save('preprocessor.pkl')

# Production (months later)
preprocessor = DataPreprocessor.load('preprocessor.pkl')
X_new = preprocessor.transform(X_new)  # SAME preprocessing guaranteed!
```

### Question 3: "What exactly goes into preprocessing?"

**ANSWER: Preprocessing includes ALL operations that transform raw data into model-ready format.**

```
INPUT: Raw messy data
   ↓
1. CLEANING (no learning)
   - Remove duplicates
   - Fix data types
   
2. MISSING VALUES (learns statistics!)
   - Impute with mean/median/mode
   
3. OUTLIERS (learns statistics!)
   - Detect and remove based on IQR/Z-score
   
4. FEATURE ENGINEERING (may learn!)
   - Create new features
   - Transform features (log, sqrt)
   
5. SCALING (learns statistics!)
   - StandardScaler: learns mean & std
   - MinMaxScaler: learns min & max
   
6. ENCODING (learns categories!)
   - One-hot: learns unique categories
   - Label: learns label mapping
   
OUTPUT: Clean, scaled, encoded data ready for model
```

## 🏗️ THE PRODUCTION-GRADE STRUCTURE

### What Makes Code "Production-Grade"?

| Aspect | Development Code | Production Code |
|--------|-----------------|-----------------|
| **Structure** | Single script | Modular components |
| **Config** | Hardcoded values | YAML config files |
| **Splitting** | Anywhere | Before preprocessing |
| **Artifacts** | Don't save | Save everything |
| **Testing** | Manual | Automated tests |
| **Logging** | Print statements | Structured logging |
| **Errors** | Crashes | Graceful handling |
| **Documentation** | Maybe README | Complete docs |
| **Versioning** | No tracking | Model + data versions |
| **Deployment** | Manual | CI/CD automated |

### The 7 Core Modules

```
1. CONFIG MANAGER
   Purpose: Load and manage all settings
   Why: Change hyperparameters without code changes
   
2. DATA SPLITTER
   Purpose: Split data correctly
   Why: Prevent data leakage, ensure reproducibility
   
3. PREPROCESSOR
   Purpose: Transform data consistently
   Why: Same preprocessing in training and production
   
4. MODEL TRAINER
   Purpose: Train and evaluate models
   Why: Compare multiple models, select best
   
5. TRAINING PIPELINE
   Purpose: Orchestrate complete workflow
   Why: Reproducible, automated training
   
6. INFERENCE PIPELINE
   Purpose: Make predictions in production
   Why: Load saved artifacts, handle new data
   
7. UTILITIES
   Purpose: Logging, metrics, helpers
   Why: Reusable functions across modules
```

## 🎯 WHY CLASSES vs FUNCTIONS?

### When to Use Classes

**Use classes when:**
1. **State needs to persist** (fitted scaler, encoder)
2. **Multiple related operations** (fit, transform, save, load)
3. **Configuration is complex** (many parameters)

**Example:**
```python
class DataPreprocessor:
    def __init__(self):
        self.scaler = None       # Persists between calls
        self.encoder = None      # Persists between calls
    
    def fit_transform(self, X):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)
    
    def transform(self, X):
        return self.scaler.transform(X)  # Uses saved scaler!
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
```

### When to Use Functions

**Use functions when:**
1. **Stateless operations** (no memory needed)
2. **Simple transformations** (one input → one output)
3. **Utility operations** (logging, validation)

**Example:**
```python
def remove_duplicates(df):
    """Simple, stateless operation"""
    return df.drop_duplicates()

def validate_data_types(df, expected_types):
    """Utility function, no state"""
    for col, dtype in expected_types.items():
        if df[col].dtype != dtype:
            raise ValueError(f"Wrong type for {col}")
```

## 🔄 THE COMPLETE WORKFLOW EXPLAINED

### Training Time

```python
# STEP 1: Initialize with configuration
config = ConfigManager("config/config.yaml").get_config()
pipeline = TrainingPipeline(config)

# STEP 2: Load raw data
df = pipeline.load_data()
# df shape: (10000, 50)

# STEP 3: Clean data (NO learning)
df_clean = pipeline.clean_data()
# df_clean shape: (9800, 50)  ← Removed 200 duplicates

# STEP 4: SPLIT DATA (CRITICAL!)
# This is THE answer to your question!
X_train, X_test, X_val, y_train, y_test, y_val = pipeline.split_data()
# Train: 6860 (70%)
# Val:   980 (10%)
# Test: 1960 (20%)

# STEP 5: Preprocess (FIT on train ONLY)
preprocessor = ProductionPreprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
# Preprocessor learns:
#   - Mean of each feature (for imputation)
#   - Std of each feature (for scaling)
#   - Unique categories (for encoding)
# ALL from training data ONLY!

X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)
# Val and test are transformed using TRAIN statistics

# STEP 6: Train models
trainer = ModelTrainer()
trainer.train_all(X_train_scaled, y_train)
# Trains: Linear, Ridge, Lasso, RF, GBM, SVR

# STEP 7: Evaluate
results = trainer.evaluate_all(X_test_scaled, y_test)
# Linear Regression: R² = 0.85
# Random Forest:     R² = 0.91  ← Best!
# Gradient Boosting: R² = 0.89

# STEP 8: Select best
best_model = trainer.select_best_model(metric='r2')
# Selected: Random Forest

# STEP 9: Save everything
pipeline.save_artifacts('models/production')
# Saved:
#   - model.pkl (Random Forest)
#   - preprocessor.pkl (fitted transformers)
#   - config.yaml (exact configuration)
#   - metadata.json (metrics, feature names)
```

### Production Time (Weeks Later)

```python
# STEP 1: Load saved pipeline
pipeline = InferencePipeline('models/production')
# Loads:
#   - Trained Random Forest model
#   - Fitted preprocessor (with training statistics)
#   - Metadata

# STEP 2: New data arrives
new_customer = {
    'age': 35,
    'income': 50000,
    'credit_score': 720,
    'employment_years': 5,
    'city': 'New York'
}

# STEP 3: Make prediction
prediction = pipeline.predict_single(new_customer)
# Behind the scenes:
#   1. Convert to DataFrame
#   2. Preprocessor.transform(new_data)
#      - Uses SAME mean/std from training
#      - Uses SAME category encoding
#   3. Model.predict(preprocessed_data)
#   4. Return prediction

print(f"Predicted house price: ${prediction:,.2f}")
# Output: Predicted house price: $325,000.00
```

## 🚨 COMMON MISTAKES TO AVOID

### Mistake 1: Data Leakage

```python
# ❌ WRONG
df = handle_missing(df, method='mean')  # Uses ALL data!
X_train, X_test = train_test_split(df)
# Test data influenced the mean calculation!

# ✅ CORRECT
X_train, X_test = train_test_split(df)
X_train = handle_missing(X_train, method='mean')  # Train only
X_test = handle_missing(X_test, method='mean')    # Test only
```

### Mistake 2: Fitting on Test Data

```python
# ❌ WRONG
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # FIT AGAIN! Wrong!

# ✅ CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)         # Transform test
```

### Mistake 3: Not Saving Preprocessor

```python
# ❌ WRONG - Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
pickle.dump(model, open('model.pkl', 'wb'))
# Forgot to save scaler!

# ❌ WRONG - Production
model = pickle.load(open('model.pkl', 'rb'))
scaler = StandardScaler()  # NEW scaler! Different parameters!
X_new_scaled = scaler.fit_transform(X_new)  # WRONG!
prediction = model.predict(X_new_scaled)  # Garbage prediction!

# ✅ CORRECT - Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))  # Save scaler!

# ✅ CORRECT - Production
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load SAME scaler
X_new_scaled = scaler.transform(X_new)  # Use saved parameters
prediction = model.predict(X_new_scaled)  # Correct!
```

### Mistake 4: Inconsistent Preprocessing

```python
# ❌ WRONG
# Training: filled missing with mean
X_train = X_train.fillna(X_train.mean())

# Production: filled missing with median (DIFFERENT!)
X_new = X_new.fillna(X_new.median())
# Model expects mean-filled, gets median-filled!

# ✅ CORRECT
# Training
preprocessor.set_imputation_method('mean')
X_train = preprocessor.fit_transform(X_train)
preprocessor.save('preprocessor.pkl')

# Production
preprocessor = load('preprocessor.pkl')
X_new = preprocessor.transform(X_new)  # Uses 'mean' method!
```

## 📚 LEARNING PATH

### Week 1: Understand the Basics
- What is data leakage?
- When to split data?
- Fit vs transform

### Week 2: Build Modular Components
- Create DataSplitter class
- Create Preprocessor class
- Save and load artifacts

### Week 3: Build Pipelines
- Training pipeline
- Inference pipeline
- Configuration management

### Week 4: Add Production Features
- Error handling
- Logging
- Input validation
- Testing

### Week 5: Automation & CI/CD
- Makefile
- GitHub Actions
- Docker
- Monitoring

## 🎯 KEY TAKEAWAYS

1. **Split data BEFORE preprocessing** that learns from data
2. **Fit ONLY on training data**, transform on all sets
3. **Save ALL artifacts** (model, preprocessor, config)
4. **Use classes** when state needs to persist
5. **Make everything modular** for reusability
6. **Configuration-driven** for flexibility
7. **Test everything** to catch bugs early
8. **Document thoroughly** for future you and team

## 📖 NEXT STEPS

Now that you understand the structure:

1. **Practice**: Build a simple project following this structure
2. **Experiment**: Try different models, preprocessing methods
3. **Test**: Write tests for each component
4. **Deploy**: Put a model into production
5. **Monitor**: Track performance over time
6. **Iterate**: Improve based on feedback

## 🤝 REMEMBER

> "Perfect is the enemy of good. Start simple, make it work, then make it better."

Start with basic train/test split and preprocessing. Add complexity as you understand each piece. The goal is working, maintainable code, not perfect code.

Good luck! 🚀
