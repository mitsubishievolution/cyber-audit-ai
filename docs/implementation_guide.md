# Implementation Guide - Cybersecurity Audit AI

## Purpose

This guide walks you through implementing and testing the AI component of the cybersecurity audit tool. It's designed for university students with basic programming knowledge - every step is explained clearly.

**What You'll Build**: An AI system that analyzes website vulnerabilities and predicts risk scores from 0-10.

## Prerequisites

### Software Requirements
- **Python 3.8+** (check with `python --version`)
- **pip** package manager
- **Basic command line knowledge**

### Hardware Requirements
- Standard laptop or desktop
- 4GB RAM minimum
- No GPU required

### Knowledge Prerequisites
- Basic Python syntax
- Understanding of JSON data
- Basic command line usage
- No machine learning experience needed

## Step-by-Step Implementation

### Step 1: Project Setup

1. **Navigate to the ai_module directory:**
   ```bash
   cd ai_module
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **What this installs:**
   - `flask`: Web server for the AI API
   - `scikit-learn`: Machine learning algorithms
   - `pandas`: Data manipulation
   - `numpy`: Numerical computations
   - `matplotlib`: Charts and graphs
   - `joblib`: Model saving/loading

3. **Verify installation:**
   ```bash
   python -c "import flask, sklearn, pandas, numpy; print('All dependencies installed')"
   ```

### Step 2: Understanding the Data

**Dataset Location**: `../data/sample_vulnerabilities.json`

**Data Format:**
```json
{
  "cve": "CVE-2023-1234",
  "description": "Description of the vulnerability",
  "cvss": 7.5,
  "type": "SQL Injection"
}
```

**What Each Field Means:**
- **cve**: Unique identifier (like a license plate for vulnerabilities)
- **description**: Human-readable explanation
- **cvss**: Base severity score (0-10 scale)
- **type**: Category (SQL Injection, XSS, etc.)

**View the data:**
```bash
# Count total vulnerabilities
python -c "import json; data=json.load(open('../data/sample_vulnerabilities.json')); print(f'Total: {len(data)} vulnerabilities')"
```

### Step 3: Feature Engineering

**File**: `feature_engineering.py`

**What it does**: Converts text and categories into numbers that the AI can understand.

**Test feature engineering:**
```bash
python feature_engineering.py
```

**Expected Output:**
```
Successfully loaded 20 vulnerability examples
Converting text descriptions to TF-IDF vectors...
Encoding vulnerability types...
Feature engineering complete:
  - Text features: 100
  - Type features: 10
  - CVSS features: 1
  - Total features: 111

Feature extraction successful!
Dataset shape: (20, 111)
Target shape: (20,)
```

**What the numbers mean:**
- **20 examples**: Our training dataset size
- **111 features**: The AI considers 111 different factors
- **Text features**: Words from descriptions (100 most important)
- **Type features**: Vulnerability categories (10 types)
- **CVSS features**: The base severity score

### Step 4: Training the AI Model

**File**: `model_trainer.py`

**What it does**: Trains the Random Forest algorithm on vulnerability data.

**Run training:**
```bash
python model_trainer.py
```

**Expected Output:**
```
Starting Complete AI Model Training Pipeline
==================================================

Step 1: Loading and processing vulnerability data...
Successfully loaded 20 vulnerability examples
Converting text descriptions to TF-IDF vectors...
Encoding vulnerability types...
Feature engineering complete:
  - Text features: 100
  - Type features: 10
  - CVSS features: 1
  - Total features: 111

Step 2: Training RandomForest risk prediction model...
Starting AI model training...
   - Training data shape: (20, 111)
   - Test set: 4 samples
   - Training RandomForestRegressor...
   - Cross-validation Results:
   - CV RMSE scores: [1.2, 0.8, 1.5, 0.9, 1.1]
   - Average CV RMSE: 1.1

=== RandomForest Risk Model Performance Evaluation ===
Sample Size: 4 vulnerabilities
Mean Squared Error: 1.45
Root Mean Squared Error: 1.20
Mean Absolute Error: 1.05
R² Score: 0.72
Predictions within 1.0 points: 75.00%
Predictions within 2.0 points: 100.00%

Model training completed successfully!

Saving trained model to model/trained_model.pkl...
Model saved to model/trained_model.pkl

Analyzing feature importance...
Top 5 most important features:
  - cvss_score: 0.3500
  - text_15: 0.1200
  - type_SQL Injection: 0.0980
  - text_42: 0.0870
  - type_Cross-Site Scripting (XSS): 0.0760

AI Model Training Complete!
   - Model saved: model/trained_model.pkl
   - Training accuracy: 75.00% within 1 point
   - Ready for predictions!
```

**What the results mean:**
- **75% within 1 point**: The AI is correct within 1 point of the actual risk score 75% of the time
- **R² Score: 0.72**: Good fit (closer to 1.0 is better)
- **MAE: 1.05**: Average error of 1.05 points on a 0-10 scale

### Step 5: Testing the API

**File**: `app.py`

**What it does**: Provides a web API that other parts of the system can call.

**Start the API server:**
```bash
python app.py
```

**Expected Output:**
```
Starting Cybersecurity Audit AI Flask API...
Model loaded from model/trained_model.pkl
AI Model loaded successfully!
Flask API is ready to serve predictions.
Flask server on http://127.0.0.1:5000
API Documentation:
   GET  /health  - Check API status
   POST /predict - Make risk predictions
```

**Test the health endpoint:**
```bash
curl http://127.0.0.1:5000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "message": "Cybersecurity Audit AI API is running",
  "model_status": "loaded",
  "endpoints": {
    "/predict": "POST - Make risk predictions",
    "/health": "GET - Check API status"
  }
}
```

**Test a prediction:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "SQL injection vulnerability in login form allows attackers to bypass authentication",
    "type": "SQL Injection",
    "cvss": 7.5
  }'
```

**Expected Response:**
```json
{
  "risk_score": 7.6,
  "confidence": "high",
  "explanation": "The AI model predicts a risk score of 7.6 with high confidence, closely matching the input CVSS score of 7.5. This assessment considers the sql injection vulnerability type and description text.",
  "input_summary": {
    "type": "SQL Injection",
    "cvss_score": 7.5,
    "description_length": 85
  }
}
```

## Customization Guide

### Changing Model Parameters

**File**: `model_trainer.py`

**Location**: Around line 35 in the `CybersecurityRiskModel.__init__` method

```python
def __init__(self, n_estimators=100, max_depth=10, random_state=42):
    # Change these values:
    # n_estimators: Number of trees (more = better but slower)
    # max_depth: How deep each tree can grow (deeper = more complex)
```

**Try different settings:**
```python
# More trees (potentially more accurate but slower)
model = CybersecurityRiskModel(n_estimators=200, max_depth=15)

# Fewer trees (faster but potentially less accurate)
model = CybersecurityRiskModel(n_estimators=50, max_depth=5)
```

### Adding More Training Data

**File**: `../data/sample_vulnerabilities.json`

**Add new vulnerabilities:**
```json
{
  "cve": "CVE-2023-9999",
  "description": "Your new vulnerability description",
  "cvss": 8.0,
  "type": "New Vulnerability Type"
}
```

**Then retrain:**
```bash
python model_trainer.py
```

### Changing Text Processing

**File**: `feature_engineering.py`

**Location**: Around line 87 in the `extract_features` method

```python
self.vectorizer = TfidfVectorizer(
    max_features=100,  # Change: more features = more detailed but slower
    stop_words='english',  # Change: None to include common words
    ngram_range=(1, 2)  # Change: (1, 1) for single words only
)
```

## Testing Checklist

### Basic Functionality Tests

- [ ] Dependencies install without errors
- [ ] Data loads correctly (20 vulnerabilities)
- [ ] Feature engineering produces expected shapes
- [ ] Model training completes successfully
- [ ] Model saves to file
- [ ] API server starts
- [ ] Health endpoint returns "healthy"
- [ ] Prediction endpoint accepts JSON and returns results

### Accuracy Tests

- [ ] Test with known high-risk vulnerability (expect score > 7)
- [ ] Test with known low-risk vulnerability (expect score < 5)
- [ ] Verify predictions are between 0-10
- [ ] Check confidence levels are assigned correctly

### Error Handling Tests

- [ ] Test with missing required fields
- [ ] Test with invalid CVSS score (> 10 or < 0)
- [ ] Test with empty description
- [ ] Verify appropriate error messages returned

## Troubleshooting

### "Module not found" errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install scikit-learn pandas numpy
```

### "Model not found" error
```bash
# Train the model first
python model_trainer.py

# Check if model file exists
ls -la model/trained_model.pkl
```

### API server won't start
```bash
# Kill any existing processes on port 5000
lsof -ti:5000 | xargs kill -9

# Try different port
python -c "from app import app; app.run(port=5001)"
```

### Poor model accuracy
- Add more training data to `sample_vulnerabilities.json`
- Adjust model parameters (more trees, deeper trees)
- Check that vulnerability types are consistent

### Memory errors
- Reduce `n_estimators` in model training
- Decrease `max_features` in text processing
- Use smaller dataset for testing

## Understanding Results

### Risk Score Interpretation
- **0-3**: Low risk - minor issues
- **4-6**: Medium risk - should be addressed
- **7-10**: High risk - critical security issues

### Confidence Levels
- **High**: Prediction within 1 point of input CVSS
- **Medium**: Prediction within 2 points of input CVSS
- **Low**: Prediction differs by more than 2 points

### Performance Metrics
- **MAE < 1.0**: Excellent performance
- **MAE 1.0-2.0**: Good performance (our current level)
- **MAE > 2.0**: Needs improvement

## Integration with Other Components

### Connecting to Spring Boot Backend

**The AI API expects POST requests to `/predict` with:**
```json
{
  "description": "string",
  "type": "string",
  "cvss": number
}
```

**It returns:**
```json
{
  "risk_score": number,
  "confidence": "string",
  "explanation": "string"
}
```

### Connecting to React Frontend

**Frontend can call the API directly or through the backend:**
```javascript
// Direct API call from React
const response = await fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    description: vulnerability.description,
    type: vulnerability.type,
    cvss: vulnerability.cvss
  })
});
```

## Academic Notes

### Research Questions Answered
- Can machine learning predict vulnerability risk scores?
- How well does Random Forest perform on small datasets?
- Is the model interpretable for non-technical users?

### Limitations Documented
- Small training dataset (20 examples)
- Limited vulnerability types covered
- Simple algorithm choice for educational purposes

### Future Research Directions
- Compare with other ML algorithms (SVM, Neural Networks)
- Test on larger real-world datasets
- Add temporal features (vulnerability age, exploit availability)

---

## Need Help?

**Common Issues:**
1. Check that all dependencies are installed
2. Verify the model is trained before starting the API
3. Test with the exact JSON format shown above

**Debug Mode:**
Run with `python -c "import [module_name]; print('Import successful')"` to test each module.

**Reset Everything:**
```bash
# Remove trained model to start fresh
rm -f model/trained_model.pkl

# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

**Note**: This is a university project - it's okay if it's not perfect. The goal is learning and demonstration!