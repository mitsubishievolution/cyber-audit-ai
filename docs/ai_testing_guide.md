# AI System Testing Guide - Week 2

## Simple Testing Tutorial for Week 2 AI System

This guide provides step-by-step instructions to test your Week 2 AI cybersecurity audit system and verify everything is working properly.

---

## Preparation: Open VS Code

1. **Open your project** in VS Code
2. **Open terminal** in VS Code (`Ctrl + \` backtick)
3. **Navigate to AI folder**:
   ```bash
   cd ai_module
   ```

---

## Step 1: Train the AI Model

```bash
python3 model_trainer.py
```

**What you should see:**
- Training progress messages
- "Model training completed successfully!"
- "Model saved to model/trained_model.pkl"

**Time**: ~10-30 seconds

---

## Step 2: Start the API Server

```bash
python3 app.py
```

**What you should see:**
- "Model loaded successfully!"
- "Flask server on http://127.0.0.1:5000"

**Time**: ~3-5 seconds

---

## Step 3: Test the Health Check

**Open a new terminal** in VS Code (`Ctrl + \` backtick again)

```bash
curl http://127.0.0.1:5000/health
```

**Expected result:**
```json
{
  "status": "healthy",
  "model_status": "loaded"
}
```

**Time**: ~1 second

---

## Step 4: Test a Real Prediction

In the same terminal:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "SQL injection vulnerability in login form",
    "type": "SQL Injection",
    "cvss": 7.5
  }'
```

**Expected result:**
```json
{
  "risk_score": 7.4,
  "confidence": "high"
}
```

**Time**: ~1 second

---

## Step 5: Test Different Vulnerability Types

**Test XSS (should be lower risk):**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "XSS allows script execution",
    "type": "Cross-Site Scripting (XSS)",
    "cvss": 6.1
  }'
```

**Expected result: Risk score around 5-6**

---

## Step 6: Run Automated Test

```bash
python3 test_ai_pipeline.py
```

**Expected result:**
```
ALL TESTS PASSED!
```

**Time**: ~5-10 seconds

---

## How to Know It's Working

### Success Signs:
- Model trains without errors
- API server starts successfully
- Health endpoint returns "healthy"
- Prediction returns numbers 0-10
- Different vulnerability types give different scores
- Automated test suite passes

### What Each Test Proves:

1. **Training Test** → Your AI can learn from data
2. **API Test** → Your AI can serve predictions
3. **Health Test** → Your system reports its status correctly
4. **Prediction Test** → Your AI gives sensible risk scores
5. **Variety Test** → Your AI understands different vulnerability types
6. **Automated Test** → All components work together

---

## Quick Troubleshooting

### If Training Fails:
```bash
# Check Python version
python3 --version
# Install dependencies
pip3 install -r requirements.txt
```

### If API Won't Start:
```bash
# Kill any existing processes
pkill -f python3
# Try different port
FLASK_RUN_PORT=5001 python3 app.py
```

### If Predictions Fail:
```bash
# Check if model file exists
ls -la model/trained_model.pkl
# Restart everything
python3 model_trainer.py && python3 app.py
```

---

## Final Success Checklist

After following all steps, you should have:

- [ ] **Model trained** (no errors during training)
- [ ] **Server running** (API starts on port 5000)
- [ ] **Health check works** (returns healthy status)
- [ ] **SQL injection test** (risk score ~7-8)
- [ ] **XSS test** (risk score ~5-6)
- [ ] **All tests pass** (automated test succeeds)

**If all boxes checked**: Your Week 2 AI system is working perfectly!

---

## Next Steps

Once everything works:
1. **Stop the server** (`Ctrl + C` in terminal)
2. **Read the documentation** in `docs/` folder
3. **Prepare for Week 3** (Spring Boot backend integration)

---

## Need Help?

**Common Issues:**
1. Check that all dependencies are installed
2. Verify the model is trained before starting the API
3. Test with the exact JSON format shown above
4. Make sure no other processes are using port 5000

**Debug Mode:**
Run with `python -c "import [module_name]; print('Import successful')"` to test each module.

---

**Note**: This is a university project demonstrating AI application in cybersecurity. The system is designed for educational purposes and learning machine learning concepts.