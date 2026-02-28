#!/usr/bin/env python3
"""
test_ai_pipeline.py

A simple test script to verify the AI pipeline works end-to-end.
This tests the complete flow: data loading, feature engineering, model prediction.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import VulnerabilityFeatureEngineer
from model_trainer import CybersecurityRiskModel

def test_data_loading():
    """Test loading vulnerability data."""
    print(" Testing data loading...")
    engineer = VulnerabilityFeatureEngineer("../data/sample_vulnerabilities.json")
    data = engineer.load_data()
    assert len(data) == 2682, f"Expected 2682 vulnerabilities, got {len(data)}"
    print("Data loading works")
    return data

def test_feature_engineering(data):
    """Test feature engineering."""
    print("Testing feature engineering...")
    engineer = VulnerabilityFeatureEngineer("../data/sample_vulnerabilities.json")
    processed = engineer.extract_features(data)
    assert processed is not None, "Feature extraction failed"
    assert processed['features'].shape[0] == 2682, f"Expected 20 samples, got {processed['features'].shape[0]}"
    print(" Feature engineering works")
    return processed

def test_model_loading():
    """Test loading the trained model."""
    print("Testing model loading")
    model, feature_engineer = CybersecurityRiskModel.load_model("model/trained_model.pkl")
    assert model.is_trained, "Model should be trained"
    assert feature_engineer is not None, "Feature engineer should be loaded"
    print("Model loading works")
    return model

def test_prediction(model, features):
    """Test making predictions."""
    print("Testing predictions")
    # Use the first sample for testing
    test_features = features[0:1]  # Take first sample
    prediction = model.predict(test_features)
    assert 0 <= prediction[0] <= 10, f"Prediction {prediction[0]} out of range 0-10"
    print(f" Prediction works: {prediction[0]:.2f}")
    return prediction[0]

def main():
    """Run all tests."""
    print("Testing Complete AI Pipeline")
    print("=" * 40)

    try:
        # Test data loading
        data = test_data_loading()

        # Test feature engineering
        processed_data = test_feature_engineering(data)

        # Test model loading
        model = test_model_loading()

        # Test prediction
        prediction = test_prediction(model, processed_data['features'])

        print("\n" + "=" * 40)
        print("ALL TESTS PASSED!")
        print("Data loading: OK")
        print("Feature engineering: OK")
        print("Model loading: OK")
        print("Predictions: OK")
        print(f"Sample prediction: {prediction:.2f}")
        print("\n The AI system is ready for use!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)