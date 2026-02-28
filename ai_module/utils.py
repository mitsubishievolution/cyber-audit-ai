"""
utils.py

Utility functions for the cybersecurity audit AI system.

This module provides helper functions for:
- Loading and validating vulnerability data from JSON files
- Evaluating model performance using regression metrics (MAE, RMSE, R²)
- Generating visualization plots for model predictions
- Managing file system operations (directories, saving/loading results)
"""

import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_vulnerability_data(data_path):
    """
    Load vulnerability data from a JSON file with error handling.

    Performs validation checks to ensure the file exists, is valid JSON,
    and contains a list of vulnerability records.

    Args:
        data_path (str): Path to the vulnerability data JSON file

    Returns:
        list or None: List of vulnerability dictionaries, or None if loading fails
    """
    try:
        # Check if file exists before attempting to open
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            return None

        # Load JSON data from file
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Validate that data is a list (not a single object or other type)
        if not isinstance(data, list):
            print("Error: Data file must contain a list of vulnerability records")
            return None

        # Check for empty dataset
        if len(data) == 0:
            print("Warning: Data file is empty - no vulnerability records found")
            return []

        print(f"Successfully loaded {len(data)} vulnerability records")
        return data

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return None
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None

def validate_vulnerability_record(record):
    """
    Validate that a vulnerability record has all required fields.

    Args:
        record (dict): Single vulnerability record

    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['cve', 'description', 'cvss', 'type']

    for field in required_fields:
        if field not in record:
            print(f"Error: Missing required field '{field}' in record")
            return False

    # Validate CVSS score range
    if not isinstance(record['cvss'], (int, float)) or not (0 <= record['cvss'] <= 10):
        print(f"Error: CVSS score must be a number between 0 and 10, got {record['cvss']}")
        return False

    # Validate description length
    if not isinstance(record['description'], str) or len(record['description'].strip()) == 0:
        print("Error: Description must be a non-empty string")
        return False

    return True

def validate_dataset(data):
    """
    Validate the entire dataset for consistency.

    Args:
        data (list): List of vulnerability records

    Returns:
        bool: True if all records are valid, False otherwise
    """
    if not data:
        print("Error: Dataset is empty")
        return False

    valid_records = 0
    for i, record in enumerate(data):
        if validate_vulnerability_record(record):
            valid_records += 1
        else:
            print(f"Invalid record at index {i}")

    if valid_records != len(data):
        print(f"Error: Only {valid_records}/{len(data)} records are valid")
        return False

    return True

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """
    Evaluate regression model performance using multiple metrics.

    Calculates MAE, RMSE, and R² scores, then checks against dissertation
    success criteria (R² ≥ 0.75, MAE ≤ 1.0). Also computes accuracy within
    tolerance thresholds (1.0 and 2.0 points).

    Args:
        y_true (array): Ground truth CVSS scores
        y_pred (array): Predicted risk scores from model
        model_name (str): Name to display in output (default: "Model")

    Returns:
        dict: Dictionary containing all evaluation metrics and pass/fail status
    """
    # Define dissertation success thresholds
    R2_THRESHOLD = 0.75   # Minimum acceptable R² score
    MAE_THRESHOLD = 1.0   # Maximum acceptable mean absolute error
    
    # Compute regression metrics
    mse = mean_squared_error(y_true, y_pred)  # Mean squared error
    rmse = np.sqrt(mse)                       # Root mean squared error
    r2 = r2_score(y_true, y_pred)             # Coefficient of determination

    # Compute mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))

    # Calculate accuracy within tolerance thresholds
    within_1_point = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    within_2_points = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100

    # Check against dissertation success criteria
    r2_pass = r2 >= R2_THRESHOLD
    mae_pass = mae <= MAE_THRESHOLD
    overall_pass = r2_pass and mae_pass

    # Package metrics into dictionary
    metrics = {
        'model_name': model_name,
        'mean_squared_error': round(mse, 4),
        'root_mean_squared_error': round(rmse, 4),
        'mean_absolute_error': round(mae, 4),
        'r_squared': round(r2, 4),
        'accuracy_within_1_point': round(within_1_point, 2),
        'accuracy_within_2_points': round(within_2_points, 2),
        'sample_count': len(y_true),
        'r2_threshold': R2_THRESHOLD,
        'mae_threshold': MAE_THRESHOLD,
        'r2_pass': r2_pass,
        'mae_pass': mae_pass,
        'overall_pass': overall_pass
    }

    # Display performance metrics
    print(f"\n=== {model_name} Performance Report ===")
    print(f"Test Set Size: {metrics['sample_count']} vulnerabilities")
    print(f"\n Key Metrics:")
    print(f"   Mean Absolute Error (MAE):  {metrics['mean_absolute_error']:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {metrics['root_mean_squared_error']:.4f}")
    print(f"   R² Score (R-squared):       {metrics['r_squared']:.4f}")
    print(f"\n🎯 Accuracy Benchmarks:")
    print(f"   Within 1.0 point:  {metrics['accuracy_within_1_point']:.2f}%")
    print(f"   Within 2.0 points: {metrics['accuracy_within_2_points']:.2f}%")

    # Display dissertation success criteria evaluation
    print(f"\n{'='*60}")
    print("DISSERTATION SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")
    print(f"R² Score:  {r2:.4f} | Target: ≥ {R2_THRESHOLD} | {'✅ PASS' if r2_pass else '❌ FAIL'}")
    print(f"MAE:       {mae:.4f} | Target: ≤ {MAE_THRESHOLD} | {'✅ PASS' if mae_pass else '❌ FAIL'}")
    print(f"{'='*60}")
    
    # Print overall verdict
    if overall_pass:
        print("SUCCESS: Model meets all dissertation requirements!")
    else:
        print(" Model does not meet all dissertation requirements.")
        if not r2_pass:
            print(f"   - R² score {r2:.4f} is below threshold {R2_THRESHOLD}")
        if not mae_pass:
            print(f"   - MAE {mae:.4f} is above threshold {MAE_THRESHOLD}")
    print(f"{'='*60}\n")

    return metrics

def plot_prediction_comparison(y_true, y_pred, save_path=None):
    """
    Generate scatter plot comparing predicted vs actual CVSS scores.

    Creates a visualization with:
    - Blue scatter points for each prediction
    - Red dashed diagonal line showing perfect prediction
    - Text box displaying MSE and R² statistics

    Args:
        y_true (array): Ground truth CVSS scores
        y_pred (array): Model predicted scores
        save_path (str, optional): File path to save plot image (if provided)
    """
    plt.figure(figsize=(10, 6))

    # Plot predicted vs actual as scatter points
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='black')

    # Draw diagonal line representing perfect predictions
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual CVSS Score', fontsize=12)
    plt.ylabel('Predicted Risk Score', fontsize=12)
    plt.title('Model Performance: Predicted vs Actual Risk Scores', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add performance statistics as text overlay
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    stats_text = f'MSE: {mse:.3f}\nR²: {r2:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')

    # Save to file if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def get_project_root():
    """
    Get the absolute path to the project root directory.

    Returns:
        str: Absolute path to project root (parent of ai_module directory)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Move up one level from ai_module
    return project_root

def ensure_directory_exists(directory_path):
    """
    Create directory if it doesn't exist.

    Args:
        directory_path (str): Path to directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_evaluation_results(metrics, filepath):
    """
    Save evaluation metrics to JSON file.

    Creates parent directory if needed, then writes metrics dictionary
    to JSON file with indentation for readability.

    Args:
        metrics (dict): Evaluation metrics to save
        filepath (str): Destination path for JSON file
    """
    ensure_directory_exists(os.path.dirname(filepath))

    try:
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation results saved to {filepath}")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")

def load_evaluation_results(filepath):
    """
    Load evaluation metrics from JSON file.

    Args:
        filepath (str): Path to JSON file containing metrics

    Returns:
        dict or None: Loaded metrics dictionary, or None if loading fails
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        return None

def print_model_summary(feature_engineer, model, metrics):
    """
    Display comprehensive summary of trained model.

    Prints formatted summary including dataset size, feature count,
    performance metrics, and feature engineering breakdown.

    Args:
        feature_engineer: VulnerabilityFeatureEngineer instance
        model: Trained CybersecurityRiskModel instance
        metrics (dict): Evaluation metrics from evaluate_model_performance()
    """
    print("\n" + "="*60)
    print(" CYBERSECURITY AUDIT AI MODEL SUMMARY")
    print("="*60)

    print(f"\n Dataset Information:")
    print(f"   - Total vulnerabilities: {metrics['sample_count']}")
    print(f"   - Features used: {len(feature_engineer.feature_names)}")

    print(f"\n Model Performance:")
    print(f"   - Mean Absolute Error: {metrics['mean_absolute_error']}")
    print(f"   - Predictions within 1 point: {metrics['accuracy_within_1_point']}%")
    print(f"   - R² Score: {metrics['r_squared']}")

    print(f"\n Feature Engineering:")
    importance_info = feature_engineer.get_feature_importance_template()
    for group, info in importance_info.items():
        print(f"   - {group}: {info['count']} features")

    print(f"\n✅ Model Status: Ready for predictions!")
    print("="*60)