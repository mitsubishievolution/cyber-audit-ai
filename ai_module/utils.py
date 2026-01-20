"""
utils.py

Hello! This is the "Helper Toolbox" file. Think of it like a toolbox full of useful
tools that other parts of our AI system can borrow when they need help.

What this toolbox contains:
- Data loading tools (like opening files safely)
- Model grading tools (like checking how well our AI student did)
- File management tools (like creating folders and saving files)
- Chart-making tools (like creating graphs to show results)

Imagine you're building a treehouse with friends. Some friends are good at cutting wood,
some are good at hammering nails, some are good at drawing plans. This file is like the
shared toolbox where everyone can find the tools they need.

Author: University Cybersecurity Audit AI Project
"""

import json          # For reading and writing JSON files (our data format)
import os            # For working with files and folders on the computer
import numpy as np   # For math operations on numbers
from sklearn.metrics import mean_squared_error, r2_score  # For grading AI performance
import matplotlib.pyplot as plt  # For creating charts and graphs

def load_vulnerability_data(data_path):
    """
    This is like a "safe file opener" tool. It carefully opens our vulnerability
    database file and makes sure everything is okay before giving it to other tools.

    Think of it like a librarian who:
    - Checks if the book exists on the shelf
    - Opens it carefully without damaging the pages
    - Counts how many recipes are inside
    - Makes sure the book is in the right format

    Args:
        data_path (str): The file path to our vulnerability "cookbook"

    Returns:
        list or None: A list of vulnerability recipes, or None if something went wrong
    """
    try:
        # First, check if the file actually exists (like checking if book is on shelf)
        if not os.path.exists(data_path):
            print(f"Error: Cookbook not found at {data_path}")
            return None

        # Open the file carefully and read all the recipes
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load the JSON data

        # Make sure we got a list of recipes (not a single recipe or something else)
        if not isinstance(data, list):
            print("Error: Cookbook must contain a list of vulnerability recipes")
            return None

        # Check if the cookbook has any recipes at all
        if len(data) == 0:
            print("Warning: Cookbook is empty - no recipes found")
            return []

        # Success! Tell the user how many recipes we found
        print(f"Successfully loaded {len(data)} vulnerability recipes from the cookbook")
        return data

    except json.JSONDecodeError as e:
        # The cookbook has damaged pages or bad formatting
        print(f"Error: Cookbook has bad formatting - {e}")
        return None
    except Exception as e:
        # Something unexpected went wrong
        print(f"Error opening cookbook: {e}")
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

def evaluate_model_performance(y_true, y_pred, model_name="AI Student"):
    """
    This is the "Grade Calculator" tool! It grades how well our AI student performed
    on their test, using several different scoring methods.

    Think of it like a teacher grading a student's quiz:
    - How close were the answers to correct? (Mean Absolute Error)
    - What percentage got within 1 point? (Accuracy within 1 point)
    - How consistent were the mistakes? (R² Score)

    Args:
        y_true (array): The correct answers (real danger scores)
        y_pred (array): The AI's answers (predicted danger scores)
        model_name (str): What to call our AI student in the report

    Returns:
        dict: A report card with all the grades and scores
    """
    # Calculate different types of "grades" for the AI student
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error (punishes big mistakes more)
    rmse = np.sqrt(mse)                       # Root Mean Squared Error (easier to understand)
    r2 = r2_score(y_true, y_pred)             # R² Score (how well the AI explains the data)

    # Calculate average mistake size
    mae = np.mean(np.abs(y_true - y_pred))    # Mean Absolute Error

    # Calculate "close enough" percentages
    within_1_point = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100  # Within 1 point of correct
    within_2_points = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100 # Within 2 points of correct

    # Package all the grades into a report card
    metrics = {
        'model_name': model_name,
        'mean_squared_error': round(mse, 4),          # Overall error score
        'root_mean_squared_error': round(rmse, 4),    # Easier to understand error
        'mean_absolute_error': round(mae, 4),         # Average mistake size
        'r_squared': round(r2, 4),                    # How well AI explains patterns
        'accuracy_within_1_point': round(within_1_point, 2),  # % within 1 point
        'accuracy_within_2_points': round(within_2_points, 2), # % within 2 points
        'sample_count': len(y_true)                   # How many questions were asked
    }

    # Print the report card in a nice format
    print(f"\n=== {model_name} Report Card ===")
    print(f"Number of test questions: {metrics['sample_count']} vulnerabilities")
    print(f"Average mistake size: {metrics['mean_absolute_error']} points")
    print(f"Overall pattern understanding: {metrics['r_squared']} (closer to 1.0 is better)")
    print(f"Got within 1 point of correct: {metrics['accuracy_within_1_point']}%")
    print(f"Got within 2 points of correct: {metrics['accuracy_within_2_points']}%")

    # Give some encouragement based on performance
    if metrics['accuracy_within_1_point'] > 70:
        print("Excellent work! The AI student is very accurate.")
    elif metrics['accuracy_within_1_point'] > 50:
        print("Good job! The AI student is learning well.")
    else:
        print("Keep practicing! The AI student needs more training data.")

    return metrics

def plot_prediction_comparison(y_true, y_pred, save_path=None):
    """
    Create a scatter plot comparing true vs predicted values.

    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        save_path (str, optional): Path to save the plot image
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of predictions vs actual
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='black')

    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual CVSS Score', fontsize=12)
    plt.ylabel('Predicted Risk Score', fontsize=12)
    plt.title('AI Model: Predicted vs Actual Risk Scores', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add statistics text
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    stats_text = f'MSE: {mse:.3f}\nR²: {r2:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def get_project_root():
    """
    Get the root directory of the project.

    Returns:
        str: Absolute path to project root
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from ai_module to project root
    project_root = os.path.dirname(current_dir)
    return project_root

def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_evaluation_results(metrics, filepath):
    """
    Save model evaluation results to a JSON file.

    Args:
        metrics (dict): Evaluation metrics dictionary
        filepath (str): Path to save the results
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
    Load model evaluation results from a JSON file.

    Args:
        filepath (str): Path to the results file

    Returns:
        dict or None: Loaded metrics, or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        return None

def print_model_summary(feature_engineer, model, metrics):
    """
    Print a comprehensive summary of the trained model.

    Args:
        feature_engineer: The feature engineering object
        model: The trained ML model
        metrics (dict): Evaluation metrics
    """
    print("\n" + "="*60)
    print("🤖 CYBERSECURITY AUDIT AI MODEL SUMMARY")
    print("="*60)

    print(f"\n📊 Dataset Information:")
    print(f"   - Total vulnerabilities: {metrics['sample_count']}")
    print(f"   - Features used: {len(feature_engineer.feature_names)}")

    print(f"\n🎯 Model Performance:")
    print(f"   - Mean Absolute Error: {metrics['mean_absolute_error']}")
    print(f"   - Predictions within 1 point: {metrics['accuracy_within_1_point']}%")
    print(f"   - R² Score: {metrics['r_squared']}")

    print(f"\n🔧 Feature Engineering:")
    importance_info = feature_engineer.get_feature_importance_template()
    for group, info in importance_info.items():
        print(f"   - {group}: {info['count']} features")

    print(f"\n✅ Model Status: Ready for predictions!")
    print("="*60)