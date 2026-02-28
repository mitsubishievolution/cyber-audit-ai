"""
feature_engineering.py

Transforms raw vulnerability data into numerical features for machine learning.

This module handles the preprocessing pipeline that converts vulnerability records
(text descriptions, types, CVSS scores) into feature matrices suitable for training
regression models. Uses TF-IDF vectorization for text and one-hot encoding for categorical data.
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import os

class VulnerabilityFeatureEngineer:
    """
    Converts vulnerability data into numerical features for ML model training.
    
    This class manages the feature engineering pipeline, transforming vulnerability
    descriptions (text) and types (categorical) into numerical representations using
    TF-IDF vectorization and one-hot encoding respectively.
    """

    def __init__(self, data_path="../data/sample_vulnerabilities.json"):
        """
        Initialize the feature engineer with data source path.

        Args:
            data_path (str): Path to vulnerability data JSON file
        """
        self.data_path = data_path
        self.vectorizer = None  # TF-IDF vectorizer for text features (fitted during extract_features)
        self.encoder = None     # One-hot encoder for categorical features (fitted during extract_features)
        self.feature_names = [] # List of all feature names after transformation

    def load_data(self):
        """
        Load vulnerability data from JSON file.

        Reads the vulnerability dataset from the specified JSON file path.
        Each record should contain fields: cve, description, cvss, type.

        Returns:
            list: List of vulnerability dictionaries, or empty list if loading fails
        """
        try:
            with open(self.data_path, 'r') as file:
                data = json.load(file)

            print(f"Successfully loaded {len(data)} vulnerability examples")
            return data

        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return []

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.data_path}")
            return []

    def extract_features(self, data):
        """
        Transform raw vulnerability data into numerical feature matrix.

        This method performs the complete feature engineering pipeline:
        1. Converts text descriptions to TF-IDF vectors
        2. Encodes vulnerability types using one-hot encoding
        3. Combines all features into a single matrix

        Args:
            data (list): List of vulnerability dictionaries from load_data()

        Returns:
            dict: Contains 'features' (numpy array), 'targets' (CVSS scores),
                  'feature_names' (list), and 'original_data' (DataFrame)
                  Returns None if data is empty
        """
        if not data:
            return None

        # Convert list of dictionaries to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Extract the three components we need from each record
        descriptions = df['description'].tolist()
        vulnerability_types = df['type'].tolist()
        cvss_scores = df['cvss'].values.reshape(-1, 1)

        # Step 1: Convert text descriptions to TF-IDF vectors
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=100,      # Limit to top 100 most important terms
                stop_words='english',  # Remove common English words
                ngram_range=(1, 2)     # Consider both single words and word pairs
            )

        print("Converting text descriptions to TF-IDF vectors...")
        text_features = self.vectorizer.fit_transform(descriptions).toarray()

        # Step 2: Convert vulnerability types to one-hot encoded vectors
        if self.encoder is None:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        print("Encoding vulnerability types...")
        type_features = self.encoder.fit_transform(np.array(vulnerability_types).reshape(-1, 1))

        # Step 3: Combine all feature types into single matrix
        # Horizontally stack: [text_features | type_features | cvss_scores]
        combined_features = np.hstack([
            text_features,   # TF-IDF vectors (100 columns)
            type_features,   # One-hot encoded types (~10 columns)
            cvss_scores      # Original CVSS score (1 column)
        ])

        # Generate descriptive names for each feature column
        text_feature_names = [f"text_{i}" for i in range(text_features.shape[1])]
        type_feature_names = self.encoder.get_feature_names_out(['type'])
        cvss_feature_names = ['cvss_score']

        # Store complete list of feature names
        self.feature_names = text_feature_names + type_feature_names.tolist() + cvss_feature_names

        # Extract target values (what we're trying to predict)
        target_scores = df['cvss'].values

        # Display feature engineering summary
        print(f"Feature engineering complete:")
        print(f"  - Text features: {text_features.shape[1]}")
        print(f"  - Type features: {type_features.shape[1]}")
        print(f"  - CVSS features: 1")
        print(f"  - Total features: {combined_features.shape[1]}")

        # Return packaged results
        return {
            'features': combined_features,
            'targets': target_scores,
            'feature_names': self.feature_names,
            'original_data': df
        }

    def get_feature_importance_template(self):
        """
        Generate metadata about the engineered feature groups.

        Returns a dictionary describing each feature group (text, type, cvss),
        including counts and descriptions. Useful for interpreting feature
        importance scores from trained models.

        Returns:
            dict: Metadata for each feature group
        """
        return {
            'text_features': {
                'description': 'TF-IDF vectors from vulnerability descriptions',
                'count': len([f for f in self.feature_names if f.startswith('text_')]),
                'example': 'Terms like "SQL", "injection", "XSS" may have high importance'
            },
            'type_features': {
                'description': 'One-hot encoded vulnerability type categories',
                'count': len([f for f in self.feature_names if f.startswith('type_')]),
                'categories': self.encoder.get_feature_names_out(['type']).tolist() if self.encoder else []
            },
            'cvss_features': {
                'description': 'Original CVSS score from vulnerability record',
                'count': 1,
                'purpose': 'Provides baseline severity information to the model'
            }
        }

def main():
    """
    Test the feature engineering pipeline.

    This function provides a simple way to verify the feature engineering
    process is working correctly. It loads data, transforms it, and displays
    the resulting feature matrix dimensions.

    Run directly: python feature_engineering.py
    """
    print("=== Vulnerability Feature Engineering Test ===")

    # Initialize feature engineer
    engineer = VulnerabilityFeatureEngineer()

    # Load vulnerability data
    data = engineer.load_data()
    if not data:
        print("No data found. Check if the file exists at the specified path.")
        return

    # Transform data into features
    processed_data = engineer.extract_features(data)

    if processed_data:
        print("\n✅ Feature extraction successful!")
        print(f"Feature matrix shape: {processed_data['features'].shape}")
        print(f"  → {processed_data['features'].shape[0]} samples")
        print(f"  → {processed_data['features'].shape[1]} features per sample")
        print(f"Target vector shape: {processed_data['targets'].shape}")

        # Display feature group breakdown
        importance_info = engineer.get_feature_importance_template()
        print("\nFeature Groups:")
        for group, info in importance_info.items():
            print(f"  {group}: {info['count']} features - {info['description']}")
    else:
        print("❌ Feature extraction failed.")

if __name__ == "__main__":
    main()