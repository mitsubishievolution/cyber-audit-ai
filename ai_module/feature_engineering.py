"""
feature_engineering.py

Hello! This is the "Data Translator" file. Think of it like a chef who takes raw ingredients
(vulnerability data) and prepares them for cooking (machine learning).

What this file does:
- Takes vulnerability information (like "SQL injection in login form")
- Converts it into numbers that the computer can understand
- Prepares everything so the AI can learn from the data

Imagine you have recipes (vulnerabilities) with ingredients (descriptions) and categories (SQL Injection, XSS).
This file turns those recipes into a format the AI can "read" and learn from.

Author: University Cybersecurity Audit AI Project
"""

import json  # For reading JSON files (like our vulnerability database)
import pandas as pd  # Like Excel for Python - helps organize data in tables
import numpy as np  # Math library - handles numbers and calculations
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numbers
from sklearn.preprocessing import OneHotEncoder  # Converts categories to numbers
import os  # Helps with file paths and checking if files exist

class VulnerabilityFeatureEngineer:
    """
    This is like a "Data Chef" class. It takes messy vulnerability data and makes it
    ready for the AI to learn from.

    Think of it as a translator that converts human language about security problems
    into computer language (numbers) that the AI can understand and learn patterns from.
    """

    def __init__(self, data_path="../data/sample_vulnerabilities.json"):
        """
        This is the "setup" method that gets called when we create a new Data Chef.

        Args:
            data_path (str): The file path to our vulnerability data file
                            (like telling the chef where the recipe book is)
        """
        self.data_path = data_path  # Remember where our data lives
        self.vectorizer = None  # This will become our "text-to-numbers converter" later
        self.encoder = None     # This will become our "category-to-numbers converter" later
        self.feature_names = [] # We'll store the names of all our converted features here

    def load_data(self):
        """
        This method opens our vulnerability "recipe book" and reads all the recipes.

        Think of it like opening a cookbook and reading all the recipes inside.
        Each recipe is a vulnerability with its description, type, and danger level.

        Returns:
            list: A list of vulnerability recipes (each is a dictionary with the details)
        """
        try:
            # Open the file like opening a book
            with open(self.data_path, 'r') as file:
                # Read all the recipes from the JSON cookbook
                data = json.load(file)

            # Tell the user how many recipes we found
            print(f"Successfully loaded {len(data)} vulnerability examples")
            return data

        except FileNotFoundError:
            # If the cookbook file is missing
            print(f"Error: Data file not found at {self.data_path}")
            return []

        except json.JSONDecodeError:
            # If the cookbook has bad formatting (like a ripped page)
            print(f"Error: Invalid JSON format in {self.data_path}")
            return []

    def extract_features(self, data):
        """
        This is the main "cooking" method! It takes raw vulnerability data and turns it
        into numbers that the AI can understand and learn from.

        Imagine you have recipes with ingredients (words) and categories (types).
        This method converts them into a "numeric recipe book" that the AI can read.

        Args:
            data (list): List of vulnerability recipes from our cookbook

        Returns:
            dict: A package containing all the processed ingredients ready for cooking
        """
        if not data:
            # If there are no recipes, we can't cook!
            return None

        # Convert to DataFrame - like organizing recipes into a neat table
        df = pd.DataFrame(data)

        # Extract the three main ingredients from each recipe:
        descriptions = df['description'].tolist()  # The text descriptions (like "mix flour, sugar...")
        vulnerability_types = df['type'].tolist()  # The categories (like "Dessert", "Main Course")
        cvss_scores = df['cvss'].values.reshape(-1, 1)  # The danger scores (like "spicy level: 7")

        # Step 1: Convert text descriptions to numbers
        # We use a "Text-to-Numbers Converter" (TF-IDF Vectorizer)
        if self.vectorizer is None:
            # Create our text converter if we haven't already
            self.vectorizer = TfidfVectorizer(
                max_features=100,  # Only keep the 100 most important words
                stop_words='english',  # Ignore common words like "the", "a", "is"
                ngram_range=(1, 2)  # Look at single words AND word pairs
            )

        # Convert text to numbers - like translating a recipe into math!
        print("Converting text descriptions to TF-IDF vectors...")
        text_features = self.vectorizer.fit_transform(descriptions).toarray()

        # Step 2: Convert vulnerability types to numbers
        # We use a "Category-to-Numbers Converter" (One-Hot Encoder)
        if self.encoder is None:
            # Create our category converter if we haven't already
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Convert categories to numbers - like assigning numbers to recipe types
        print("Encoding vulnerability types...")
        type_features = self.encoder.fit_transform(np.array(vulnerability_types).reshape(-1, 1))

        # Step 3: Combine all ingredients into one big recipe book
        # We stack them side by side like combining ingredients in a mixing bowl
        combined_features = np.hstack([
            text_features,      # Text ingredients (100 columns)
            type_features,      # Category ingredients (10 columns)
            cvss_scores         # Danger score ingredient (1 column)
        ])

        # Create names for each ingredient so we can remember what they are
        text_feature_names = [f"text_{i}" for i in range(text_features.shape[1])]  # text_0, text_1, etc.
        type_feature_names = self.encoder.get_feature_names_out(['type'])         # type_SQL Injection, etc.
        cvss_feature_names = ['cvss_score']                                       # The danger score

        # Store all ingredient names together
        self.feature_names = text_feature_names + type_feature_names.tolist() + cvss_feature_names

        # Extract the "correct answers" - what danger score each recipe should have
        # For now, we use the CVSS score as our target (what we're trying to predict)
        target_scores = df['cvss'].values

        # Tell the user what we accomplished
        print(f"Feature engineering complete:")
        print(f"  - Text features: {text_features.shape[1]} (words converted to numbers)")
        print(f"  - Type features: {type_features.shape[1]} (categories converted to numbers)")
        print(f"  - CVSS features: 1 (danger score)")
        print(f"  - Total features: {combined_features.shape[1]} (all ingredients combined)")

        # Package everything up and return it
        return {
            'features': combined_features,    # All the numeric ingredients
            'targets': target_scores,         # The correct danger scores
            'feature_names': self.feature_names,  # Names of all ingredients
            'original_data': df               # The original recipes (for reference)
        }

    def get_feature_importance_template(self):
        """
        This method creates a "map" or "legend" that explains what each group of
        ingredients (features) represents in our recipe book.

        It's like having labels on different sections of your kitchen:
        - "Spices" section
        - "Vegetables" section
        - "Proteins" section

        This helps us understand which ingredients the AI thinks are most important.

        Returns:
            dict: A guidebook explaining what each feature group means
        """
        return {
            'text_features': {
                'description': 'Word ingredients from vulnerability descriptions',
                'count': len([f for f in self.feature_names if f.startswith('text_')]),
                'example': 'Words like "SQL", "injection", "XSS" get higher importance scores'
            },
            'type_features': {
                'description': 'Category labels turned into numbers',
                'count': len([f for f in self.feature_names if f.startswith('type_')]),
                'categories': self.encoder.get_feature_names_out(['type']).tolist() if self.encoder else []
            },
            'cvss_features': {
                'description': 'The original danger score from the vulnerability database',
                'count': 1,
                'purpose': 'Helps the AI understand the base danger level of each vulnerability'
            }
        }

def main():
    """
    This is the "test kitchen" function! It's like trying out a new recipe before
    serving it to guests. This function lets us test our data translation system
    to make sure everything works correctly.

    When you run this file directly (python feature_engineering.py), it will:
    1. Load our vulnerability recipes
    2. Convert them to numbers
    3. Show us what the final result looks like
    """
    print("=== Vulnerability Feature Engineering Test ===")
    print("Welcome to the test kitchen! Let's see if our data translation works.")

    # Hire our Data Chef
    engineer = VulnerabilityFeatureEngineer()

    # Load the recipes from our cookbook
    data = engineer.load_data()
    if not data:
        print("No recipes found in the cookbook. Check if the file exists.")
        return

    # Let the chef translate the recipes into computer language
    processed_data = engineer.extract_features(data)

    if processed_data:
        print("\nSuccess! Feature extraction worked!")
        print(f"Final recipe book shape: {processed_data['features'].shape}")
        print(f"(This means {processed_data['features'].shape[0]} recipes with {processed_data['features'].shape[1]} numeric ingredients each)")
        print(f"Target scores shape: {processed_data['targets'].shape}")
        print(f"(This means {processed_data['targets'].shape[0]} correct danger scores to learn from)")

        # Show what types of ingredients we created
        importance_info = engineer.get_feature_importance_template()
        print("\nIngredient Types Created:")
        for group, info in importance_info.items():
            print(f"  {group}: {info['count']} ingredients - {info['description']}")
    else:
        print("Feature extraction failed. The chef couldn't translate the recipes.")

if __name__ == "__main__":
    main()