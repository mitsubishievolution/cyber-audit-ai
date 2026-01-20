"""
model_trainer.py

Hello! This is the "AI Brain Builder" file. Think of it like a personal trainer
who teaches an athlete (our AI) how to make good decisions.

What this file does:
- Takes our prepared vulnerability data (from the Data Chef)
- Trains a Random Forest AI model to predict risk scores
- Tests how well the AI learned
- Saves the trained AI brain for later use

Imagine you have a student who needs to learn how dangerous different security
problems are. This file teaches the student by showing examples and testing
how well they understand the patterns.

Author: University Cybersecurity Audit AI Project
"""

import numpy as np  # For math operations and number handling
from sklearn.ensemble import RandomForestRegressor  # The AI algorithm we'll use
from sklearn.model_selection import train_test_split, cross_val_score  # Tools to split and test data
import joblib  # Tool to save and load our trained AI brain
import os  # File system operations
from feature_engineering import VulnerabilityFeatureEngineer  # Our Data Chef
from utils import evaluate_model_performance, ensure_directory_exists  # Helper tools

class CybersecurityRiskModel:
    """
    This is our "AI Student" class. It represents the AI brain that will learn
    to predict cybersecurity risk scores.

    Think of it like a student who:
    - Starts as a beginner (untrained)
    - Learns from examples (training)
    - Gets tested on new problems (prediction)
    - Gets better with practice (more training data)
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        This is like "creating a new student" for our AI class.

        Args:
            n_estimators (int): How many "mini-brains" (decision trees) to create
                               More trees = smarter but slower
            max_depth (int): How deep each mini-brain can think
                            Deeper = more detailed thinking but slower
            random_state (int): A seed to make results consistent (like same starting point)
        """
        # Create our AI student (Random Forest algorithm)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,  # Number of decision trees
            max_depth=max_depth,        # How deep each tree can grow
            random_state=random_state,  # For consistent results
            n_jobs=-1                   # Use all computer cores for speed
        )
        self.is_trained = False          # The student hasn't learned yet
        self.feature_names = []          # We'll remember what ingredients we taught with
        self.model_params = {            # Remember our settings
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }

    def train(self, X, y, test_size=0.2):
        """
        This is the main "teaching" method! It teaches our AI student using examples.

        Think of it like teaching a child to recognize dangerous animals:
        - Show them pictures of lions, tigers, bears (training data)
        - Tell them "these are dangerous!" (target scores)
        - Test them on new animals they haven't seen (testing data)
        - See how well they learned (evaluation metrics)

        Args:
            X (array): The "pictures" - our numeric features (ingredients from Data Chef)
            y (array): The "answers" - correct danger scores for each vulnerability
            test_size (float): What fraction to save for testing (like keeping some quiz questions secret)

        Returns:
            dict: A report card showing how well the student learned
        """
        print("🚀 Starting AI model training...")
        print(f"   - Training data shape: {X.shape}")
        print(f"   - Target values shape: {y.shape}")
        print(f"   - Test size: {test_size*100}%")

        # Split our lesson materials into study materials and quiz questions
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.model_params['random_state']
        )

        print(f"   - Study materials: {X_train.shape[0]} examples (what the AI learns from)")
        print(f"   - Quiz questions: {X_test.shape[0]} examples (what we test the AI on)")

        # Let the AI student study the examples
        print("   - Teaching the Random Forest algorithm...")
        self.model.fit(X_train, y_train)  # This is where the AI learns!
        self.is_trained = True  # The student is now educated

        # Give the student a quiz on the secret questions
        y_pred = self.model.predict(X_test)  # AI makes predictions on test data

        # Grade the student's performance
        metrics = evaluate_model_performance(y_test, y_pred, "Random Forest Risk Model")

        # Do some extra testing to make sure the student really understands
        # This is like giving the student multiple quizzes with different questions
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error'  # Test 5 times with different question sets
        )
        cv_rmse_scores = np.sqrt(-cv_scores)  # Convert to positive error scores

        print("\n📊 Multiple Quiz Results (Cross-validation):")
        print(f"   - Quiz scores: {cv_rmse_scores}")
        print(f"   - Average quiz score: {cv_rmse_scores.mean():.3f} (lower is better)")
        print(f"   - Quiz score consistency: {cv_rmse_scores.std():.3f} (lower is more consistent)")

        # Add the extra quiz results to our report card
        metrics['cv_mean_rmse'] = float(cv_rmse_scores.mean())
        metrics['cv_std_rmse'] = float(cv_rmse_scores.std())

        print("✅ AI training completed successfully! The student is ready to predict risk scores.")

        # Return the complete report card
        return {
            'model': self.model,           # The trained AI student
            'metrics': metrics,           # Report card with grades
            'X_train': X_train,           # Study materials used
            'X_test': X_test,             # Quiz questions
            'y_train': y_train,           # Correct answers for study materials
            'y_test': y_test,             # Correct answers for quiz
            'y_pred': y_pred              # AI's quiz answers
        }

    def predict(self, X):
        """
        This is like giving our trained AI student a new test! The student uses
        what they learned to predict danger scores for vulnerabilities they've never seen.

        Think of it like:
        - Student studied lions, tigers, bears
        - Now we show them a wolf and ask "how dangerous is this?"
        - Student says "8 out of 10" based on learned patterns

        Args:
            X (array): New "test questions" - numeric features of vulnerabilities to analyze

        Returns:
            array: The AI's predictions (danger scores from 0-10)
        """
        if not self.is_trained:
            raise ValueError("Cannot take a test without studying first! Train the model before predicting.")

        # Ask the AI student to make predictions
        return self.model.predict(X)

    def get_feature_importance(self):
        """
        This method asks "which ingredients were most important for learning?"
        It's like asking a chef "which spices made the biggest difference in your recipes?"

        The AI tells us which features (ingredients) it found most useful for making predictions.
        For example, maybe "SQL" words were very important for predicting database risks.

        Returns:
            array: Scores showing how important each ingredient was (0-1 scale)
        """
        if not self.is_trained:
            raise ValueError("Cannot check ingredient importance until the AI has been trained!")

        # Ask the AI which ingredients it found most useful
        return self.model.feature_importances_

    def save_model(self, filepath, feature_engineer=None):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path where to save the model
            feature_engineer: The feature engineering object used for training
        """
        ensure_directory_exists(os.path.dirname(filepath))

        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'feature_engineer': feature_engineer  # Save the fitted feature engineer
        }

        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model

        Returns:
            tuple: (CybersecurityRiskModel, feature_engineer) - Loaded model and feature engineer
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        model_data = joblib.load(filepath)

        # Create new instance
        instance = cls(
            n_estimators=model_data['model_params']['n_estimators'],
            max_depth=model_data['model_params']['max_depth'],
            random_state=model_data['model_params']['random_state']
        )

        # Restore trained state
        instance.model = model_data['model']
        instance.is_trained = model_data['is_trained']
        instance.feature_names = model_data.get('feature_names', [])

        # Return both model and feature engineer
        feature_engineer = model_data.get('feature_engineer')

        print(f"📂 Model loaded from {filepath}")
        return instance, feature_engineer

def train_complete_model(data_path="../data/sample_vulnerabilities.json",
                        model_save_path="model/trained_model.pkl"):
    """
    Complete pipeline to train the AI model from start to finish.

    Args:
        data_path (str): Path to vulnerability data
        model_save_path (str): Where to save the trained model

    Returns:
        dict: Training results and metrics
    """
    print("🎯 Starting Complete AI Model Training Pipeline")
    print("=" * 50)

    # Step 1: Load and process data
    print("\n📥 Step 1: Loading and processing vulnerability data...")
    engineer = VulnerabilityFeatureEngineer(data_path)
    raw_data = engineer.load_data()

    if not raw_data:
        print("❌ No data available for training")
        return None

    processed_data = engineer.extract_features(raw_data)
    if processed_data is None:
        print("❌ Feature extraction failed")
        return None

    # Step 2: Initialize and train model
    print("\n🤖 Step 2: Training RandomForest risk prediction model...")
    model = CybersecurityRiskModel(
        n_estimators=100,  # Number of trees
        max_depth=10,      # Maximum tree depth
        random_state=42    # For reproducible results
    )

    # Store feature names for later interpretation
    model.feature_names = processed_data['feature_names']

    # Train the model
    training_results = model.train(
        processed_data['features'],
        processed_data['targets'],
        test_size=0.2
    )

    # Step 3: Save the trained model
    print(f"\n💾 Step 3: Saving trained model to {model_save_path}...")
    model.save_model(model_save_path, engineer)

    # Step 4: Generate feature importance analysis
    print("\n📊 Step 4: Analyzing feature importance...")
    feature_importance = model.get_feature_importance()

    # Create importance summary
    importance_summary = []
    for name, importance in zip(model.feature_names, feature_importance):
        if importance > 0.01:  # Only show features with >1% importance
            importance_summary.append({
                'feature': name,
                'importance': float(importance)
            })

    # Sort by importance
    importance_summary.sort(key=lambda x: x['importance'], reverse=True)

    print("Top 5 most important features:")
    for i, item in enumerate(importance_summary[:5]):
        print(".4f")

    # Step 5: Final summary
    print("\n🎉 AI Model Training Complete!")
    print(f"   - Model saved: {model_save_path}")
    print(f"   - Training accuracy: {training_results['metrics']['accuracy_within_1_point']}% within 1 point")
    print(f"   - Ready for predictions!")

    return {
        'model': model,
        'training_results': training_results,
        'feature_importance': importance_summary,
        'engineer': engineer,
        'processed_data': processed_data
    }

def main():
    """
    Main function to run the complete training pipeline.
    This can be executed directly to train the model.
    """
    # Ensure model directory exists
    ensure_directory_exists("model")

    # Run complete training
    results = train_complete_model()

    if results:
        print("\n✅ Training pipeline completed successfully!")
        print("You can now use the trained model for risk predictions.")
    else:
        print("\n❌ Training pipeline failed. Check the error messages above.")

if __name__ == "__main__":
    main()