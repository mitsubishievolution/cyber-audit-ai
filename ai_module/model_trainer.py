"""
model_trainer.py

Trains a Random Forest regression model to predict CVSS risk scores for cybersecurity vulnerabilities.

Functionality:
- Loads and processes vulnerability data from JSON
- Trains RandomForestRegressor on engineered features
- Evaluates model performance using MAE, RMSE, and R² metrics
- Saves trained model and feature engineer for inference
- Generates performance visualization graphs
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
from feature_engineering import VulnerabilityFeatureEngineer
from utils import evaluate_model_performance, ensure_directory_exists, plot_prediction_comparison

class CybersecurityRiskModel:
    """
    Random Forest regression model for predicting vulnerability risk scores.
    
    Wraps sklearn's RandomForestRegressor with training, evaluation, and persistence methods.
    Maintains training state and model hyperparameters for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize the Random Forest model with specified hyperparameters.

        Args:
            n_estimators (int): Number of decision trees in the forest
            max_depth (int): Maximum depth of each decision tree
            random_state (int): Seed for random number generator (ensures reproducibility)
        """
        # Initialize RandomForestRegressor with specified hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available CPU cores for parallel processing
        )
        self.is_trained = False  # Training status flag
        self.feature_names = []  # Store feature names for interpretability
        self.model_params = {  # Store hyperparameters for model persistence
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }

    def train(self, X, y, test_size=0.2):
        """
        Train the Random Forest model and evaluate performance on test set.

        Splits data into train/test sets, fits the model, generates predictions,
        and evaluates using MAE, RMSE, R², and cross-validation metrics.

        Args:
            X (array): Feature matrix (n_samples, n_features)
            y (array): Target values (CVSS scores)
            test_size (float): Proportion of data to use for testing (default: 0.2)

        Returns:
            dict: Training results containing model, metrics, and train/test data splits
        """
        print(" Starting AI model training...")
        print(f"   - Training data shape: {X.shape}")
        print(f"   - Target values shape: {y.shape}")
        print(f"   - Test size: {test_size*100}%")

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.model_params['random_state']
        )

        print(f"   - Training set: {X_train.shape[0]} samples")
        print(f"   - Test set: {X_test.shape[0]} samples")

        # Fit the Random Forest model on training data
        print("   - Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Generate predictions on test set
        y_pred = self.model.predict(X_test)

        # Evaluate model performance against dissertation criteria
        metrics = evaluate_model_performance(y_test, y_pred, "Random Forest Risk Model")

        # Perform 5-fold cross-validation on training set
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse_scores = np.sqrt(-cv_scores)  # Convert negative MSE to RMSE

        print("\n Cross-Validation Results:")
        print(f"   - RMSE scores: {cv_rmse_scores}")
        print(f"   - Mean RMSE: {cv_rmse_scores.mean():.3f}")
        print(f"   - Std Dev: {cv_rmse_scores.std():.3f}")

        # Store cross-validation metrics
        metrics['cv_mean_rmse'] = float(cv_rmse_scores.mean())
        metrics['cv_std_rmse'] = float(cv_rmse_scores.std())

        print("✅ Model training completed successfully")

        # Return training results
        return {
            'model': self.model,
            'metrics': metrics,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def predict(self, X):
        """
        Generate risk score predictions for new vulnerability data.

        Args:
            X (array): Feature matrix for vulnerabilities to score

        Returns:
            array: Predicted CVSS risk scores (0-10 scale)
        
        Raises:
            ValueError: If model has not been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Generate predictions using trained Random Forest
        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Extract feature importance scores from the trained Random Forest.
        
        Returns feature importance values indicating which features contributed
        most to the model's predictions. Higher values indicate more important features.

        Returns:
            array: Feature importance scores (sum to 1.0)
        
        Raises:
            ValueError: If model has not been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before accessing feature importances")

        # Return feature importance from Random Forest
        return self.model.feature_importances_

    def save_model(self, filepath, feature_engineer=None):
        """
        Serialize and save the trained model to disk using joblib.

        Args:
            filepath (str): Destination path for model file (.pkl)
            feature_engineer: Fitted VulnerabilityFeatureEngineer instance
        """
        # Create directory if it doesn't exist
        ensure_directory_exists(os.path.dirname(filepath))

        # Package model and metadata
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'feature_engineer': feature_engineer
        }

        # Serialize to disk
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a previously saved model from disk.

        Args:
            filepath (str): Path to saved model file (.pkl)

        Returns:
            tuple: (CybersecurityRiskModel instance, feature_engineer)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        # Deserialize model data
        model_data = joblib.load(filepath)

        # Reconstruct model instance with original hyperparameters
        instance = cls(
            n_estimators=model_data['model_params']['n_estimators'],
            max_depth=model_data['model_params']['max_depth'],
            random_state=model_data['model_params']['random_state']
        )

        # Restore trained model state
        instance.model = model_data['model']
        instance.is_trained = model_data['is_trained']
        instance.feature_names = model_data.get('feature_names', [])

        # Extract feature engineer
        feature_engineer = model_data.get('feature_engineer')

        print(f"📂 Model loaded from {filepath}")
        return instance, feature_engineer

def train_complete_model(data_path="../data/sample_vulnerabilities.json",
                        model_save_path="model/trained_model.pkl"):
    """
    Execute complete training pipeline: load data, train model, evaluate, and save.

    Pipeline steps:
    1. Load and engineer features from vulnerability JSON
    2. Train Random Forest model on engineered features
    3. Generate performance visualization (scatter plot)
    4. Save trained model and feature engineer to disk
    5. Compute and display feature importance rankings

    Args:
        data_path (str): Path to vulnerability data JSON file
        model_save_path (str): Destination path for trained model (.pkl)

    Returns:
        dict: Training results containing model, metrics, feature importance, and data
              Returns None if training fails
    """
    print("🎯 Starting Complete AI Model Training Pipeline")
    print("=" * 50)

    # Step 1: Load and engineer features from vulnerability data
    print("\n📥 Loading and processing vulnerability data...")
    engineer = VulnerabilityFeatureEngineer(data_path)
    raw_data = engineer.load_data()

    if not raw_data:
        print("❌ No data available for training")
        return None

    processed_data = engineer.extract_features(raw_data)
    if processed_data is None:
        print("❌ Feature extraction failed")
        return None

    # Step 2: Initialize Random Forest model with hyperparameters
    print("\n🤖 Training RandomForest risk prediction model...")
    model = CybersecurityRiskModel(
        n_estimators=100,  # Number of decision trees in ensemble
        max_depth=10,      # Maximum tree depth
        random_state=42    # Seed for reproducibility
    )

    # Attach feature names to model for interpretability
    model.feature_names = processed_data['feature_names']

    # Train model and evaluate on test set
    training_results = model.train(
        processed_data['features'],
        processed_data['targets'],
        test_size=0.2
    )

    # Step 3: Generate scatter plot visualization of predictions vs actual
    print("\n📈 Generating performance graphs...")
    graph_save_path = "model/training_performance.png"
    plot_prediction_comparison(
        training_results['y_test'],   # Ground truth CVSS scores
        training_results['y_pred'],   # Model predictions
        save_path=graph_save_path
    )
    print(f"   ✅ Performance graph saved to: {graph_save_path}")

    # Step 4: Serialize trained model and feature engineer to disk
    print(f"\n Saving trained model to {model_save_path}...")
    model.save_model(model_save_path, engineer)

    # Step 5: Extract and rank feature importance scores
    print("\n Analyzing feature importance...")
    feature_importance = model.get_feature_importance()

    # Build list of features with >1% importance
    importance_summary = []
    for name, importance in zip(model.feature_names, feature_importance):
        if importance > 0.01:
            importance_summary.append({
                'feature': name,
                'importance': float(importance)
            })

    # Sort features by importance (descending)
    importance_summary.sort(key=lambda x: x['importance'], reverse=True)

    print("Top 5 most important features:")
    for i, item in enumerate(importance_summary[:5]):
        print(f"   {i+1}. {item['feature']}: {item['importance']:.4f}")

    # Step 6: Display final training summary
    print("\n🎉 AI Model Training Complete!")
    print(f"   - Model saved: {model_save_path}")
    print(f"   - Performance graph: {graph_save_path}")
    print(f"   - Predictions within 1 point: {training_results['metrics']['accuracy_within_1_point']}%")
    print(f"   - Model meets requirements: {'✅ YES' if training_results['metrics']['overall_pass'] else '❌ NO'}")
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
    Entry point for model training script.
    
    Creates model directory if needed, executes training pipeline,
    and reports success or failure.
    """
    # Create model output directory if it doesn't exist
    ensure_directory_exists("model")

    # Execute full training pipeline
    results = train_complete_model()

    if results:
        print("\n Training pipeline completed successfully!")
        print("You can now use the trained model for risk predictions.")
    else:
        print("\n Training pipeline failed. Check the error messages above.")

if __name__ == "__main__":
    main()