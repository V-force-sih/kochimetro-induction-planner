# ml_trainer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

class OptimizationTrainer:
    def __init__(self):
        self.model = None
        self.features = [
            'required_trains', 'available_trains', 'unfit_trains', 
            'critical_jobs', 'branding_priority_avg', 'mileage_avg'
        ]
        self.targets = ['branding_weight', 'mileage_weight', 'stabling_weight']
        
    def load_data(self, data_path):
        """Load historical optimization data"""
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
            # Filter only successful optimizations for training
            self.training_data = self.data[self.data['success'] == True]
            return True
        return False
    
    def prepare_data(self):
        """Prepare data for training"""
        if len(self.training_data) < 10:  # Need minimum data points
            return False
            
        X = self.training_data[self.features]
        y = self.training_data[self.targets]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return True
    
    def train_model(self):
        """Train the Random Forest model"""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"Model trained. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        return test_score > 0.6  # Return True if model is reasonably accurate
    
    def predict_weights(self, features_dict):
        """Predict optimal weights for a new scenario"""
        if self.model is None:
            return None
            
        # Convert features to DataFrame
        features_df = pd.DataFrame([features_dict])
        # Ensure all required features are present
        for feature in self.features:
            if feature not in features_df.columns:
                features_df[feature] = 0
                
        features_df = features_df[self.features]  # Reorder columns
        
        # Predict weights
        weights = self.model.predict(features_df)[0]
        
        return {
            'branding_weight': max(0.1, min(weights[0], 2.0)),
            'mileage_weight': max(0.00001, min(weights[1], 0.001)),
            'stabling_weight': max(0.001, min(weights[2], 0.1))
        }
    
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model:
            joblib.dump(self.model, model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            return True
        return False

# Example usage
if __name__ == "__main__":
    trainer = OptimizationTrainer()
    if trainer.load_data("data/optimization_history.csv"):
        if trainer.prepare_data():
            trainer.train_model()
            trainer.save_model("models/optimization_model.joblib")