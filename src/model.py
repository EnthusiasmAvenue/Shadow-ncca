import pickle
import os
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

class ScorePredictor:
    def __init__(self, model_path='model.pkl'):
        # Using Random Forest for better non-linear modeling
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        self.feature_names = None
        self.error_sigma = 10.0
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            self.load_model()

    def train(self, X, y):
        """
        Trains the model using GridSearchCV for hyperparameter tuning.
        """
        if len(X) < 20:
            print(f"Small dataset ({len(X)}). Using default parameters.")
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.feature_names = list(X.columns)
            self.save_model()
            return 0.0

        print(f"Tuning Random Forest on {len(X)} samples...")
        X_scaled = self.scaler.fit_transform(X)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5],
            'bootstrap': [True, False]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.feature_names = list(X.columns)
        
        mae = abs(grid_search.best_score_)
        print(f"Model tuned. Best MAE: {mae:.2f}")
        print(f"Best Params: {grid_search.best_params_}")
        
        # Calculate robust sigma from residuals
        predictions = self.model.predict(X_scaled)
        resid = y - predictions
        self.error_sigma = float(np.percentile(np.abs(resid), 90))
        
        self.save_model()
        return mae

    def predict(self, X):
        """
        Predicts total score for input features X.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]
        
        # Scale features using the same scaler from training
        try:
            X_scaled = self.scaler.transform(X)
        except Exception:
            # Fallback if scaler is not fitted
            print("Warning: Scaler not fitted, using raw features.")
            X_scaled = X
            
        return self.model.predict(X_scaled)

    def predict_game(self, game_features, line, home_team="Home", away_team="Away"):
        """
        Predicts the total score for a single game and returns decision details.
        """
        if not self.is_trained:
            return {
                'predicted_total': line,
                'decision': 'No Model',
                'confidence': '0%',
                'diff': 0,
                'explanation': "Model not yet trained."
            }

        predicted_total = float(self.predict(game_features)[0])
        diff = predicted_total - line
        
        # Calculate confidence using CDF of normal distribution
        sigma = max(self.error_sigma, 1.0)
        prob_over = 0.5 * (1.0 + math.erf(diff / (math.sqrt(2) * sigma)))
        
        if diff > 0:
            decision = "Over"
            confidence = prob_over
        else:
            decision = "Under"
            confidence = 1.0 - prob_over
            
        # --- EXPLANATION LOGIC ---
        explanation_parts = []
        try:
            # Check pace
            pace = game_features['pace_avg'].iloc[0]
            if pace > 155: 
                explanation_parts.append(f"Fast-paced matchup expected ({pace:.1f} avg pace).")
            elif pace < 140: 
                explanation_parts.append(f"Slow, defensive grind expected ({pace:.1f} avg pace).")
            
            # Check edge
            home_edge = game_features['off_edge_home'].iloc[0]
            away_edge = game_features['off_edge_away'].iloc[0]
            if home_edge > 12: 
                explanation_parts.append(f"{home_team} offense has a massive mismatch against {away_team} defense.")
            elif home_edge > 7:
                explanation_parts.append(f"{home_team} has a solid offensive edge.")
                
            if away_edge > 12: 
                explanation_parts.append(f"{away_team} offense looks dominant in this matchup.")
            elif away_edge > 7:
                explanation_parts.append(f"{away_team} has a clear offensive advantage.")
            
            # Check SoS (Strength of Schedule)
            h_sos_off = game_features['home_sos_off'].iloc[0]
            a_sos_off = game_features['away_sos_off'].iloc[0]
            if h_sos_off > 80: 
                explanation_parts.append(f"{home_team} is battle-tested against elite defenses.")
            if a_sos_off > 80:
                explanation_parts.append(f"{away_team} has faced high-level defensive pressure.")

            # Check Rest/B2B
            h_rest = game_features['home_rest'].iloc[0]
            a_rest = game_features['away_rest'].iloc[0]
            if h_rest <= 1: explanation_parts.append(f"{home_team} is on short rest.")
            if a_rest <= 1: explanation_parts.append(f"{away_team} is on short rest.")

            # Check Scenario Score (Meta-learning)
            scenario = game_features.get('scenario_score', pd.Series([0.5])).iloc[0]
            if scenario > 0.65:
                explanation_parts.append("Historical data suggests this is a high-probability scenario.")
            elif scenario < 0.35:
                explanation_parts.append("Caution: Model has historically struggled in this specific pace/line range.")
            
        except Exception as e:
            print(f"Explanation error: {e}")
            pass
            
        if not explanation_parts:
            explanation_parts.append("Balanced matchup with slight edge found in statistical trends.")

        return {
            'predicted_total': round(predicted_total, 1),
            'decision': decision,
            'confidence': f"{confidence*100:.1f}%",
            'diff': round(diff, 1),
            'prob_over': round(prob_over, 3),
            'explanation': " ".join(explanation_parts)
        }

    def save_model(self):
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'error_sigma': self.error_sigma
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                obj = pickle.load(f)
                if isinstance(obj, dict):
                    self.model = obj.get('model', self.model)
                    self.scaler = obj.get('scaler', self.scaler)
                    self.feature_names = obj.get('feature_names', None)
                    self.error_sigma = obj.get('error_sigma', 12.0)
                else:
                    self.model = obj
            self.is_trained = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
