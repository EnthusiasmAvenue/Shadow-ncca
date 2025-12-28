import sys
import os
from dotenv import load_dotenv
import pandas as pd
from database import init_db, Game, Prediction
from data_loader import DataLoader
from api_loader import APIDataLoader
from model import ScorePredictor
from datetime import datetime

# Load environment variables
load_dotenv()

def main():
    print("Initializing NCAA Prediction Bot...")
    
    # 1. Setup Database
    session = init_db()
    
    # 2. Determine Data Source
    api_key = os.getenv('ODDS_API_KEY')
    use_api = False
    
    if api_key:
        print("API Key found. Using Real Data from The-Odds-Api.")
        loader = APIDataLoader(api_key)
        # We also need the Mock loader for fallback/historical training if DB is empty
        mock_loader = DataLoader(session)
        use_api = True
    else:
        print("No API Key found. Using Mock Data Simulation.")
        print("(To use real data, add ODDS_API_KEY to a .env file)")
        loader = DataLoader(session)
        mock_loader = loader

    predictor = ScorePredictor()
    
    # 3. Check for initial training
    if not predictor.is_trained:
        print("Model not trained. Fetching historical data for initial training...")
        # API doesn't support history well on free tier, so use Mock data for cold start
        if use_api:
            print("Using Mock Data for initial training (API history is restricted)...")
            history_df = mock_loader.fetch_historical_data(num_games=500)
        else:
            history_df = loader.fetch_historical_data(num_games=500)
            
        X, y = loader.prepare_features(history_df)
        predictor.train(X, y)
    
    while True:
        print("\n--- NCAA Prediction Bot Menu ---")
        print("1. Predict Upcoming Games")
        print("2. Retrain/Upgrade Model (Fetch new history)")
        print("3. View Past Predictions")
        print("4. Exit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            print("\nFetching upcoming games...")
            upcoming_df = loader.fetch_upcoming_games()
            
            if upcoming_df.empty:
                print("No upcoming games found (or API limit reached).")
                continue
            
            print(f"\nFound {len(upcoming_df)} games. Generating predictions...\n")
            print(f"{'Matchup':<40} | {'Line':<6} | {'Pred Total':<10} | {'Pick':<6} | {'Conf':<6}")
            print("-" * 85)
            
            for _, row in upcoming_df.iterrows():
                # Prepare single row feature
                features_df = pd.DataFrame([row]) 
                X_pred = loader.prepare_features(features_df)
                
                result = predictor.predict_game(X_pred, row['over_under_line'])
                
                matchup = f"{row['team_away']} @ {row['team_home']}"
                print(f"{matchup:<40} | {row['over_under_line']:<6} | {result['predicted_total']:<10} | {result['decision']:<6} | {result['confidence']:<6}")
                
        elif choice == '2':
            print("\nSelf-Upgrading: Fetching latest game results to improve model...")
            
            if use_api:
                # In a real scenario, this would query the DB for resolved games
                # or use a 'scores' endpoint if available/paid.
                # For now, we fallback to mock simulation for the "learning" process
                # unless we have a database of real past results.
                print("Note: Real-time self-upgrading requires paid API history or collected DB data.")
                print("Simulating upgrade with mock data for demonstration...")
                new_history = mock_loader.fetch_historical_data(num_games=100)
            else:
                new_history = loader.fetch_historical_data(num_games=100)

            X_new, y_new = loader.prepare_features(new_history)
            mae = predictor.train(X_new, y_new)
            print(f"Model upgraded! New Mean Absolute Error: {mae:.2f}")
            
        elif choice == '3':
            print("\nFeature not fully implemented in this MVP (requires persistence of predictions).")
            
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
