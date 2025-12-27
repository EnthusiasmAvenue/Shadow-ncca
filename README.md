# Shadow NCAA Prediction Bot

This project is an AI-powered bot that predicts NCAA game results (Over/Under) and self-upgrades based on historical data.

## Features

- **Predict Upcoming Games**: Generates Over/Under predictions with confidence levels.
- **Self-Upgrading**: Retrains the machine learning model on new data to improve accuracy over time.
- **Real Data Integration**: Supports fetching real-time odds from **The-Odds-Api**.
- **Historical Data Simulation**: Includes a mock data loader for initial training and simulation when API limits are reached.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API (Optional)**:
    - Get a free API key from [The-Odds-Api](https://the-odds-api.com/).
    - Create a `.env` file in the project root:
      ```
      ODDS_API_KEY=your_api_key_here
      ```
    - If no key is provided, the bot will default to Mock Data mode.

3.  **Run the Application**:
    ```bash
    python src/main.py
    ```

## Usage

When you run the bot, you will see a menu:

1.  **Predict Upcoming Games**: Fetches today's games (Real or Mock) and prints predictions.
2.  **Retrain/Upgrade Model**: Simulates fetching new game results and retrains the model.
    - *Note: On the free API tier, historical data is simulated for training purposes.*
3.  **View Past Predictions**: (Planned feature)

## Project Structure

- `src/main.py`: Entry point and CLI interface.
- `src/model.py`: Random Forest model logic.
- `src/api_loader.py`: Handles fetching real data from The-Odds-Api.
- `src/data_loader.py`: Handles mock data generation.
- `src/database.py`: Database schema.
