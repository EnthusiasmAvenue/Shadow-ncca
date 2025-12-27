import requests
import pandas as pd
from datetime import datetime
import os

class APIDataLoader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.the-odds-api.com/v4/sports'
        self.sport_key = 'basketball_ncaab'

    def fetch_upcoming_games(self):
        """
        Fetches upcoming NCAA basketball games and odds from The-Odds-Api.
        """
        if not self.api_key:
            raise ValueError("API Key is missing.")

        # Request odds for upcoming games
        # We focus on the 'totals' market (Over/Under)
        url = f"{self.base_url}/{self.sport_key}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            print(f"Requesting odds from: {url}")
            response = requests.get(url, params=params)
            print(f"API Response Status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            print(f"API returned {len(data)} games")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return pd.DataFrame()

        games_list = []
        for game in data:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time')
            over_under_line = None
            points = []
            bookmakers = game.get('bookmakers', [])
            for bk in bookmakers or []:
                markets = bk.get('markets', [])
                for m in markets:
                    if m.get('key') == 'totals':
                        outcomes = m.get('outcomes', [])
                        if outcomes:
                            pt = outcomes[0].get('point')
                            if pt is not None:
                                try:
                                    points.append(float(pt))
                                except Exception:
                                    pass
            if bookmakers:
                market_first = next((m for m in bookmakers[0].get('markets', []) if m.get('key') == 'totals'), None)
                if market_first and market_first.get('outcomes'):
                    over_under_line = market_first['outcomes'][0].get('point')
            consensus_line = None
            line_std = None
            bk_count = len(bookmakers or [])
            if points:
                import numpy as np
                consensus_line = float(np.mean(points))
                line_std = float(np.std(points))
            if over_under_line is not None:
                row = {
                    'date': commence_time,
                    'team_home': home_team,
                    'team_away': away_team,
                    'over_under_line': float(over_under_line)
                }
                if consensus_line is not None:
                    row['consensus_line'] = consensus_line
                if line_std is not None:
                    row['line_std'] = line_std
                row['bookmakers_count'] = bk_count
                games_list.append(row)

        return pd.DataFrame(games_list)

    def fetch_historical_data(self, num_games=100):
        """
        The free tier of The-Odds-Api does NOT support historical data easily (requires paid plan).
        So we will return an empty DataFrame or fallback to mock data if needed.
        For now, we warn the user.
        """
        print("Warning: Historical data fetching is not supported on the free tier of this API.")
        print("Returning empty dataset. Please rely on collected data in database or mock data for initial training.")
        return pd.DataFrame()

    def prepare_features(self, df):
        """
        Same feature preparation as the base loader.
        """
        if df.empty:
            return pd.DataFrame()
            
        X = df[['over_under_line']].copy()
        X['implied_total'] = X['over_under_line']
        
        if 'total_score' in df.columns:
            y = df['total_score']
            return X, y
        else:
            return X

    def fetch_recent_scores(self, days=3):
        if not self.api_key:
            return pd.DataFrame()
        url = f"{self.base_url}/{self.sport_key}/scores"
        params = {
            'apiKey': self.api_key,
            'daysFrom': days,
            'dateFormat': 'iso'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching scores from API: {e}")
            return pd.DataFrame()
        rows = []
        for game in data:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time')
            home_score = None
            away_score = None
            if 'scores' in game and isinstance(game['scores'], list):
                for s in game['scores']:
                    if s.get('name') == home_team:
                        home_score = s.get('score')
                    if s.get('name') == away_team:
                        away_score = s.get('score')
            else:
                home_score = game.get('home_score')
                away_score = game.get('away_score')
            if home_team and away_team:
                rows.append({
                    'date': commence_time,
                    'team_home': home_team,
                    'team_away': away_team,
                    'score_home': int(home_score) if home_score is not None else None,
                    'score_away': int(away_score) if away_score is not None else None,
                    'completed': game.get('completed', False)
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df['total_score'] = df.apply(
                lambda r: (r['score_home'] + r['score_away']) if r['score_home'] is not None and r['score_away'] is not None else None,
                axis=1
            )
        return df
