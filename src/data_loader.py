import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from database import Game, Prediction

class DataLoader:
    def __init__(self, db_session):
        self.session = db_session
        self.team_stats = {}      # This will hold the "Blended" or "Primary" stats
        self.csv_team_stats = {}  # Specifically for the study data from CSV
        self.db_team_stats = {}   # Specifically for stats calculated from DB history
        self.global_stats = {'off': 75.0, 'def': 72.0, 'pace': 147.0}
        self.team_bias = {}
        self.aliases = {}
    
    def load_team_stats_from_csv(self, path):
        def try_parse(df):
            cols = set(df.columns)
            if {'team', 'off', 'def', 'pace'}.issubset(cols):
                print(f"Parsed CSV using schema: team/off/def/pace")
                self.csv_team_stats = {r['team']: {'off': float(r['off']), 'def': float(r['def']), 'pace': float(r['pace'])} for _, r in df.iterrows()}
                self.team_stats = self.csv_team_stats.copy() # Initial load
                self.global_stats = {
                    'off': float(df['off'].mean()),
                    'def': float(df['def'].mean()),
                    'pace': float(df['pace'].mean())
                }
                return True
            if {'School', 'G', 'Tm.', 'Opp.'}.issubset(cols) or {'Team', 'G', 'Tm.', 'Opp.'}.issubset(cols):
                print(f"Parsed CSV using schema: SportsReference (School/Tm./Opp.)")
                name_col = 'School' if 'School' in cols else 'Team'
                g = pd.to_numeric(df['G'], errors='coerce')
                tm = pd.to_numeric(df['Tm.'], errors='coerce')
                opp = pd.to_numeric(df['Opp.'], errors='coerce')
                off = tm / g
                deff = opp / g
                if 'Pace' in cols:
                    pace_series = pd.to_numeric(df['Pace'], errors='coerce')
                else:
                    pace_series = off + deff
                names = df[name_col].astype(str)
                valid = (~off.isna()) & (~deff.isna()) & (~pace_series.isna())
                names = names[valid]
                off = off[valid]
                deff = deff[valid]
                pace_series = pace_series[valid]
                self.csv_team_stats = {self.normalize_team_name(n): {'off': float(o), 'def': float(d), 'pace': float(p)} for n, o, d, p in zip(names, off, deff, pace_series)}
                self.team_stats = self.csv_team_stats.copy() # Initial load
                self.global_stats = {
                    'off': float(off.mean()) if len(off) else self.global_stats['off'],
                    'def': float(deff.mean()) if len(deff) else self.global_stats['def'],
                    'pace': float(pace_series.mean()) if len(pace_series) else self.global_stats['pace']
                }
                return True
            if {'team_home', 'team_away', 'score_home', 'score_away'}.issubset(cols):
                print(f"Parsed CSV using schema: Game History (home/away/score)")
                self.build_team_stats(df)
                return True
            print(f"Failed to parse CSV. Columns found: {cols}")
            return False

        try:
            print(f"Attempting to read CSV from: {path}")
            df = pd.read_csv(path)
            if try_parse(df):
                return True
            print("Trying with header=1 (skipping first row)...")
            df2 = pd.read_csv(path, header=1)
            if try_parse(df2):
                return True
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
        return False
    
    def normalize_team_name(self, name):
        return self.aliases.get(name, name)

    def fetch_historical_data(self, num_games=100):
        """
        Simulates fetching historical NCAA game data.
        In a real app, this would call an API (e.g., SportsReference, ESPN).
        """
        data = []
        teams = ['Duke', 'UNC', 'Kansas', 'Kentucky', 'UCLA', 'Villanova', 'Gonzaga', 'Baylor']
        
        start_date = datetime.now() - timedelta(days=365)
        
        for _ in range(num_games):
            home, away = random.sample(teams, 2)
            # Simulate scores roughly around NCAA averages (e.g., 70-80 points)
            score_home = int(np.random.normal(75, 10))
            score_away = int(np.random.normal(72, 10))
            total_score = score_home + score_away
            
            # Simulate a bookmaker line that is somewhat accurate but not perfect
            over_under_line = total_score + np.random.normal(0, 5)
            over_under_line = round(over_under_line * 2) / 2  # Round to nearest 0.5
            
            result = 'Push'
            if total_score > over_under_line:
                result = 'Over'
            elif total_score < over_under_line:
                result = 'Under'
                
            game_date = start_date + timedelta(days=random.randint(0, 300))
            
            data.append({
                'date': game_date,
                'team_home': home,
                'team_away': away,
                'score_home': score_home,
                'score_away': score_away,
                'total_score': total_score,
                'over_under_line': over_under_line,
                'result': result
            })
            
        return pd.DataFrame(data)

    def build_team_stats(self, df):
        if df.empty:
            return
        if not {'team_home', 'team_away', 'score_home', 'score_away'}.issubset(df.columns):
            return
        
        # 1. Calculate basic averages
        home = df[['team_home', 'score_home', 'score_away']].copy()
        home.columns = ['team', 'pf', 'pa']
        away = df[['team_away', 'score_away', 'score_home']].copy()
        away.columns = ['team', 'pf', 'pa']
        both = pd.concat([home, away], ignore_index=True)
        both['total'] = both['pf'] + both['pa']
        
        team_avg = both.groupby('team').agg(
            off=('pf', 'mean'), 
            defn=('pa', 'mean'), 
            pace=('total', 'mean'),
            count=('team', 'count')
        ).reset_index()
        
        # 2. Calculate Strength of Schedule (SoS)
        # SoS Offense = Average 'defn' of opponents
        # SoS Defense = Average 'off' of opponents
        team_ratings = {r['team']: {'off': r['off'], 'defn': r['defn']} for _, r in team_avg.iterrows()}
        
        def get_opp_stats(row, is_home):
            opp = row['team_away'] if is_home else row['team_home']
            return team_ratings.get(opp, {'off': self.global_stats['off'], 'defn': self.global_stats['def']})

        df['home_opp_off'] = df.apply(lambda r: get_opp_stats(r, True)['off'], axis=1)
        df['home_opp_def'] = df.apply(lambda r: get_opp_stats(r, True)['defn'], axis=1)
        df['away_opp_off'] = df.apply(lambda r: get_opp_stats(r, False)['off'], axis=1)
        df['away_opp_def'] = df.apply(lambda r: get_opp_stats(r, False)['defn'], axis=1)

        home_sos = df.groupby('team_home').agg(sos_off=('home_opp_off', 'mean'), sos_def=('home_opp_def', 'mean')).reset_index().rename(columns={'team_home': 'team'})
        away_sos = df.groupby('team_away').agg(sos_off=('away_opp_off', 'mean'), sos_def=('away_opp_def', 'mean')).reset_index().rename(columns={'team_away': 'team'})
        
        sos_combined = pd.concat([home_sos, away_sos]).groupby('team').mean().reset_index()
        
        # Merge stats and SOS
        final_stats = team_avg.merge(sos_combined, on='team', how='left')
        
        # Save historical stats from database separately
        self.db_team_stats = {r['team']: {
            'off': float(r['off']), 
            'def': float(r['defn']), 
            'pace': float(r['pace']),
            'sos_off': float(r['sos_off']) if not np.isnan(r['sos_off']) else self.global_stats['off'],
            'sos_def': float(r['sos_def']) if not np.isnan(r['sos_def']) else self.global_stats['def']
        } for _, r in final_stats.iterrows()}
        
        # Blend CSV and DB stats into the primary team_stats
        all_teams = set(self.csv_team_stats.keys()) | set(self.db_team_stats.keys())
        for team in all_teams:
            csv = self.csv_team_stats.get(team)
            db = self.db_team_stats.get(team)
            
            if csv and db:
                self.team_stats[team] = {
                    'off': (csv['off'] * 0.6) + (db['off'] * 0.4),
                    'def': (csv['def'] * 0.6) + (db['def'] * 0.4),
                    'pace': (csv['pace'] * 0.6) + (db['pace'] * 0.4),
                    'sos_off': db['sos_off'],
                    'sos_def': db['sos_def']
                }
            elif csv:
                self.team_stats[team] = {**csv, 'sos_off': self.global_stats['off'], 'sos_def': self.global_stats['def']}
            elif db:
                self.team_stats[team] = db

        self.global_stats = {
            'off': float(both['pf'].mean()),
            'def': float(both['pa'].mean()),
            'pace': float(both['total'].mean())
        }
    
    def fetch_db_training_data(self, min_rows=10):
        games = self.session.query(Game).filter(Game.total_score != None).all()
        rows = []
        for g in games:
            rows.append({
                'date': g.date,
                'team_home': g.team_home,
                'team_away': g.team_away,
                'over_under_line': g.over_under_line,
                'total_score': g.total_score
            })
        df = pd.DataFrame(rows)
        if len(df) >= min_rows:
            return df
        return pd.DataFrame()
    
    def fetch_team_recent(self, team, limit=10):
        q_home = self.session.query(Game).filter(Game.team_home == team, Game.total_score != None).order_by(Game.date.desc()).limit(limit).all()
        q_away = self.session.query(Game).filter(Game.team_away == team, Game.total_score != None).order_by(Game.date.desc()).limit(limit).all()
        rows = []
        for g in q_home:
            rows.append({'pf': g.score_home or 0, 'pa': g.score_away or 0, 'date': g.date})
        for g in q_away:
            rows.append({'pf': g.score_away or 0, 'pa': g.score_home or 0, 'date': g.date})
        if not rows:
            return {
                'pf5': self.global_stats['off'], 'pa5': self.global_stats['def'], 'pace5': self.global_stats['pace'],
                'pf10': self.global_stats['off'], 'pa10': self.global_stats['def'], 'pace10': self.global_stats['pace'],
                'days_since_last': 7, 'b2b': 0
            }
        df = pd.DataFrame(rows).sort_values('date', ascending=False)
        n5 = df.head(5)
        n10 = df.head(10)
        pf5 = float(n5['pf'].mean()) if len(n5) else self.global_stats['off']
        pa5 = float(n5['pa'].mean()) if len(n5) else self.global_stats['def']
        pace5 = float((n5['pf'] + n5['pa']).mean()) if len(n5) else self.global_stats['pace']
        last_date = df.iloc[0]['date']
        from datetime import datetime
        now = datetime.now(last_date.tzinfo) if hasattr(last_date, 'tzinfo') else datetime.now()
        days_since = max(int((now - last_date).days), 0)
        b2b = 1 if days_since <= 1 else 0
        return {'pf5': pf5, 'pa5': pa5, 'pace5': pace5, 'pf10': float(n10['pf'].mean()) if len(n10) else pf5, 'pa10': float(n10['pa'].mean()) if len(n10) else pa5, 'pace10': float((n10['pf'] + n10['pa']).mean()) if len(n10) else pace5, 'days_since_last': days_since, 'b2b': b2b}
    
    def build_team_bias_from_db(self, decay=0.03):
        preds = (
            self.session.query(Prediction)
            .join(Game)
            .filter(Game.total_score != None)
            .all()
        )
        if not preds:
            self.team_bias = {}
            return
        agg = {}
        from datetime import datetime
        now = datetime.now()
        for p in preds:
            g = p.game
            if not g or g.total_score is None:
                continue
            err = float(p.predicted_total) - float(g.total_score)
            try:
                days = max((now - g.date).days, 0)
            except Exception:
                days = 0
            import math
            w = math.exp(-decay * days)
            for team in [g.team_home, g.team_away]:
                if team not in agg:
                    agg[team] = {'w_sum': 0.0, 'sum_abs': 0.0, 'sum': 0.0, 'sum_sq': 0.0}
                agg[team]['w_sum'] += w
                agg[team]['sum_abs'] += w * abs(err)
                agg[team]['sum'] += w * err
                agg[team]['sum_sq'] += w * (err * err)
        self.team_bias = {}
        for team, v in agg.items():
            c = max(v['w_sum'], 1e-6)
            self.team_bias[team] = {
                'mae': v['sum_abs'] / c,
                'bias': v['sum'] / c,
                'sigma': float(max(0, (v['sum_sq'] / c) - (v['sum'] / c) ** 2) ** 0.5) if c > 0 else 0.0
            }

    def fetch_upcoming_games(self, num_games=10):
        """
        Simulates fetching upcoming games schedule using UTC dates.
        """
        data = []
        teams = ['Duke', 'UNC', 'Kansas', 'Kentucky', 'UCLA', 'Villanova', 'Gonzaga', 'Baylor', 'Arizona', 'Purdue']
        
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("UTC"))
        
        for i in range(num_games):
            home, away = random.sample(teams, 2)
            # Bookmaker line estimation
            over_under_line = 145.5 + np.random.normal(0, 5) 
            over_under_line = round(over_under_line * 2) / 2
            
            # Mix of games: some today, some tomorrow, some day after
            days_out = (i % 3) 
            game_time = today + timedelta(days=days_out, hours=random.randint(2, 20))
            
            data.append({
                'date': game_time,
                'team_home': home,
                'team_away': away,
                'over_under_line': over_under_line
            })
            
        return pd.DataFrame(data)

    def analyze_prediction_performance(self):
        """
        Analyzes past predictions to find patterns in wins and losses.
        Identifies 'what to look for' (high accuracy scenarios) 
        and 'what to avoid' (high error scenarios).
        """
        preds = (
            self.session.query(Prediction)
            .join(Game)
            .filter(Game.total_score != None)
            .all()
        )
        if not preds:
            self.performance_insights = {
                'good_scenarios': [],
                'bad_scenarios': [],
                'avg_accuracy': 0.0
            }
            return

        perf_data = []
        for p in preds:
            g = p.game
            error = p.predicted_total - g.total_score
            is_win = False
            if p.predicted_class == "Over" and g.total_score > g.over_under_line:
                is_win = True
            elif p.predicted_class == "Under" and g.total_score < g.over_under_line:
                is_win = True
            
            perf_data.append({
                'line': g.over_under_line,
                'pred': p.predicted_total,
                'diff': p.predicted_total - g.over_under_line,
                'abs_error': abs(error),
                'is_win': 1 if is_win else 0,
                'pace': (self.team_stats.get(g.team_home, self.global_stats)['pace'] + 
                         self.team_stats.get(g.team_away, self.global_stats)['pace']) / 2
            })
        
        df = pd.DataFrame(perf_data)
        insights = {
            'avg_accuracy': df['is_win'].mean(),
            'mae': df['abs_error'].mean(),
            'bad_scenarios': [],
            'good_scenarios': []
        }

        # Analyze by Pace (Look for high-error pace ranges)
        if len(df) >= 10:
            pace_bins = pd.cut(df['pace'], bins=3)
            pace_perf = df.groupby(pace_bins, observed=True)['is_win'].mean()
            for interval, win_rate in pace_perf.items():
                if win_rate < 0.45:
                    insights['bad_scenarios'].append({'type': 'pace', 'range': (interval.left, interval.right), 'win_rate': win_rate})
                elif win_rate > 0.60:
                    insights['good_scenarios'].append({'type': 'pace', 'range': (interval.left, interval.right), 'win_rate': win_rate})

            # Analyze by Line (Look for high-error line ranges)
            line_bins = pd.cut(df['line'], bins=3)
            line_perf = df.groupby(line_bins, observed=True)['is_win'].mean()
            for interval, win_rate in line_perf.items():
                if win_rate < 0.45:
                    insights['bad_scenarios'].append({'type': 'line', 'range': (interval.left, interval.right), 'win_rate': win_rate})
                elif win_rate > 0.60:
                    insights['good_scenarios'].append({'type': 'line', 'range': (interval.left, interval.right), 'win_rate': win_rate})

        self.performance_insights = insights
        return insights

    def prepare_features(self, df):
        """
        Converts raw game data into features for the model.
        Uses team stats from CSV and historical database analysis.
        """
        X = df[['over_under_line']].copy()
        X['implied_total'] = X['over_under_line']
        if 'team_home' in df.columns and 'team_away' in df.columns:
            def get_stats(team):
                return self.team_stats.get(team, self.global_stats)
            def get_bias(team):
                return self.team_bias.get(team, {'mae': 10.0, 'bias': 0.0, 'sigma': 10.0})
            def get_recent(team):
                return self.fetch_team_recent(team, limit=10)
            
            home_stats = df['team_home'].apply(get_stats)
            away_stats = df['team_away'].apply(get_stats)
            home_bias = df['team_home'].apply(get_bias)
            away_bias = df['team_away'].apply(get_bias)
            home_recent = df['team_home'].apply(get_recent)
            away_recent = df['team_away'].apply(get_recent)

            X['home_off'] = home_stats.apply(lambda s: s['off'])
            X['home_def'] = home_stats.apply(lambda s: s['def'])
            X['home_pace'] = home_stats.apply(lambda s: s['pace'])
            X['away_off'] = away_stats.apply(lambda s: s['off'])
            X['away_def'] = away_stats.apply(lambda s: s['def'])
            X['away_pace'] = away_stats.apply(lambda s: s['pace'])
            
            # Historical Research Features
            X['home_mae'] = home_bias.apply(lambda b: b['mae'])
            X['home_bias'] = home_bias.apply(lambda b: b['bias'])
            X['away_mae'] = away_bias.apply(lambda b: b['mae'])
            X['away_bias'] = away_bias.apply(lambda b: b['bias'])

            # Recent Form Features
            X['home_pf5'] = home_recent.apply(lambda r: r['pf5'])
            X['home_pa5'] = home_recent.apply(lambda r: r['pa5'])
            X['home_pace5'] = home_recent.apply(lambda r: r['pace5'])
            X['away_pf5'] = away_recent.apply(lambda r: r['pf5'])
            X['away_pa5'] = away_recent.apply(lambda r: r['pa5'])
            X['away_pace5'] = away_recent.apply(lambda r: r['pace5'])
            X['home_days_rest'] = home_recent.apply(lambda r: r['days_since_last'])
            X['away_days_rest'] = away_recent.apply(lambda r: r['days_since_last'])
            
            # New Meta-Learning Feature: Performance Context
            # Is this a "high-risk" or "high-reward" scenario based on past wins/losses?
            def get_scenario_score(row):
                if not hasattr(self, 'performance_insights'):
                    return 0.5
                
                score = 0.5
                pace = (row['home_pace'] + row['away_pace']) / 2
                line = row['over_under_line']
                
                for s in self.performance_insights.get('bad_scenarios', []):
                    if s['type'] == 'pace' and s['range'][0] <= pace <= s['range'][1]:
                        score -= 0.1
                    if s['type'] == 'line' and s['range'][0] <= line <= s['range'][1]:
                        score -= 0.1
                        
                for s in self.performance_insights.get('good_scenarios', []):
                    if s['type'] == 'pace' and s['range'][0] <= pace <= s['range'][1]:
                        score += 0.1
                    if s['type'] == 'line' and s['range'][0] <= line <= s['range'][1]:
                        score += 0.1
                return max(0.1, min(0.9, score))

            # Apply scenario scoring to each row
            X['scenario_score'] = X.apply(get_scenario_score, axis=1)

            X['home_sigma'] = home_bias.apply(lambda b: b.get('sigma', 10.0))
            X['away_sigma'] = away_bias.apply(lambda b: b.get('sigma', 10.0))
            
            # Recent Form (Last 10 Games)
            X['home_pf10'] = home_recent.apply(lambda s: s.get('pf10', s.get('pf_avg', 0.0)))
            X['home_pa10'] = home_recent.apply(lambda s: s.get('pa10', s.get('pa_avg', 0.0)))
            X['home_pace10'] = home_recent.apply(lambda s: s.get('pace10', s.get('pace_avg', 0.0)))
            X['away_pf10'] = away_recent.apply(lambda s: s.get('pf10', s.get('pf_avg', 0.0)))
            X['away_pa10'] = away_recent.apply(lambda s: s.get('pa10', s.get('pa_avg', 0.0)))
            X['away_pace10'] = away_recent.apply(lambda s: s.get('pace10', s.get('pace_avg', 0.0)))
            
            # Rest and Fatigue
            X['home_rest'] = home_recent.apply(lambda s: s.get('days_since_last', 4))
            X['away_rest'] = away_recent.apply(lambda s: s.get('days_since_last', 4))
            X['home_b2b'] = home_recent.apply(lambda s: s.get('b2b', 0))
            X['away_b2b'] = away_recent.apply(lambda s: s.get('b2b', 0))
            
            # --- DYNAMIC BLENDING ---
            # As the database grows, we weight recent form more heavily.
            db_game_count = self.session.query(Game).filter(Game.total_score != None).count()
            # weight_db starts at 0.1 and grows to 0.5 as we reach 200 games
            weight_db = min(0.5, 0.1 + (db_game_count / 400.0))
            weight_csv = 1.0 - weight_db
            
            X['home_off_blended'] = (X['home_off'] * weight_csv) + (X['home_pf10'] * weight_db)
            X['home_def_blended'] = (X['home_def'] * weight_csv) + (X['home_pa10'] * weight_db)
            X['away_off_blended'] = (X['away_off'] * weight_csv) + (X['away_pf10'] * weight_db)
            X['away_def_blended'] = (X['away_def'] * weight_csv) + (X['away_pa10'] * weight_db)

            # Strength of Schedule Features
            X['home_sos_off'] = home_stats.apply(lambda s: s.get('sos_off', self.global_stats['off']))
            X['home_sos_def'] = home_stats.apply(lambda s: s.get('sos_def', self.global_stats['def']))
            X['away_sos_off'] = away_stats.apply(lambda s: s.get('sos_off', self.global_stats['off']))
            X['away_sos_def'] = away_stats.apply(lambda s: s.get('sos_def', self.global_stats['def']))

            # --- MATCHUP FEATURES (ENHANCED) ---
            X['pace_avg'] = (X['home_pace'] + X['away_pace']) / 2
            X['off_edge_home'] = X['home_off_blended'] - X['away_def_blended']
            X['off_edge_away'] = X['away_off_blended'] - X['home_def_blended']
            X['total_offense'] = X['home_off_blended'] + X['away_off_blended']
            X['total_defense'] = X['home_def_blended'] + X['away_def_blended']
            X['efficiency_gap'] = abs(X['off_edge_home'] - X['off_edge_away'])
            X['pace_clash'] = abs(X['home_pace'] - X['away_pace'])
        
        # Ensure all columns exist for model
        expected = [
            'over_under_line', 'implied_total', 'home_off', 'home_def', 'home_pace',
            'away_off', 'away_def', 'away_pace', 'home_mae', 'home_bias', 'home_sigma',
            'away_mae', 'away_bias', 'away_sigma', 'home_pf10', 'home_pa10', 'home_pace10',
            'away_pf10', 'away_pa10', 'away_pace10', 'home_rest', 'away_rest', 'home_b2b',
            'away_b2b', 'home_off_blended', 'home_def_blended', 'away_off_blended', 
            'away_def_blended', 'home_pf5', 'away_pf5', 'scenario_score',
            'pace_avg', 'off_edge_home', 'off_edge_away', 'total_offense', 
            'total_defense', 'efficiency_gap', 'pace_clash',
            'home_sos_off', 'home_sos_def', 'away_sos_off', 'away_sos_def'
        ]
        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
                
        if 'total_score' in df.columns:
            return X[expected], df['total_score']
        return X[expected]
