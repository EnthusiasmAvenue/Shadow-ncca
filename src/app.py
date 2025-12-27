from flask import Flask, render_template, request, flash, session, redirect, url_for
import os
import sys

# Add the current directory to sys.path so it can find database, data_loader, etc.
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from database import init_db, Game, Prediction
from data_loader import DataLoader
from api_loader import APIDataLoader
from model import ScorePredictor

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'shadow_ncca_secret_v1')

@app.before_request
def check_access():
    # Bypass auth for static files, health checks, and the login page itself
    if request.path.startswith('/static') or request.path == '/healthz' or request.path == '/login':
        return
    
    access_key = os.getenv('ACCESS_KEY')
    if not access_key:
        return # No access key set, app is public

    # Check if user is authorized in session
    if session.get('authorized') == access_key:
        return

    # Otherwise redirect to login
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    access_key = os.getenv('ACCESS_KEY')
    if not access_key:
        return redirect(url_for('home'))

    if request.method == 'POST':
        entered_key = request.form.get('access_key')
        if entered_key == access_key:
            session['authorized'] = access_key
            return redirect(url_for('home'))
        else:
            flash("Invalid access code.")
    
    return render_template('login.html')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)

# Initialize components
session = init_db()
predictor = ScorePredictor()
api_key = os.getenv('ODDS_API_KEY')
tz_wat = ZoneInfo("Africa/Lagos")

def wat_time(value):
    try:
        dt = pd.to_datetime(value)
    except Exception:
        dt = value
    if hasattr(dt, 'tzinfo') and dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    if not hasattr(dt, 'astimezone'):
        return str(value)
    return dt.astimezone(tz_wat).strftime("%Y-%m-%d %H:%M")

app.jinja_env.filters['wat_time'] = wat_time

@app.context_processor
def inject_globals():
    return {'os_getenv': os.getenv}

# Initialize Loaders
mock_loader = DataLoader(session)
if api_key:
    api_loader = APIDataLoader(api_key)
    current_loader = api_loader
else:
    api_loader = None
    current_loader = mock_loader

# Ensure model is trained AND team stats are loaded on startup
def startup_load_stats():
    csv_paths = [
        os.path.join(BASE_DIR, 'team_stats.csv'),
        os.path.join(SRC_DIR, 'data', 'team_stats.csv')
    ]
    loaded = False
    for p in csv_paths:
        if os.path.exists(p):
            if mock_loader.load_team_stats_from_csv(p):
                loaded = True
                break
    
    # Also load stats from DB if available
    db_df = mock_loader.fetch_db_training_data(min_rows=1)
    if not db_df.empty:
        mock_loader.build_team_stats(db_df)
        mock_loader.build_team_bias_from_db()
        mock_loader.analyze_prediction_performance()

startup_load_stats()

# Improved check for model readiness
def check_model_ready():
    if not predictor.is_trained: return False
    if not getattr(predictor, 'feature_names', None): return False
    if len(predictor.feature_names) < 3: return False
    
    # Check if scaler is fitted
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    try:
        check_is_fitted(predictor.scaler)
    except NotFittedError:
        return False
    return True

if not check_model_ready():
    print("Training initial model (scaler or features missing)...")
    db_df = mock_loader.fetch_db_training_data(min_rows=50)
    if not db_df.empty:
        X, y = mock_loader.prepare_features(db_df)
    else:
        history_df = mock_loader.fetch_historical_data(num_games=500)
        X, y = mock_loader.prepare_features(history_df)
    predictor.train(X, y)

@app.route('/')
def home():
    # Fetch some summary stats for the home page
    try:
        total_games = session.query(Game).filter(Game.status == 'finished').count()
        total_preds = session.query(Prediction).join(Game).filter(Game.status == 'finished').count()
        
        # Calculate accuracy
        wins = 0
        all_finished = session.query(Prediction).join(Game).filter(Game.status == 'finished').all()
        for p in all_finished:
            if p.predicted_class == p.game.result:
                wins += 1
        
        accuracy = (wins / total_preds * 100) if total_preds > 0 else 0
        
        stats = {
            'total_games': total_games,
            'total_preds': total_preds,
            'accuracy': round(accuracy, 1),
            'last_update': datetime.now(tz_wat).strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        print(f"Error fetching home stats: {e}")
        stats = None

    return render_template('index.html', stats=stats)

@app.route('/predict')
def predict():
    try:
        print("Entered /predict route")
        
        # --- AUTOMATED RETRAINING CHECK ---
        # If we have new finished games since the last training, retrain.
        last_train_file = os.path.join(BASE_DIR, 'last_train.txt')
        needs_retrain = False
        if os.path.exists(last_train_file):
            try:
                with open(last_train_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        last_train_time = datetime.fromisoformat(content)
                        if datetime.now() - last_train_time > timedelta(hours=24):
                            needs_retrain = True
                    else:
                        needs_retrain = True
            except Exception:
                needs_retrain = True
        else:
            needs_retrain = True

        if needs_retrain:
            print("Auto-retraining model with latest data...")
            db_df = mock_loader.fetch_db_training_data(min_rows=50)
            if not db_df.empty:
                X, y = mock_loader.prepare_features(db_df)
                predictor.train(X, y)
                with open(last_train_file, 'w') as f:
                    f.write(datetime.now().isoformat())
        # Clear today's predictions from the DB to force a refresh
        try:
            from datetime import time
            today_start = datetime.combine(datetime.now().date(), time.min)
            today_end = datetime.combine(datetime.now().date(), time.max)
            
            preds_to_delete = (
                session.query(Prediction)
                .join(Game)
                .filter(Game.date >= today_start, Game.date <= today_end)
                .all()
            )
            print(f"Found {len(preds_to_delete)} predictions to clear")
            for p in preds_to_delete:
                session.delete(p)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error clearing today's predictions: {e}")

        pick_filter = request.args.get('pick')
        conf_min = request.args.get('min_conf')
        try:
            conf_min_val = float(conf_min) if conf_min is not None else None
        except Exception:
            conf_min_val = None
        try:
            print("Deduping games...")
            dedupe_games()
        except Exception as e:
            print(f"Dedupe error: {e}")
        
        print("Fetching games...")
        if api_loader:
            games_df = api_loader.fetch_upcoming_games()
            source = "Real-Time API"
        else:
            games_df = mock_loader.fetch_upcoming_games()
            source = "Mock Simulation"

        print(f"Fetched {len(games_df)} games")
        predictions = []
        
        if not games_df.empty:
            now_utc = datetime.now(ZoneInfo("UTC"))
            # Show games starting between now and 48 hours from now
            limit_future = now_utc + timedelta(hours=48)
            
            for i, row in games_df.iterrows():
                print(f"Processing game {i+1}/{len(games_df)}: {row['team_home']} vs {row['team_away']}")
                # Skip games without a line
                line_val = row.get('over_under_line')
                if pd.isna(line_val) or line_val is None:
                    print(f"Skipping game {i+1} due to missing line")
                    continue
                    
                dt = pd.to_datetime(row['date'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    dt = dt.astimezone(ZoneInfo("UTC"))
                
                # Filter: only show games starting in the next 20 hours (covers tonight and tomorrow morning)
                if not (now_utc <= dt <= limit_future):
                    continue
                
                game_date = dt # Already converted to UTC datetime object above

                game = session.query(Game).filter_by(
                    date=game_date,
                    team_home=row['team_home'],
                    team_away=row['team_away']
                ).first()

                if not game:
                    game = Game(
                        date=game_date,
                        team_home=row['team_home'],
                        team_away=row['team_away'],
                        over_under_line=row['over_under_line']
                    )
                    session.add(game)
                    session.flush() # Ensure ID is populated

                existing_pred = (
                    session.query(Prediction)
                    .filter(Prediction.game_id == game.id)
                    .order_by(Prediction.created_at.asc())
                    .first()
                )

                if existing_pred:
                    pred_total = round(existing_pred.predicted_total, 1)
                    decision = existing_pred.predicted_class
                    line_val = game.over_under_line if game.over_under_line is not None else row['over_under_line']
                    diff_val = round(pred_total - line_val, 1)
                    
                    conf_val = existing_pred.confidence if existing_pred.confidence is not None else 50.0
                    conf_str = f"{float(conf_val):.1f}%"
                    
                    predictions.append({
                        'matchup': f"{row['team_home']} vs {row['team_away']}",
                        'date': row['date'],
                        'line': line_val,
                        'predicted_total': pred_total,
                        'decision': decision,
                        'confidence': conf_str,
                        'diff': diff_val,
                        'explanation': existing_pred.explanation or "No analysis available."
                    })
                else:
                    features_df = pd.DataFrame([row])
                    X_pred = mock_loader.prepare_features(features_df)
                    result = predictor.predict_game(
                        X_pred, 
                        row['over_under_line'],
                        home_team=row['team_home'],
                        away_team=row['team_away']
                    )
                    
                    # Safeguard for confidence conversion
                    try:
                        conf_str_raw = str(result['confidence'])
                        confidence_value = float(conf_str_raw.replace('%', ''))
                    except Exception:
                        confidence_value = 50.0

                    prediction_record = Prediction(
                        game=game,
                        predicted_total=result['predicted_total'],
                        predicted_class=result['decision'],
                        confidence=confidence_value,
                        explanation=result.get('explanation', ''),
                        model_version="v2"
                    )
                    session.add(prediction_record)
                    
                    predictions.append({
                        'matchup': f"{row['team_home']} vs {row['team_away']}",
                        'date': row['date'],
                        'line': game.over_under_line if game.over_under_line is not None else row['over_under_line'],
                        'predicted_total': result['predicted_total'],
                        'decision': result['decision'],
                        'confidence': result['confidence'],
                        'diff': result['diff'],
                        'explanation': result.get('explanation', '')
                    })
            session.commit()
        
        print(f"Generated {len(predictions)} predictions")
        
        if pick_filter:
            predictions = [p for p in predictions if p['decision'].lower() == pick_filter.lower()]
        if conf_min_val is not None:
            def conf_to_float(c):
                try:
                    return float(str(c).replace('%', ''))
                except Exception:
                    return 0.0
            predictions = [p for p in predictions if conf_to_float(p['confidence']) >= conf_min_val]

        return render_template('predictions.html', predictions=predictions, source=source)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"CRITICAL ERROR in /predict: {error_msg}")
        return f"Internal Server Error: {str(e)}<br><pre>{error_msg}</pre>", 500


def dedupe_games():
    games = session.query(Game).order_by(Game.date.asc()).all()
    keep = {}
    for g in games:
        key = (g.date, g.team_home, g.team_away)
        if key not in keep:
            keep[key] = g
        else:
            canonical = keep[key]
            for p in g.predictions:
                p.game = canonical
            if canonical.total_score is None and g.total_score is not None:
                canonical.score_home = g.score_home
                canonical.score_away = g.score_away
                canonical.total_score = g.total_score
                canonical.result = g.result
            session.delete(g)
    session.commit()


@app.route('/history')
def history():
    auto_update_results()
    try:
        dedupe_games()
    except Exception:
        pass
    
    # 1. Fetch ALL predictions for overall stats
    all_predictions = (
        session.query(Prediction)
        .join(Game)
        .filter((Game.total_score != None) | (Game.status == 'live'))
        .order_by(Game.date.desc(), Prediction.created_at.desc())
        .all()
    )

    all_records = []
    for prediction in all_predictions:
        game = prediction.game
        if not game: continue

        if game.status == 'live':
            outcome_label, mark, outcome_class, final_result = "Live", "⋯", "live-row", "TBD"
        elif game.result == "Push":
            outcome_label, mark, outcome_class, final_result = "Push", "–", "push", "Push"
        elif game.result:
            if prediction.predicted_class == game.result:
                outcome_label, mark, outcome_class = "Win", "✓", "good"
            else:
                outcome_label, mark, outcome_class = "Loss", "✗", "bad"
            final_result = game.result
        else: continue

        all_records.append({
            'date': game.date,
            'matchup': f"{game.team_home} vs {game.team_away}",
            'line': game.over_under_line,
            'predicted_total': prediction.predicted_total,
            'predicted_class': prediction.predicted_class,
            'confidence': prediction.confidence,
            'total_score': game.total_score if game.total_score is not None else 0,
            'final_result': final_result,
            'status': getattr(game, 'status', 'finished'),
            'outcome': outcome_label,
            'mark': mark,
            'outcome_class': outcome_class,
            'explanation': prediction.explanation or "No analysis available."
        })

    # Deduplicate records by date/matchup
    seen = set()
    unique_all = []
    for r in all_records:
        key = (r['date'], r['matchup'])
        if key not in seen:
            seen.add(key)
            unique_all.append(r)

    # 2. Filter for a 24-hour "Current Window" (past 16h to future 12h)
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now_utc = datetime.now(ZoneInfo("UTC"))
    window_start = now_utc - timedelta(hours=16)
    window_end = now_utc + timedelta(hours=12)
    
    def in_display_window(record_date):
        try:
            if isinstance(record_date, str):
                dt = pd.to_datetime(record_date)
            else:
                dt = record_date
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            else:
                dt = dt.astimezone(ZoneInfo("UTC"))
            
            return window_start <= dt <= window_end
        except Exception:
            return False

    display_records = [r for r in unique_all if in_display_window(r['date'])]
    
    # If no games in window, show the most recent day's games
    if not display_records and unique_all:
        latest_date = unique_all[0]['date']
        latest_dt = pd.to_datetime(latest_date) if isinstance(latest_date, str) else latest_date
        display_records = [r for r in unique_all if (r['date'] if isinstance(r['date'], str) else r['date'].strftime('%Y-%m-%d')).startswith(latest_dt.strftime('%Y-%m-%d'))]

    # 3. Calculate Overall Stats (from all time)
    wins_count = len([r for r in unique_all if r['outcome'] == 'Win'])
    losses_count = len([r for r in unique_all if r['outcome'] == 'Loss'])
    pushes_count = len([r for r in unique_all if r['final_result'] == 'Push'])
    total_non_push = wins_count + losses_count
    total_win_rate = round((wins_count / total_non_push) * 100, 1) if total_non_push > 0 else 0.0

    # 4. Calculate Last Prediction Win Rate
    # Find the most recent finished prediction result
    last_finished = [r for r in unique_all if r['outcome'] in ['Win', 'Loss']]
    if last_finished:
        last_result = last_finished[0]
        last_prediction_wr = 100.0 if last_result['outcome'] == 'Win' else 0.0
    else:
        last_prediction_wr = 0.0

    stats = {
        'win_rate': total_win_rate,
        'last_prediction_wr': last_prediction_wr,
        'wins': wins_count,
        'losses': losses_count,
        'pushes': pushes_count,
        'total_pred': total_non_push
    }
    try:
        from database import Metric
        m = Metric(mae=None, win_rate=total_win_rate, note="daily")
        session.add(m)
        session.commit()
    except Exception:
        pass

    team_errors = {}
    for prediction in all_predictions:
        g = prediction.game
        if not g or g.total_score is None:
            continue
        err = float(prediction.predicted_total) - float(g.total_score)
        for team in [g.team_home, g.team_away]:
            if team not in team_errors:
                team_errors[team] = {'errs': []}
            team_errors[team]['errs'].append(err)
    team_rows = []
    for team, d in team_errors.items():
        errs = d['errs']
        c = len(errs)
        mae = round(sum(abs(e) for e in errs) / c, 2) if c > 0 else 0.0
        bias = round(sum(errs) / c, 2) if c > 0 else 0.0
        lastN = errs[:20]
        lc = len(lastN)
        last_mae = round(sum(abs(e) for e in lastN) / lc, 2) if lc > 0 else 0.0
        last_bias = round(sum(lastN) / lc, 2) if lc > 0 else 0.0
        team_rows.append({
            'team': team,
            'count': c,
            'mae': mae,
            'bias': bias,
            'last_mae': last_mae,
            'last_bias': last_bias
        })
    team_rows = sorted(team_rows, key=lambda r: r['count'], reverse=True)[:12]

    return render_template('history.html', records=display_records, stats=stats, team_rows=team_rows)

@app.route('/export-history.csv', methods=['GET'])
def export_history_csv():
    preds = (
        session.query(Prediction)
        .join(Game)
        .filter(Game.total_score != None)
        .order_by(Game.date.desc(), Prediction.created_at.asc())
        .all()
    )
    rows = []
    for p in preds:
        g = p.game
        rows.append({
            'date': str(g.date),
            'team_home': g.team_home,
            'team_away': g.team_away,
            'line': g.over_under_line,
            'predicted_total': p.predicted_total,
            'predicted_class': p.predicted_class,
            'confidence': p.confidence,
            'total_score': g.total_score,
            'result': g.result
        })
    import csv, io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()) if rows else ['date'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    from flask import Response
    return Response(output.getvalue(), mimetype='text/csv')

@app.route('/export-history.json', methods=['GET'])
def export_history_json():
    preds = (
        session.query(Prediction)
        .join(Game)
        .filter(Game.total_score != None)
        .order_by(Game.date.desc(), Prediction.created_at.asc())
        .all()
    )
    rows = []
    for p in preds:
        g = p.game
        rows.append({
            'date': str(g.date),
            'team_home': g.team_home,
            'team_away': g.team_away,
            'line': g.over_under_line,
            'predicted_total': p.predicted_total,
            'predicted_class': p.predicted_class,
            'confidence': p.confidence,
            'total_score': g.total_score,
            'result': g.result
        })
    from flask import jsonify
    return jsonify(rows)

def auto_update_results():
    now = datetime.now(ZoneInfo("UTC"))
    # Query games that are NOT finished yet
    open_games = session.query(Game).filter(Game.status != 'finished').all()
    updated = 0
    if api_loader:
        scores_df = api_loader.fetch_recent_scores(days=3)
        for g in open_games:
            match = None
            if not scores_df.empty:
                m1 = scores_df[(scores_df['team_home'] == g.team_home) & (scores_df['team_away'] == g.team_away)]
                m2 = scores_df[(scores_df['team_home'] == g.team_away) & (scores_df['team_away'] == g.team_home)]
                if not m1.empty:
                    match = m1.iloc[0]
                elif not m2.empty:
                    match = m2.iloc[0]
            
            if match is not None:
                import math
                import pandas as pd
                def to_int_safe(v):
                    try:
                        if v is None:
                            return None
                        if isinstance(v, float) and (math.isnan(v)):
                            return None
                        if isinstance(v, str) and v.strip() == "":
                            return None
                        return int(float(v))
                    except Exception:
                        return None
                
                sh = to_int_safe(match.get('score_home'))
                sa = to_int_safe(match.get('score_away'))
                is_completed = match.get('completed', False)

                if sh is not None and sa is not None:
                    # Update scores (even if live)
                    g.score_home = sh
                    g.score_away = sa
                    g.total_score = sh + sa
                    
                    # Determine status
                    if is_completed:
                        g.status = 'finished'
                        if g.total_score > g.over_under_line:
                            g.result = "Over"
                        elif g.total_score < g.over_under_line:
                            g.result = "Under"
                        else:
                            g.result = "Push"
                    else:
                        g.status = 'live'
                    
                    updated += 1
    else:
        # Mock mode logic
        for g in open_games:
            dt = g.date
            if hasattr(dt, 'tzinfo') and dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            
            # If game time has passed
            if dt <= now:
                import numpy as np
                # If more than 3 hours have passed, mark as finished
                if now > (dt + timedelta(hours=3)):
                    if g.status != 'finished':
                        g.score_home = int(np.random.normal(75, 10))
                        g.score_away = int(np.random.normal(72, 10))
                        g.total_score = g.score_home + g.score_away
                        g.status = 'finished'
                        if g.total_score > g.over_under_line:
                            g.result = "Over"
                        elif g.total_score < g.over_under_line:
                            g.result = "Under"
                        else:
                            g.result = "Push"
                        updated += 1
                else:
                    # Otherwise mark as live and give a partial score
                    if g.status != 'live':
                        g.status = 'live'
                    g.score_home = int(np.random.normal(35, 5))
                    g.score_away = int(np.random.normal(33, 5))
                    g.total_score = g.score_home + g.score_away
                    updated += 1
    if updated:
        session.commit()
        try:
            maybe_retrain(updated_count=updated)
        except Exception:
            pass
        return updated
    return 0

last_retrain_time = None
def maybe_retrain(updated_count=0):
    global last_retrain_time
    from datetime import datetime, timedelta
    now = datetime.now(ZoneInfo("UTC"))
    
    # Retrain if we have enough new games or it's been a while
    if last_retrain_time and (now - last_retrain_time) < timedelta(hours=1):
        return
        
    db_df = mock_loader.fetch_db_training_data(min_rows=5) # Lowered min_rows to be more responsive
    if not db_df.empty and (updated_count >= 1 or not last_retrain_time):
        print(f"Self-learning triggered with {len(db_df)} games...")
        mock_loader.build_team_stats(db_df)
        mock_loader.build_team_bias_from_db()
        # Analyze wins/losses to find "what to look for" and "what to avoid"
        mock_loader.analyze_prediction_performance()
        X_new, y_new = mock_loader.prepare_features(db_df)
        
        if len(X_new) >= 10:
            mae = predictor.train(X_new, y_new)
            print(f"Model retrained. New MAE: {mae:.2f}")
            last_retrain_time = now
        else:
            print("Not enough featured data for retraining.")

@app.route('/auto-update', methods=['POST'])
def auto_update_endpoint():
    from flask import redirect
    n = auto_update_results()
    if n:
        flash(f"Auto-updated {n} games. Check the History page for live scores!")
    else:
        flash("No games to auto-update at this time.")
    return redirect('/')

@app.route('/reload-stats', methods=['POST'])
def reload_stats():
    from flask import redirect
    csv_paths = [
        os.path.join(BASE_DIR, 'team_stats.csv'),
        os.path.join(SRC_DIR, 'data', 'team_stats.csv')
    ]
    for p in csv_paths:
        if os.path.exists(p) and mock_loader.load_team_stats_from_csv(p):
            flash("Team stats reloaded from CSV.")
            return redirect('/')
    flash("No team_stats.csv found.")
    return redirect('/')

@app.route('/healthz', methods=['GET'])
def healthz():
    try:
        session.query(Game).limit(1).all()
        return "ok", 200
    except Exception:
        return "db error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
