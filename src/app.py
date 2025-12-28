# Package markers
import sys
import os
import threading
import time
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from flask import Flask, render_template, request, flash, session, redirect, url_for, g
from dotenv import load_dotenv

# Add the current directory to sys.path so it can find database, data_loader, etc.
sys.path.append(os.path.dirname(__file__))

# Initialize basic app immediately to bind port
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'shadow_ncca_secret_v1')

@app.route('/healthz')
def healthz():
    # Render's health check should be ultra-fast and never trigger imports
    return {"status": "ok", "message": "Port bound"}, 200

@app.route('/startup-status')
def startup_status():
    global imports_loaded
    return {
        "imports_loaded": imports_loaded,
        "db_connected": db_session is not None,
        "model_ready": check_model_ready() if predictor else False
    }

# Now import the rest lazily
def load_heavy_imports():
    global pd, init_db, Game, Prediction, DataLoader, APIDataLoader, ScorePredictor
    import pandas as pd
    from database import init_db, Game, Prediction
    from data_loader import DataLoader
    from api_loader import APIDataLoader
    from model import ScorePredictor

# Placeholder globals
pd = None
init_db = None
Game = None
Prediction = None
DataLoader = None
APIDataLoader = None
ScorePredictor = None
imports_loaded = False

load_dotenv()

@app.errorhandler(500)
def internal_error(error):
    import traceback
    return f"500 Error: {str(error)}<br><pre>{traceback.format_exc()}</pre>", 500

@app.before_request
def setup_and_check_access():
    # Bypass auth and lazy loading for static files, health checks, and status
    if request.path.startswith('/static') or request.path in ['/healthz', '/startup-status', '/login']:
        return

    # 1. Access control (Fast check)
    access_key = os.getenv('ACCESS_KEY')
    if access_key and session.get('authorized') != access_key:
        return redirect(url_for('login'))

    # 2. Lazy loading (only if authorized and not a static/health path)
    if not imports_loaded:
        # If imports aren't ready, we don't want to block the request thread and hit OOM
        # But we also can't show the page. 
        # For now, let's try to load them but with a warning.
        print(f"Request to {request.path} triggered lazy loading...")
        try:
            get_loaders()
            get_predictor()
        except Exception as e:
            print(f"Lazy loading error: {e}")

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

# Initialize components as None for lazy loading
db_session = None
predictor = None
mock_loader = None
api_loader = None
current_loader = None
tz_wat = ZoneInfo("Africa/Lagos")

import threading
init_lock = threading.Lock()

def get_db_session():
    global db_session, imports_loaded
    with init_lock:
        if not imports_loaded:
            load_heavy_imports()
            imports_loaded = True
        if db_session is None:
            try:
                db_session = init_db()
            except Exception as e:
                print(f"DATABASE CONNECTION ERROR: {e}")
                return None
    return db_session

def get_predictor():
    global predictor, imports_loaded
    with init_lock:
        if not imports_loaded:
            load_heavy_imports()
            imports_loaded = True
        if predictor is None:
            predictor = ScorePredictor()
    return predictor

def get_loaders():
    global mock_loader, api_loader, current_loader, db_session, imports_loaded
    with init_lock:
        if not imports_loaded:
            load_heavy_imports()
            imports_loaded = True
        if mock_loader is None:
            if db_session is None:
                # Inside lock already, but get_db_session also locks. 
                # Re-entrant locks or just direct call? 
                # Let's just do it directly here since we have the lock.
                try:
                    db_session = init_db()
                except Exception as e:
                    print(f"DATABASE CONNECTION ERROR in get_loaders: {e}")
            
            mock_loader = DataLoader(db_session)
            api_key = os.getenv('ODDS_API_KEY')
            if api_key:
                api_loader = APIDataLoader(api_key)
                current_loader = api_loader
            else:
                api_loader = None
                current_loader = mock_loader
    return mock_loader, api_loader, current_loader

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

# Ensure model is trained AND team stats are loaded on startup
def startup_load_stats():
    get_loaders()
    import os
    print("--- STARTUP FILE DISCOVERY ---")
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")
    
    # Identify the actual project root (where .git or team_stats.csv should be)
    # If we are in /opt/render/project/src/src, the root is /opt/render/project/src
    # If we are in /opt/render/project/src, the root is /opt/render/project/src
    
    potential_roots = [
        cwd,
        os.path.dirname(cwd),
        BASE_DIR,
        os.path.abspath(os.path.join(SRC_DIR, '..'))
    ]
    
    csv_paths = []
    for root in potential_roots:
        csv_paths.append(os.path.join(root, 'team_stats.csv'))
    
    # Add some hardcoded absolute paths common on Render/Docker
    csv_paths.extend([
        '/app/team_stats.csv',
        '/opt/render/project/src/team_stats.csv',
        'team_stats.csv'
    ])
    
    # Remove duplicates
    csv_paths = list(dict.fromkeys([os.path.abspath(p) for p in csv_paths]))
    
    loaded = False
    print("Searching for team_stats.csv...")
    for p in csv_paths:
        if os.path.exists(p):
            print(f"FOUND: {p}")
            if mock_loader.load_team_stats_from_csv(p):
                print(f"SUCCESS: Loaded {len(mock_loader.team_stats)} teams from {p}")
                loaded = True
                break
        else:
            print(f"NOT FOUND at: {p}")
    
    if not loaded:
         print("CRITICAL: team_stats.csv not found in standard paths. Starting deep search...")
         # Search everywhere from current dir downwards
         for root, dirs, files in os.walk(os.path.dirname(BASE_DIR) if os.path.exists(os.path.dirname(BASE_DIR)) else BASE_DIR):
             for f in files:
                 if f.lower() == 'team_stats.csv':
                     target = os.path.join(root, f)
                     print(f"DEEP SEARCH FOUND: {target}")
                     if mock_loader.load_team_stats_from_csv(target):
                         print(f"SUCCESS: Loaded {len(mock_loader.team_stats)} teams from deep search.")
                         loaded = True
                         break
             if loaded: break
    
    if not loaded:
        print("ALERT: team_stats.csv is missing from the deployment. Please ensure it is in your GitHub root.")

    # Also load stats from DB if available
    db_df = mock_loader.fetch_db_training_data(min_rows=1)
    if not db_df.empty:
        print(f"Found {len(db_df)} games in database. Updating team stats...")
        mock_loader.build_team_stats(db_df)
        mock_loader.build_team_bias_from_db()
        mock_loader.analyze_prediction_performance()

# Improved check for model readiness
def check_model_ready():
    pred = get_predictor()
    if not pred.is_trained: return False
    if not getattr(pred, 'feature_names', None): return False
    if len(pred.feature_names) < 3: return False
    
    # Check if scaler is fitted
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    try:
        check_is_fitted(pred.scaler)
    except NotFittedError:
        return False
    return True

# Wrap startup logic to prevent crash
def background_initialization():
    try:
        # Initialize everything
        print("Initializing database and loaders in background...")
        get_loaders()
        get_predictor()
        
        startup_load_stats()
        if not check_model_ready():
            print("Model not ready. Starting initial training sequence...")
            db_df = mock_loader.fetch_db_training_data(min_rows=50)
            if not db_df.empty:
                print(f"Training on {len(db_df)} database games.")
                X, y = mock_loader.prepare_features(db_df)
            else:
                print("Database empty. Generating 500 simulated games for initial training...")
                history_df = mock_loader.fetch_historical_data(num_games=500)
                X, y = mock_loader.prepare_features(history_df)
            
            mae = predictor.train(X, y)
            print(f"Initial training complete. Model MAE: {mae:.2f}")
        else:
            print("Model loaded and ready for predictions.")
    except Exception as e:
        import traceback
        print(f"STARTUP ERROR: {e}")
        print(traceback.format_exc())

# Run initialization in a thread to not block gunicorn port binding
import time

def singleton_background_initialization():
    global imports_loaded
    # Simple file lock to ensure only one worker runs initialization
    lock_file = "init.lock"
    
    # On Render, the filesystem is ephemeral, so this works for process synchronization
    # across workers in the same container.
    if os.path.exists(lock_file):
        # Check if the lock is old (e.g., > 10 mins) in case of a crash
        if time.time() - os.path.getmtime(lock_file) < 600:
            print("Initialization already in progress by another worker. Skipping.")
            return
            
    try:
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))
        
        # Ensure imports are done in the background thread too
        if not imports_loaded:
            load_heavy_imports()
            imports_loaded = True
            
        background_initialization()
        
        # Keep the lock file but update it
        with open(lock_file, "w") as f:
            f.write("done")
    except Exception as e:
        print(f"Locking error: {e}")
        if os.path.exists(lock_file):
            os.remove(lock_file)

print("Starting background initialization thread...")
threading.Thread(target=singleton_background_initialization, daemon=True).start()

@app.route('/')
def home():
    # Fetch some summary stats for the home page
    stats = {
        'total_games': 0,
        'total_preds': 0,
        'accuracy': 0,
        'last_update': datetime.now(tz_wat).strftime("%Y-%m-%d %H:%M")
    }
    
    if db_session:
        try:
            total_games = db_session.query(Game).filter(Game.status == 'finished').count()
            total_preds = db_session.query(Prediction).join(Game).filter(Game.status == 'finished').count()
            
            # Calculate accuracy
            wins = 0
            all_finished = db_session.query(Prediction).join(Game).filter(Game.status == 'finished').all()
            for p in all_finished:
                if p.predicted_class == p.game.result:
                    wins += 1
            
            accuracy = (wins / total_preds * 100) if total_preds > 0 else 0
            
            stats.update({
                'total_games': total_games,
                'total_preds': total_preds,
                'accuracy': round(accuracy, 1)
            })
        except Exception as e:
            print(f"Error fetching home stats: {e}")
    
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
        # Clear old logs if they get too big
        if os.path.exists('output.txt') and os.path.getsize('output.txt') > 1024 * 1024:
            with open('output.txt', 'w') as f: f.write("Log rotated\n")

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
        # Try to fetch games
        try:
            if api_loader:
                games_df = api_loader.fetch_upcoming_games()
                source = "Real-Time API"
            else:
                # If no API key, we show nothing (no mock games as requested)
                games_df = pd.DataFrame()
                source = "No Data Source"
        except Exception as e:
            print(f"Error fetching games: {e}")
            games_df = pd.DataFrame()
            source = "Error"

        print(f"Fetched {len(games_df)} games from {source}")
        predictions = []
        
        # Relax the time filter to show games from 12 hours ago to 72 hours in the future
        now_utc = datetime.now(ZoneInfo("UTC"))
        start_limit = now_utc - timedelta(hours=12)
        end_limit = now_utc + timedelta(hours=72)
        
        print(f"Filtering games between {start_limit} and {end_limit}")
        
        if not games_df.empty:
            for i, row in games_df.iterrows():
                # Skip games without a line
                line_val = row.get('over_under_line')
                if pd.isna(line_val) or line_val is None:
                    print(f"Skipping {row.get('team_home')} vs {row.get('team_away')} - No Line")
                    continue
                    
                dt = pd.to_datetime(row['date'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    dt = dt.astimezone(ZoneInfo("UTC"))
                
                if not (start_limit <= dt <= end_limit):
                    print(f"Skipping {row.get('team_home')} vs {row.get('team_away')} - Outside time range ({dt})")
                    continue
                
                print(f"Processing game: {row['team_home']} vs {row['team_away']} at {dt}")
                game_date = dt

                game = db_session.query(Game).filter_by(
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
                    db_session.add(game)
                    db_session.flush() # Ensure ID is populated

                existing_pred = (
                    db_session.query(Prediction)
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
                    db_session.add(prediction_record)
                    
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
            db_session.commit()
        
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
    games = db_session.query(Game).order_by(Game.date.asc()).all()
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
            db_session.delete(g)
    db_session.commit()


@app.route('/history')
def history():
    auto_update_results()
    try:
        dedupe_games()
    except Exception:
        pass
    
    # 1. Fetch ALL predictions for overall stats
    all_predictions = (
        db_session.query(Prediction)
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
        db_session.add(m)
        db_session.commit()
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
        db_session.query(Prediction)
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
        db_session.query(Prediction)
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
    open_games = db_session.query(Game).filter(Game.status != 'finished').all()
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
        db_session.commit()
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
        os.path.join(os.getcwd(), 'team_stats.csv'),
        os.path.join(SRC_DIR, 'team_stats.csv'),
        os.path.join(SRC_DIR, 'data', 'team_stats.csv'),
        'team_stats.csv'
    ]
    for p in csv_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and mock_loader.load_team_stats_from_csv(abs_p):
            flash(f"Team stats reloaded from {abs_p}")
            return redirect('/')
    
    # Final recursive fallback
    for root, dirs, files in os.walk(BASE_DIR):
        if 'team_stats.csv' in files:
            target = os.path.join(root, 'team_stats.csv')
            if mock_loader.load_team_stats_from_csv(target):
                flash(f"Team stats reloaded from recursive search: {target}")
                return redirect('/')

    flash("No team_stats.csv found in any expected location.")
    return redirect('/')

if __name__ == '__main__':
    # When running locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
