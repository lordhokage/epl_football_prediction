
import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import gradio as gr
import requests

# Import backtesting modules
from backtesting import run_backtest, calculate_metrics
from utils import season_to_date, format_backtest_results, get_available_seasons

MODEL_JSON_PATH = "model/model.json"
TEAM_MAP_PATH = "model/team_code_map.json"
TEAMS_PATH = "model/teams.json"
TEAM_STATS_PATH = "model/team_stats.json"

LABELS = ["Home Win", "Draw", "Away Win"]

ALIASES = {
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Newcastle United": "Newcastle United",
    "West Ham United": "West Ham United",
    "AFC Bournemouth": "Bournemouth",
    "Nottingham Forest": "Nottingham Forest",
    "Leeds United": "Leeds United",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Everton": "Everton",
    "Chelsea": "Chelsea",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Burnley": "Burnley",
    "Sunderland": "Sunderland",
}

status_boot = ""
predict_proba_fn = None
feature_names = None


try:
    import xgboost as xgb
    if os.path.exists(MODEL_JSON_PATH):
        booster = xgb.Booster()
        booster.load_model(MODEL_JSON_PATH)
        # Try to read feature names saved with the model
        try:
            feature_names = booster.feature_names
        except Exception:
            feature_names = None

        def predict_proba_fn_df(df_row: pd.DataFrame):
            dm = xgb.DMatrix(df_row, feature_names=list(df_row.columns))
            out = booster.predict(dm)
            return np.asarray(out, dtype=np.float64).ravel()

        def predict_proba_fn_np(X_np: np.ndarray):
            dm = xgb.DMatrix(X_np.astype(np.float32))
            out = booster.predict(dm)
            return np.asarray(out, dtype=np.float64).ravel()

        # Choose DF flow if we have named features
        if feature_names and len(feature_names) > 0:
            predict_proba_fn = ("df", predict_proba_fn_df)
            status_boot = f"Model loaded (JSON Booster) with {len(feature_names)} features"
        else:
            predict_proba_fn = ("np", predict_proba_fn_np)
            status_boot = "Model loaded (JSON Booster) without saved feature names"
    else:
        status_boot = "Model not found: upload model/model.json"
except Exception as e:
    status_boot = f"Model load error: {type(e).__name__}: {e}"

# Teams and mapping
try:
    with open(TEAM_MAP_PATH, "r", encoding="utf-8") as f:
        team_code_map = json.load(f)
except Exception:
    team_code_map = {}
try:
    with open(TEAMS_PATH, "r", encoding="utf-8") as f:
        TEAM_LIST = json.load(f)
except Exception:
    TEAM_LIST = sorted(list(team_code_map.keys()))

# Load team statistics
try:
    with open(TEAM_STATS_PATH, "r", encoding="utf-8") as f:
        team_stats = json.load(f)
except Exception:
    team_stats = {}

def normalize_api_team(name: str) -> str:
    v = ALIASES.get(name, name)
    return v if v in team_code_map else name

def parse_input_date(s):
    if s is None:
        return None
    if isinstance(s, dt.date):
        return dt.datetime.combine(s, dt.time.min)
    if isinstance(s, dt.datetime):
        return s
    s = str(s).strip()
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None

def fetch_fixture_date(home_team: str, away_team: str):
    try:
        token = os.getenv("FOOTBALL_DATA_API_TOKEN", "")
        headers = {"X-Auth-Token": token} if token else {}
        today = dt.date.today()
        next_range = (today + dt.timedelta(days=180)).strftime("%Y-%m-%d")
        prev_range = (today - dt.timedelta(days=365)).strftime("%Y-%m-%d")
        url = "https://api.football-data.org/v4/matches"

        # Upcoming fixtures
        r1 = requests.get(
            url,
            params={"competitions": "PL", "status": "SCHEDULED",
                    "dateFrom": today.strftime("%Y-%m-%d"), "dateTo": next_range},
            headers=headers, timeout=15
        )
        if r1.status_code == 200:
            for m in r1.json().get("matches", []):
                h = normalize_api_team(m["homeTeam"]["name"])
                a = normalize_api_team(m["awayTeam"]["name"])
                if h == home_team and a == away_team:
                    iso = m["utcDate"]
                    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).date()
                    return d

        # Recent finished fixtures (fallback)
        r2 = requests.get(
            url,
            params={"competitions": "PL", "status": "FINISHED",
                    "dateFrom": prev_range, "dateTo": today.strftime("%Y-%m-%d")},
            headers=headers, timeout=15
        )
        if r2.status_code == 200:
            cand = []
            for m in r2.json().get("matches", []):
                h = normalize_api_team(m["homeTeam"]["name"])
                a = normalize_api_team(m["awayTeam"]["name"])
                if h == home_team and a == away_team:
                    iso = m["utcDate"]
                    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).date()
                    cand.append(d)
            if cand:
                return max(cand)
    except Exception:
        pass
    return None

def build_feature_row(home_team: str, away_team: str, when: dt.datetime) -> pd.DataFrame:
    # Map to codes
    home_code = team_code_map.get(home_team)
    away_code = team_code_map.get(away_team)
    day_code = when.weekday()
    is_weekend = 1.0 if day_code in (5, 6) else 0.0

    if not feature_names or len(feature_names) == 0:
        # No named features saved; return only the minimal predictors as DataFrame
        cols = ["home_code", "away_code", "day_code"]
        return pd.DataFrame([[home_code, away_code, day_code]], columns=cols)

    # Get team statistics
    home_stats = team_stats.get(home_team, {})
    away_stats = team_stats.get(away_team, {})
    
    # Calculate matches played (estimate based on season)
    matches_played = 20.0  # Mid-season estimate
    
    # Named features model: fill with realistic team statistics
    row = {}
    for fn in feature_names:
        if fn == "team_code":
            row[fn] = float(home_code)
        elif fn == "opponent_code":
            row[fn] = float(away_code)
        elif fn == "day_code":
            row[fn] = float(day_code)
        elif fn == "is_weekend":
            row[fn] = float(is_weekend)
        elif fn == "matches_played":
            row[fn] = matches_played
        elif fn == "opp_matches_played":
            row[fn] = matches_played
        elif fn == "season_points_to_date":
            row[fn] = home_stats.get("season_points_rate", 1.2) * matches_played
        elif fn == "opp_season_points_to_date":
            row[fn] = away_stats.get("season_points_rate", 1.2) * matches_played
        elif fn == "season_points_rate":
            row[fn] = home_stats.get("season_points_rate", 1.2)
        elif fn == "opp_season_points_rate":
            row[fn] = away_stats.get("season_points_rate", 1.2)
        elif fn == "season_goal_diff_to_date":
            row[fn] = home_stats.get("season_goal_diff_to_date", 0.0)
        elif fn == "opp_season_goal_diff_to_date":
            row[fn] = away_stats.get("season_goal_diff_to_date", 0.0)
        elif fn == "days_since_last":
            row[fn] = 7.0  # Assume weekly matches
        # Rolling statistics for home team
        elif fn.startswith("gf_roll") or fn.startswith("ga_roll") or \
             fn.startswith("shots_for_roll") or fn.startswith("shots_against_roll") or \
             fn.startswith("shots_on_target_for_roll") or fn.startswith("shots_on_target_against_roll") or \
             fn.startswith("corners_for_roll") or fn.startswith("corners_against_roll") or \
             fn.startswith("fouls_for_roll") or fn.startswith("fouls_against_roll") or \
             fn.startswith("yellows_for_roll") or fn.startswith("yellows_against_roll") or \
             fn.startswith("reds_for_roll") or fn.startswith("reds_against_roll") or \
             fn.startswith("goal_diff_roll") or fn.startswith("team_points_roll") or \
             fn.startswith("wins_last_") or fn.startswith("points_last_"):
            row[fn] = home_stats.get(fn, 0.0)
        # Gap features (home - away)
        elif fn == "season_points_gap":
            row[fn] = home_stats.get("season_points_rate", 1.2) * matches_played - \
                      away_stats.get("season_points_rate", 1.2) * matches_played
        elif fn == "season_goal_diff_gap":
            row[fn] = home_stats.get("season_goal_diff_to_date", 0.0) - \
                      away_stats.get("season_goal_diff_to_date", 0.0)
        elif fn == "season_points_rate_gap":
            row[fn] = home_stats.get("season_points_rate", 1.2) - \
                      away_stats.get("season_points_rate", 1.2)
        elif fn == "matches_played_gap":
            row[fn] = 0.0  # Assume same number of matches
        else:
            # Unknown feature; use safe default
            row[fn] = 0.0
    
    return pd.DataFrame([row], columns=feature_names)

def predict(home_team: str, away_team: str, date_str: str):
    try:
        if predict_proba_fn is None:
            return "Model unavailable. Upload model/model.json", {}
        if not home_team or not away_team:
            return "Select teams", {}
        if home_team == away_team:
            return "Home and Away cannot be the same team", {}
        when = parse_input_date(date_str)
        if when is None:
            return "Invalid date. Use YYYY-MM-DD", {}
        if home_team not in team_code_map or away_team not in team_code_map:
            return "Unknown team", {}

        kind, fn = predict_proba_fn
        if kind == "df":
            df_row = build_feature_row(home_team, away_team, when)
            probs = fn(df_row)
        else:
            # Minimal numpy path if model doesn't carry names
            day_code = when.weekday()
            X = np.array([[team_code_map[home_team], team_code_map[away_team], day_code]], dtype=np.float32)
            probs = fn(X)

        probs = np.asarray(probs, dtype=np.float64).ravel()
        if probs.size != 3:
            return f"Internal error: expected 3 classes, got {probs.size}", {}

        pred_idx = int(np.argmax(probs))
        # Convert to percentage format
        return LABELS[pred_idx], {LABELS[i]: f"{float(probs[i]) * 100:.1f}%" for i in range(3)}
    except Exception as e:
        return f"Internal error: {type(e).__name__}: {e}", {}

def find_date_ui(home_team: str, away_team: str, current_date_text: str):
    try:
        if not home_team or not away_team or home_team == away_team:
            return gr.update(), "Select different Home and Away"
        d = fetch_fixture_date(home_team, away_team)
        if d is None:
            msg = "No fixture found. Enter a date manually"
            if not os.getenv("FOOTBALL_DATA_API_TOKEN"):
                msg += " (set FOOTBALL_DATA_API_TOKEN to enable lookup)"
            return gr.update(), msg
        return d.isoformat(), f"Fixture found: {d.isoformat()}"
    except Exception as e:
        return gr.update(), f"Lookup error: {type(e).__name__}: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# EPL Match Outcome Predictor")
    gr.Markdown(status_boot)
    
    # Note about predictions
    gr.Markdown("""
    **Note:** Predictions are based on average season statistics and historical patterns.
    """)

    with gr.Row():
        home = gr.Dropdown(choices=TEAM_LIST, label="Home Team")
        away = gr.Dropdown(choices=TEAM_LIST, label="Away Team")
        date = gr.Textbox(label="Match Date (YYYY-MM-DD)",
                          value=dt.date.today().isoformat(),
                          placeholder="YYYY-MM-DD")
    with gr.Row():
        btn_find = gr.Button("Find Scheduled Fixture Date")
        btn_predict = gr.Button("Predict")
    
    # Get available seasons
    try:
        available_seasons = get_available_seasons()
    except:
        available_seasons = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", 
                            "2022-23", "2023-24", "2024-25"]
    
    with gr.Row():
        season_start = gr.Dropdown(
            choices=available_seasons,
            label="Start Season",
            value=available_seasons[-2] if len(available_seasons) >= 2 else available_seasons[0]
        )
        season_end = gr.Dropdown(
            choices=available_seasons,
            label="End Season",
            value=available_seasons[-1]
        )
        btn_backtest = gr.Button("Run Backtest", variant="primary")
    
    backtest_status = gr.Markdown("")
    backtest_metrics = gr.JSON(label="Performance Metrics")
    backtest_results = gr.Dataframe(
        label="Detailed Results (showing first 100 matches)",
        wrap=True
    )
    
    def run_backtest_ui(season_start_str, season_end_str):
        """
        Run backtesting for the selected season range.
        """
        try:
            if not season_start_str or not season_end_str:
                return "Please select both start and end seasons", {}, pd.DataFrame()
            
            # Convert seasons to dates
            start_date = season_to_date(season_start_str, is_start=True)
            end_date = season_to_date(season_end_str, is_start=False)
            
            # Check if model is loaded
            if predict_proba_fn is None or not hasattr(booster, 'predict'):
                return "Model not loaded. Cannot run backtest.", {}, pd.DataFrame()
            
            # Run backtest
            status_msg = f"Running backtest from {season_start_str} to {season_end_str}..."
            
            results_df = run_backtest(
                start_date=start_date,
                end_date=end_date,
                model=booster,
                team_code_map=team_code_map,
                team_stats=team_stats,
                feature_names=feature_names,
                build_feature_fn=build_feature_row
            )
            
            if len(results_df) == 0:
                return "No matches found in the selected date range.", {}, pd.DataFrame()
            
            # Calculate metrics
            metrics = calculate_metrics(results_df)
            
            # Format results for display
            display_df = format_backtest_results(results_df)
            
            # Limit display to first 100 rows for performance
            if len(display_df) > 100:
                display_df = display_df.head(100)
                status_msg = f"✓ Backtest complete! Analyzed {len(results_df)} matches. Showing first 100 results."
            else:
                status_msg = f"✓ Backtest complete! Analyzed {len(results_df)} matches."
            
            return status_msg, metrics, display_df
            
        except Exception as e:
            error_msg = f"Error running backtest: {type(e).__name__}: {str(e)}"
            return error_msg, {}, pd.DataFrame()
    
    btn_backtest.click(
        run_backtest_ui,
        inputs=[season_start, season_end],
        outputs=[backtest_status, backtest_metrics, backtest_results]
    )


if __name__ == "__main__":
    demo.launch()