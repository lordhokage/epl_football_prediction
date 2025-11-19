"""
Backtesting module for EPL football prediction model.
Evaluates model performance on historical test data (2017-2025).
"""

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import json


def load_test_data(start_date=None, end_date=None, data_path="data/merged_after_odds.csv"):
    """
    Load historical match data for backtesting.
    
    Args:
        start_date: Start date for filtering (datetime or string YYYY-MM-DD)
        end_date: End date for filtering (datetime or string YYYY-MM-DD)
        data_path: Path to the match data CSV file
        
    Returns:
        DataFrame with Date, HomeTeam, AwayTeam, FTR (Full Time Result), and other match data
    """
    # Load the data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by date range if provided
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df['Date'] >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df['Date'] <= end_date]
    
    # Ensure we have the required columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df.sort_values('Date').reset_index(drop=True)


def map_result_to_label(ftr_code):
    """
    Convert FTR code to readable label.
    
    Args:
        ftr_code: 'H' (Home win), 'D' (Draw), or 'A' (Away win)
        
    Returns:
        String label: "Home Win", "Draw", or "Away Win"
    """
    mapping = {
        'H': 'Home Win',
        'D': 'Draw',
        'A': 'Away Win'
    }
    return mapping.get(ftr_code, 'Unknown')


def run_backtest(start_date, end_date, model, team_code_map, team_stats, feature_names, build_feature_fn):
    """
    Run backtesting on historical data.
    
    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        model: Loaded XGBoost model (booster object)
        team_code_map: Dictionary mapping team names to codes
        team_stats: Dictionary of team statistics
        feature_names: List of feature names expected by the model
        build_feature_fn: Function to build feature row (same as in app.py)
        
    Returns:
        DataFrame with predictions, actuals, confidence scores, and comparison
    """
    # Load test data
    test_data = load_test_data(start_date, end_date)
    
    results = []
    
    for idx, row in test_data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        actual_result = row['FTR']
        
        # Skip if teams not in our mapping
        if home_team not in team_code_map or away_team not in team_code_map:
            continue
        
        try:
            # Build features for this match using the same logic as prediction
            feature_row = build_feature_fn(home_team, away_team, match_date)
            
            # Get prediction probabilities
            import xgboost as xgb
            dm = xgb.DMatrix(feature_row, feature_names=list(feature_row.columns))
            probs = model.predict(dm)
            probs = np.asarray(probs, dtype=np.float64).ravel()
            
            # Get predicted outcome (0=Home Win, 1=Draw, 2=Away Win)
            pred_idx = int(np.argmax(probs))
            pred_labels = ['Home Win', 'Draw', 'Away Win']
            predicted_outcome = pred_labels[pred_idx]
            
            # Get actual outcome
            actual_outcome = map_result_to_label(actual_result)
            
            # Check if prediction is correct
            is_correct = (predicted_outcome == actual_outcome)
            
            # Store results
            results.append({
                'Date': match_date.strftime('%Y-%m-%d'),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Predicted_Outcome': predicted_outcome,
                'Actual_Outcome': actual_outcome,
                'Home_Win_Confidence': float(probs[0] * 100),
                'Draw_Confidence': float(probs[1] * 100),
                'Away_Win_Confidence': float(probs[2] * 100),
                'Correct': is_correct
            })
            
        except Exception as e:
            # Skip matches that cause errors (e.g., missing data)
            print(f"Skipping match {home_team} vs {away_team} on {match_date}: {e}")
            continue
    
    return pd.DataFrame(results)


def calculate_metrics(backtest_results):
    """
    Calculate performance metrics from backtest results.
    
    Args:
        backtest_results: DataFrame from run_backtest()
        
    Returns:
        Dictionary with performance metrics
    """
    if len(backtest_results) == 0:
        return {
            'error': 'No results to calculate metrics',
            'total_matches': 0
        }
    
    total_matches = len(backtest_results)
    correct_predictions = backtest_results['Correct'].sum()
    overall_accuracy = (correct_predictions / total_matches * 100) if total_matches > 0 else 0
    
    # Accuracy by outcome type
    outcome_types = ['Home Win', 'Draw', 'Away Win']
    accuracy_by_outcome = {}
    
    for outcome in outcome_types:
        outcome_matches = backtest_results[backtest_results['Actual_Outcome'] == outcome]
        if len(outcome_matches) > 0:
            outcome_correct = outcome_matches['Correct'].sum()
            accuracy_by_outcome[outcome] = round(outcome_correct / len(outcome_matches) * 100, 2)
        else:
            accuracy_by_outcome[outcome] = 0.0
    
    # Confusion matrix data
    confusion = {}
    for actual in outcome_types:
        confusion[actual] = {}
        for predicted in outcome_types:
            count = len(backtest_results[
                (backtest_results['Actual_Outcome'] == actual) & 
                (backtest_results['Predicted_Outcome'] == predicted)
            ])
            confusion[actual][predicted] = count
    
    # Average confidence for correct vs incorrect predictions
    correct_results = backtest_results[backtest_results['Correct'] == True]
    incorrect_results = backtest_results[backtest_results['Correct'] == False]
    
    avg_confidence_correct = 0.0
    avg_confidence_incorrect = 0.0
    
    if len(correct_results) > 0:
        # Get max confidence for each correct prediction
        max_confidences_correct = correct_results[
            ['Home_Win_Confidence', 'Draw_Confidence', 'Away_Win_Confidence']
        ].max(axis=1)
        avg_confidence_correct = round(max_confidences_correct.mean(), 2)
    
    if len(incorrect_results) > 0:
        # Get max confidence for each incorrect prediction
        max_confidences_incorrect = incorrect_results[
            ['Home_Win_Confidence', 'Draw_Confidence', 'Away_Win_Confidence']
        ].max(axis=1)
        avg_confidence_incorrect = round(max_confidences_incorrect.mean(), 2)
    
    return {
        'total_matches': total_matches,
        'correct_predictions': int(correct_predictions),
        'overall_accuracy': round(overall_accuracy, 2),
        'accuracy_by_outcome': accuracy_by_outcome,
        'confusion_matrix': confusion,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_incorrect': avg_confidence_incorrect
    }
