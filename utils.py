"""
Utility functions for the backtesting feature.
"""

import datetime as dt
import pandas as pd


def season_to_date(season_str, is_start=True):
    """
    Convert season string to datetime.
    
    Args:
        season_str: Season string like "2023-24"
        is_start: If True, return start of season (Aug 1), else end (May 31)
        
    Returns:
        datetime object
    """
    # Extract start year from season string
    start_year = int(season_str.split('-')[0])
    
    if is_start:
        # Season starts in August
        return dt.datetime(start_year, 8, 1)
    else:
        # Season ends in May of the following year
        end_year = start_year + 1
        return dt.datetime(end_year, 5, 31)


def format_backtest_results(df):
    """
    Format backtest results DataFrame for Gradio display.
    
    Args:
        df: DataFrame from run_backtest()
        
    Returns:
        Formatted DataFrame with rounded values and visual indicators
    """
    if len(df) == 0:
        return df
    
    # Create a copy to avoid modifying original
    display_df = df.copy()
    
    # Round confidence percentages to 1 decimal place
    display_df['Home_Win_Confidence'] = display_df['Home_Win_Confidence'].round(1)
    display_df['Draw_Confidence'] = display_df['Draw_Confidence'].round(1)
    display_df['Away_Win_Confidence'] = display_df['Away_Win_Confidence'].round(1)
    
    # Add visual indicator for correct/incorrect
    display_df['Result'] = display_df['Correct'].apply(lambda x: '✓' if x else '✗')
    
    # Reorder and rename columns for better display
    display_df = display_df[[
        'Date', 'HomeTeam', 'AwayTeam', 
        'Predicted_Outcome', 'Actual_Outcome',
        'Home_Win_Confidence', 'Draw_Confidence', 'Away_Win_Confidence',
        'Result'
    ]]
    
    display_df.columns = [
        'Date', 'Home Team', 'Away Team',
        'Predicted', 'Actual',
        'Home Win %', 'Draw %', 'Away Win %',
        '✓/✗'
    ]
    
    return display_df


def get_available_seasons(data_path="data/merged_after_odds.csv"):
    """
    Get list of available seasons from the data.
    
    Args:
        data_path: Path to match data CSV
        
    Returns:
        List of season strings like ["2017-18", "2018-19", ...]
    """
    try:
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get min and max years
        min_year = df['Date'].min().year
        max_year = df['Date'].max().year
        
        # Generate season strings
        # A season like 2023-24 runs from Aug 2023 to May 2024
        seasons = []
        for year in range(min_year, max_year + 1):
            # Check if we have data for this season
            season_start = dt.datetime(year, 8, 1)
            season_end = dt.datetime(year + 1, 5, 31)
            
            matches_in_season = df[
                (df['Date'] >= season_start) & (df['Date'] <= season_end)
            ]
            
            if len(matches_in_season) > 0:
                seasons.append(f"{year}-{str(year + 1)[-2:]}")
        
        return seasons
    except Exception as e:
        # Return default seasons if we can't read the data
        print(f"Error getting available seasons: {e}")
        return ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", 
                "2022-23", "2023-24", "2024-25"]
