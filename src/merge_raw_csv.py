import pandas as pd
import glob

# Step 1: List all CSV files
files = glob.glob('../data/raw/*.csv')

# Step 2: Columns you want to check
columns_to_keep = [
    'Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAHG', 'FTR',
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
]

# Step 3: Check each file
for file in files:
    df = pd.read_csv(file)
    cols_in_file = df.columns.tolist()
    missing_cols = [col for col in columns_to_keep if col not in cols_in_file]
    if missing_cols:
        print(f"{file} is missing columns: {missing_cols}")
    else:
        print(f"{file} contains all required columns.")
