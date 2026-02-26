import os
import pandas as pd
from io import StringIO

def calculate_mae(ground_truth_df, prediction_df):
    merged = pd.merge(ground_truth_df, prediction_df, on='subject_session', suffixes=('_true', '_pred'))
    if merged.empty:
        return None
    mae = (merged['age_at_visit_true'] - merged['age_at_visit_pred']).abs().mean()
    return round(float(mae), 8)

# 1. Load Ground Truth
gt_data = os.getenv('TEST_LABELS')
if not gt_data:
    print("Error: TEST_LABELS secret not found.")
    exit(1)
gt_df = pd.read_csv(StringIO(gt_data))

# 2. LOAD EXISTING DATA (Persistent History)
csv_path = 'leaderboard/leaderboard.csv'
existing_teams = set()
leaderboard_data = []

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    # Standardize headers to find 'TEAM' regardless of original casing
    existing_df.columns = existing_df.columns.str.strip().str.upper()
    
    if 'TEAM' in existing_df.columns:
        # Cache existing names for the strict check in Section 3
        existing_teams = set(existing_df['TEAM'].astype(str).str.upper().unique())
        leaderboard_data = existing_df.to_dict('records')

# 3. Scan for NEW submissions with STRICT duplicate check
submissions_dir = 'submissions'
if os.path.exists(submissions_dir):
    for team_name in os.listdir(submissions_dir):
        if not team_name or team_name.lower() == 'nan' or team_name.startswith('.'):
            continue

        # --- THE STRICT CHECK ---
        # Blocks the merge if the folder name is already in the CSV history
        if team_name.upper() in existing_teams:
            print(f"❌ ERROR: Team '{team_name}' already exists in the leaderboard.")
            print("To update, delete the old entry from leaderboard.csv or use a unique name.")
            exit(1) 
        # ------------------------

        team_path = os.path.join(submissions_dir, team_name)
        pred_file = os.path.join(team_path, 'predictions.csv')
        
        if os.path.isdir(team_path) and os.path.exists(pred_file):
            try:
                pred_df = pd.read_csv(pred_file)
                score = calculate_mae(gt_df, pred_df)
                if score is not None:
                    leaderboard_data.append({"TEAM": team_name, "MAE": score})
            except Exception as e:
                print(f"Error processing {team_name}: {e}")

# 4. Create Leaderboard (Full History)
if leaderboard_data:
    df = pd.DataFrame(leaderboard_data)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()
    
    # Collapse duplicate columns (merges 'Team' and 'TEAM')
    df = df.groupby(level=0, axis=1).first()

    # Clean up and force numeric
    df = df.dropna(subset=['TEAM'])
    df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')

    # DEDUPLICATION: Cleans up any existing duplicates in the history
    df = df.sort_values(by=['MAE']).drop_duplicates(subset=['TEAM'], keep='first')

    # Final Sort and Rank
    df = df.dropna(subset=['MAE']).sort_values(by=["MAE", "TEAM"])
    df['RANK'] = df['MAE'].rank(method='dense').astype(int)
    
    leaderboard_df = df[['RANK', 'TEAM', 'MAE']]
    leaderboard_df.columns = ['Rank', 'Team', 'MAE']

    # 5. Save CSV & Markdown
    os.makedirs('leaderboard', exist_ok=True)
    leaderboard_df.to_csv(csv_path, index=False)
    
    with open('leaderboard/LEADERBOARD.md', 'w') as f:
        f.write("# 🏆 Full Competition History\n\n" + leaderboard_df.to_markdown(index=False))

    # 6. Generate HTML for GitHub Pages (docs/ folder)
    os.makedirs('docs', exist_ok=True)
    
    # Format MAE to show 8 decimals in the web view
    html_table = leaderboard_df.to_html(
        classes='table table-hover text-center', 
        index=False,
        formatters={'MAE': lambda x: f"{x:.8f}"}
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <title>OASIS3 Challenge</title>
        <style>
            body {{ background-color: #f4f7f6; font-family: 'Inter', sans-serif; padding: 40px 0; }}
            .leaderboard-card {{ background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; max-width: 800px; margin: auto; }}
            .header-section {{ background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 40px 20px; }}
            table {{ width: 100% !important; margin-bottom: 0 !important; }}
            th {{ background-color: #f8fafc !important; color: #64748b; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; text-align: center; padding: 15px !important; }}
            td {{ vertical-align: middle; font-size: 1rem; padding: 15px !important; text-align: center; }}
            .mae-value {{ font-family: 'monospace'; font-weight: bold; color: #059669; white-space: nowrap;}}
        </style>
    </head>
    <body>
        <div class="leaderboard-card">
            <div class="header-section text-center">
                <h1 class="fw-bold">🧠 Brain-Age Prediction Challenge Leaderboard</h1>
                <div class="badge bg-primary mt-2">Last Updated: {pd.Timestamp.now().strftime('%b %d, %H:%M UTC')}</div>
            </div>
            <div class="table-responsive">
                {html_table}
            </div>
        </div>
    </body>
    </html>
    """
    with open('docs/leaderboard.html', 'w') as f:
        f.write(html_content)
    
    print("Leaderboard and HTML updated successfully.")