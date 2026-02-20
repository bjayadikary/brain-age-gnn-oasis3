import os
import pandas as pd
from io import StringIO

def calculate_mae(ground_truth_df, prediction_df):
    merged = pd.merge(ground_truth_df, prediction_df, on='subject_session', suffixes=('_true', '_pred'))
    if merged.empty:
        return None
    # Assuming columns are 'age_at_visit' (GT) and 'age_at_visit_pred' (Submissions)
    mae = (merged['age_at_visit'] - merged['age_at_visit_pred']).abs().mean()
    return round(mae, 4)

# 1. Load Ground Truth from Environment Secret
gt_data = os.getenv('TEST_LABELS')
if not gt_data:
    print("Error: TEST_LABELS secret not found.")
    exit(1)
gt_df = pd.read_csv(StringIO(gt_data))

leaderboard_data = []

# 2. Scan submissions (expects submissions/<team_name>/predictions.csv)
submissions_dir = 'submissions'
if os.path.exists(submissions_dir):
    for team_name in os.listdir(submissions_dir):
        team_path = os.path.join(submissions_dir, team_name)
        pred_file = os.path.join(team_path, 'predictions.csv')
        
        if os.path.isdir(team_path) and os.path.exists(pred_file):
            try:
                pred_df = pd.read_csv(pred_file)
                score = calculate_mae(gt_df, pred_df)
                if score is not None:
                    leaderboard_data.append({"Team": team_name, "MAE": score})
            except Exception as e:
                print(f"Error processing {team_name}: {e}")

# 3. Create Leaderboard
if leaderboard_data:
    # Sort by MAE (Ascending) and then Team name (Alphabetical) for ties
    leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by=["MAE", "Team"])
    
    # Assign Sequential Ranks (1, 2, 3...)
    leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
    
    # Reorder columns
    leaderboard_df = leaderboard_df[['Rank', 'Team', 'MAE']]

    # Function to add medals
    def format_rank(rank):
        if rank == 1: return "🥇 1st"
        if rank == 2: return "🥈 2nd"
        if rank == 3: return "🥉 3rd"
        return f"{rank}th"

    display_df = leaderboard_df.copy()
    display_df['Rank'] = display_df['Rank'].apply(format_rank)

    # 4. Save to CSV and Markdown for the repo
    leaderboard_df.to_csv('leaderboard/leaderboard.csv', index=False)
    with open('leaderboard/leaderboard.md', 'w') as f:
        f.write("# 🏆 Leaderboard\n\n" + display_df.to_markdown(index=False))

    # 5. Generate Flashy HTML for GitHub Pages
    html_table = display_df.to_html(classes='table table-hover', index=False, escape=False)
    
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
            .header-section {{ background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 40px 20px; text-center; }}
            th {{ background-color: #f8fafc !important; color: #64748b; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
            td {{ vertical-align: middle; font-size: 1rem; }}
            .mae-value {{ font-family: 'monospace'; font-weight: bold; color: #059669; }}
        </style>
    </head>
    <body>
        <div class="leaderboard-card">
            <div class="header-section text-center">
                <h1 class="fw-bold">🧠 Brain-Age Prediction Challenge</h1>
                <p>OASIS-3 GNN Benchmark Leaderboard</p>
                <div class="badge bg-primary">Last Updated: {pd.Timestamp.now().strftime('%b %d, %H:%M UTC')}</div>
            </div>
            <div class="table-responsive">
                {html_table}
            </div>
        </div>
    </body>
    </html>
    """
    with open('leaderboard.html', 'w') as f:
        f.write(html_content)
    print("Files updated successfully.")
