import os
import pandas as pd
from io import StringIO

def calculate_mae(ground_truth_df, prediction_df):
    # Ensure columns are correct
    merged = pd.merge(ground_truth_df, prediction_df, on='subject_session', suffixes=('_true', '_pred'))
    if merged.empty:
        return None
    mae = (merged['age_at_visit_true'] - merged['age_at_visit_pred']).abs().mean()
    return round(mae, 4)

# 1. Load Ground Truth from GitHub Secret
gt_data = os.getenv('TEST_LABELS')
if not gt_data:
    print("Error: TEST_LABELS secret not found.")
    exit(1)
gt_df = pd.read_csv(StringIO(gt_data))

leaderboard_data = []

# 2. Scan the submissions folder
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

# 3. Create Leaderboard and Sort (Lower MAE is better)
if leaderboard_data:
    leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by="MAE")
    leaderboard_df.insert(0, 'Rank', range(1, len(leaderboard_df) + 1))

    # 4. Save to CSV
    leaderboard_df.to_csv('leaderboard/leaderboard.csv', index=False)

    # 5. Save to Markdown
    markdown_table = leaderboard_df.to_markdown(index=False)
    header = "# 🏆 Competition Leaderboard\n\n*Last Updated: Automatically*\n\n"
    with open('leaderboard/leaderboard.md', 'w') as f:
        f.write(header + markdown_table)
    print("Leaderboard updated successfully.")
else:
    print("No submissions found to rank.")
