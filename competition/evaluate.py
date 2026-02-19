import os
import pandas as pd
import io
import argparse
from sklearn.metrics import mean_absolute_error

def evaluate(submission_file_path):
    # 1. Load the participant's CSV
    try:
        submission_df = pd.read_csv(submission_file_path)
    except Exception as e:
        raise ValueError(f"Could not read submission CSV: {e}")

    # 2. Load Ground Truth from GitHub Secret
    secret_data = os.getenv('TEST_LABELS')
    if not secret_data:
        raise ValueError("Ground truth labels (TEST_LABELS) not found in environment")
    
    true_df = pd.read_csv(io.StringIO(secret_data))

    # 3. Standardize column names to lowercase for robust matching
    true_df.columns = true_df.columns.str.lower()
    submission_df.columns = submission_df.columns.str.lower()

    # 4. Alignment & Validation
    if 'subject_session' not in submission_df.columns:
        raise ValueError("Submission must contain 'subject_session' column")

    # Check if all required IDs are present
    missing = set(true_df['subject_session']) - set(submission_df['subject_session'])
    if missing:
        raise ValueError(f"Missing IDs in submission: {list(missing)[:5]}...")
    
    # Merge data on the ID column
    merged = pd.merge(true_df, submission_df, on='subject_session', suffixes=('_true', '_pred'))

    # 5. Calculate Metrics
    # Ensure these match the secret labels exactly: 'age_at_visit'
    y_true = merged['age_at_visit_true']
    y_pred = merged['age_at_visit_pred']

    mae = mean_absolute_error(y_true, y_pred)
    
    # We print the score so the GitHub Action can see it in the logs
    print(f"SCORE_MAE: {round(mae, 5)}")
    
    return mae

# This block handles the "--file" argument from your .yml workflow
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to predictions.csv")
    args = parser.parse_args()
    
    evaluate(args.file)