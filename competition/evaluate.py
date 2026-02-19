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
        print(f"ERROR: Could not read submission CSV - {e}")
        return

    # 2. Load Ground Truth from GitHub Secret
    secret_data = os.getenv('TEST_LABELS')
    if not secret_data:
        print("ERROR: TEST_LABELS secret is not set in GitHub.")
        return
    
    true_df = pd.read_csv(io.StringIO(secret_data))

    # 3. Standardize column names to lowercase
    true_df.columns = true_df.columns.str.lower()
    submission_df.columns = submission_df.columns.str.lower()

    # 4. Validation: Check for required columns
    required_cols = ['subject_session', 'age_at_visit']
    for col in required_cols:
        if col not in submission_df.columns:
            print(f"ERROR: Missing required column '{col}'")
            return

    # 5. Alignment
    # Ensure all required IDs are present in the submission
    missing = set(true_df['subject_session']) - set(submission_df['subject_session'])
    if missing:
        print(f"ERROR: Submission is missing {len(missing)} IDs. First few: {list(missing)[:3]}")
        return
    
    # Merge on the ID column
    merged = pd.merge(true_df, submission_df, on='subject_session', suffixes=('_true', '_pred'))

    # 6. Calculate Metric
    y_true = merged['age_at_visit_true']
    y_pred = merged['age_at_visit_pred']

    mae = mean_absolute_error(y_true, y_pred)
    
    # This print statement is the 'handshake' with the GitHub Action
    print(f"SCORE_MAE: {round(mae, 5)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to the predictions.csv")
    args = parser.parse_args()
    
    evaluate(args.file)