import os
import pandas as pd
import io
import argparse
from sklearn.metrics import mean_absolute_error
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import unpad

def decrypt_file_to_df(file_path):
    """Decrypts a .enc file and returns a pandas DataFrame."""
    # 1. Get Private Key from GitHub Secret
    private_key_str = os.getenv('RSA_PRIVATE_KEY')
    if not private_key_str:
        raise ValueError("RSA_PRIVATE_KEY secret is not set in GitHub.")

    # 2. Prepare the RSA Key
    private_key = RSA.import_key(private_key_str)
    
    with open(file_path, 'rb') as f:
        # Read metadata for hybrid decryption
        enc_session_key_size = int.from_bytes(f.read(2), byteorder='big')
        enc_session_key = f.read(enc_session_key_size)
        iv = f.read(16)
        ciphertext = f.read()

    # 3. Decrypt the AES session key with RSA
    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)

    # 4. Decrypt the CSV data with AES
    cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
    decrypted_raw = unpad(cipher_aes.decrypt(ciphertext), AES.block_size)
    
    # 5. Convert bytes to DataFrame
    return pd.read_csv(io.BytesIO(decrypted_raw))

def evaluate(submission_file_path):
    # 1. Load the participant's data (Decrypt if necessary)
    try:
        if submission_file_path.endswith('.enc'):
            submission_df = decrypt_file_to_df(submission_file_path)
        else:
            submission_df = pd.read_csv(submission_file_path)
    except Exception as e:
        print(f"ERROR: Could not process submission file - {e}")
        return

    # 2. Load Ground Truth from GitHub Secret
    secret_data = os.getenv('TEST_LABELS')
    if not secret_data:
        print("ERROR: TEST_LABELS secret is not set in GitHub.")
        return
    
    true_df = pd.read_csv(io.StringIO(secret_data))

    # 3. Standardize column names
    true_df.columns = true_df.columns.str.lower()
    submission_df.columns = submission_df.columns.str.lower()

    # 4. Validation
    required_cols = ['subject_session', 'age_at_visit']
    for col in required_cols:
        if col not in submission_df.columns:
            print(f"ERROR: Missing required column '{col}'")
            return

    # 5. Alignment
    missing = set(true_df['subject_session']) - set(submission_df['subject_session'])
    if missing:
        print(f"ERROR: Submission is missing {len(missing)} IDs. First few: {list(missing)[:3]}")
        return
    
    merged = pd.merge(true_df, submission_df, on='subject_session', suffixes='_true', '_pred')

    # 6. Calculate Metric
    y_true = merged['age_at_visit_true']
    y_pred = merged['age_at_visit_pred']

    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"SCORE_MAE: {round(mae, 5)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to the predictions file")
    args = parser.parse_args()
    
    evaluate(args.file)