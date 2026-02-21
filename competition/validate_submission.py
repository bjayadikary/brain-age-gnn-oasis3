import pandas as pd
import sys
import os
import io
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import unpad

def decrypt_to_df(encrypted_path):
    """Helper to decrypt .enc files for validation."""
    private_key_str = os.getenv('RSA_PRIVATE_KEY')
    if not private_key_str:
        raise ValueError("RSA_PRIVATE_KEY secret is not set.")

    private_key = RSA.import_key(private_key_str)
    
    with open(encrypted_path, 'rb') as f:
        enc_session_key_size = int.from_bytes(f.read(2), byteorder='big')
        enc_session_key = f.read(enc_session_key_size)
        iv = f.read(16)
        ciphertext = f.read()

    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)

    cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
    decrypted_raw = unpad(cipher_aes.decrypt(ciphertext), AES.block_size)
    
    return pd.read_csv(io.BytesIO(decrypted_raw))

def main(pred_path, test_nodes_path):
    try:
        # 1. Load the data (Handle Encrypted vs Regular)
        if pred_path.endswith('.enc'):
            preds = decrypt_to_df(pred_path)
        else:
            preds = pd.read_csv(pred_path)
            
        test_nodes = pd.read_csv(test_nodes_path)

        # 2. Standardize Column Names (Optional but safer)
        preds.columns = preds.columns.str.lower()
        test_nodes.columns = test_nodes.columns.str.lower()

        # 3. Check for required columns 
        # (Updated to match your brain-age columns: 'subject_session' and 'age_at_visit')
        if "subject_session" not in preds.columns or "age_at_visit" not in preds.columns:
            raise ValueError("Submission must contain 'subject_session' and 'age_at_visit'")

        if preds["subject_session"].duplicated().any():
            raise ValueError("Duplicate subject_session IDs found")

        if preds["age_at_visit"].isna().any():
            raise ValueError("NaN predictions found")

        # 4. Check IDs match ground truth
        if set(preds["subject_session"]) != set(test_nodes["subject_session"]):
            missing = set(test_nodes["subject_session"]) - set(preds["subject_session"])
            raise ValueError(f"IDs do not match test nodes. Missing {len(missing)} IDs.")

        print("VALID SUBMISSION")

    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])