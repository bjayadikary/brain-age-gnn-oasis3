
import pandas as pd
import sys

def main(pred_path, test_nodes_path):
    preds = pd.read_csv(pred_path)
    test_nodes = pd.read_csv(test_nodes_path)

    if "id" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("predictions.csv must contain id and y_pred")

    if preds["id"].duplicated().any():
        raise ValueError("Duplicate IDs found")

    if preds["y_pred"].isna().any():
        raise ValueError("NaN predictions found")

    if ((preds["y_pred"] < 0) | (preds["y_pred"] > 1)).any():
        raise ValueError("Predictions must be in [0,1]")

    if set(preds["id"]) != set(test_nodes["id"]):
        raise ValueError("Prediction IDs do not match test nodes")

    print("VALID SUBMISSION")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
