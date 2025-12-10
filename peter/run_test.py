"""
Load a saved model, evaluate on a test split, and write predictions.csv.

Usage:
    python run_test.py --test_csv PATH_TO_TEST --model_path saved_model.joblib --pred_out predictions.csv
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd

import peter as pipeline


def evaluate(test_csv: Path, model_path: Path, pred_out: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    test_df = pipeline.load_dataset(test_csv, "test")
    test_embeddings = pipeline.compute_embeddings(test_df, "test")

    artifacts = joblib.load(model_path)
    prototypes = artifacts["prototypes"]
    clf = artifacts["classifier"]

    metrics, details = pipeline.evaluate_split(
        test_df, test_embeddings, "test", prototypes, clf, return_details=True
    )

    pred_rows = []
    for idx, true_label, pred_label in zip(details["indices"], details["y_true"], details["y_pred"]):
        row = test_df.iloc[idx]
        pred_rows.append(
            {
                "row_index": idx,
                "metaphorID": row["metaphorID"],
                "true_label": true_label,
                "prediction": pred_label,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_out, index=False)

    accuracy, precision, recall, f1 = metrics
    print(f"Predictions saved to {pred_out}")
    print(
        f"Final metrics — Accuracy: {accuracy:.4f}, Precision (macro): {precision:.4f}, "
        f"Recall (macro): {recall:.4f}, F1 (macro): {f1:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=pipeline.TEST_CSV,
        help="Path to test CSV.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(__file__).resolve().parent / "model.joblib",
        help="Path to trained model artifacts.",
    )
    parser.add_argument(
        "--pred_out",
        type=Path,
        default=Path(__file__).resolve().parent / "predictions.csv",
        help="Path to write predictions CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.test_csv, args.model_path, args.pred_out)
