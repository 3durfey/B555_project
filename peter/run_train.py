"""
Train the metaphor detection model and save artifacts for later testing.

Usage:
    python run_train.py --train_csv PATH_TO_TRAIN --model_out saved_model.joblib
"""
from pathlib import Path
import argparse
import joblib
from sklearn.linear_model import LogisticRegression

import peter as pipeline


def train(train_csv: Path, model_out: Path):
    train_df = pipeline.load_dataset(train_csv, "train")
    train_embeddings = pipeline.compute_embeddings(train_df, "train")
    prototypes = pipeline.build_prototypes(train_df, train_embeddings)
    train_X, train_y = pipeline.build_distance_features(train_df, train_embeddings, prototypes, "train")

    clf = LogisticRegression()
    clf.fit(train_X, train_y)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"prototypes": prototypes, "classifier": clf}, model_out)
    print(f"Model saved to {model_out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=pipeline.TRAIN_CSV,
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path(__file__).resolve().parent / "model.joblib",
        help="Path to save trained artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.train_csv, args.model_out)
