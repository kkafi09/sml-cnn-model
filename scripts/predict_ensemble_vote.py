"""Build a test prediction file from soft DiT scores and hard model votes."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from document_classifier.data import load_metadata, validate_metadata
from document_classifier.submission import write_submission


def load_hard_predictions(path: Path, expected_ids: list[str]) -> np.ndarray:
    """Load a model submission and verify its IDs and labels match the test set."""
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "label"]:
        raise ValueError(f"Expected id,label columns in {path}")
    ids = frame["id"].astype(str).tolist()
    if ids != expected_ids or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"Submission IDs are missing, duplicated, or misordered: {path}")
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy()
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"Noninteger labels in {path}")
    labels = labels.astype(np.int64)
    if ((labels < 0) | (labels > 15)).any():
        raise ValueError(f"Labels outside 0-15 in {path}")
    return labels


def main() -> None:
    """Combine test-set probabilities and hard votes into a submission CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dit-probabilities", type=Path, required=True)
    parser.add_argument("--donut-submission", type=Path, required=True)
    parser.add_argument("--resnet-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dit-weight", type=float, default=0.60)
    parser.add_argument("--donut-weight", type=float, default=0.11)
    parser.add_argument("--resnet-weight", type=float, default=0.29)
    args = parser.parse_args()
    weights = np.asarray([args.dit_weight, args.donut_weight, args.resnet_weight])
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        parser.error("Weights must be nonnegative, finite, and sum to 1")

    test = load_metadata(args.data_dir, "test")
    validate_metadata(test, "test")
    ids = test["id"].astype(str).tolist()
    if len(ids) != 7994:
        raise ValueError(f"Expected 7,994 test rows, found {len(ids)}")
    with np.load(args.dit_probabilities, allow_pickle=False) as cached:
        dit_probabilities = cached["probabilities"]
    if dit_probabilities.shape != (len(ids), 16) or not np.isfinite(dit_probabilities).all():
        raise ValueError("DiT probability cache has an invalid shape or nonfinite values")
    if (dit_probabilities < 0).any() or not np.allclose(dit_probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("DiT values must be normalized class probabilities")

    donut_labels = load_hard_predictions(args.donut_submission, ids)
    resnet_labels = load_hard_predictions(args.resnet_submission, ids)
    donut_votes = np.eye(16, dtype=np.float32)[donut_labels]
    resnet_votes = np.eye(16, dtype=np.float32)[resnet_labels]
    combined = (weights[0] * dit_probabilities + weights[1] * donut_votes
                + weights[2] * resnet_votes)
    predictions = combined.argmax(axis=1).astype(np.int64).tolist()
    write_submission(ids, predictions, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Wrote {len(predictions)} predictions to {args.output}; SHA-256: {digest}")


if __name__ == "__main__":
    main()
