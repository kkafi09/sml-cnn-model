"""Summarize per-class validation errors for a recorded ensemble."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from document_classifier.data import load_metadata, validate_metadata
from evaluate_ensemble import predict_probabilities


CLASSES = ("advertisement", "budget", "email", "file_folder", "form", "handwritten",
           "invoice", "letter", "memo", "news_article", "presentation", "questionnaire",
           "resume", "scientific_publication", "scientific_report", "specification")


def main():
    """Reproduce validation predictions and write aggregate class-level errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("seed") != 42 or report.get("device") != "cpu":
        raise ValueError("Expected a CPU ensemble evaluated on the seed-42 split")
    frame = load_metadata(args.data_dir, "train")
    validate_metadata(frame, "train")
    _, validation = train_test_split(frame, test_size=0.2, random_state=42,
                                     stratify=frame["label"].astype(int))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    combined = None
    for member in report["checkpoints"]:
        checkpoint_path = Path(member["path"])
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Checkpoint hash differs from report: {checkpoint_path}")
        probabilities, _ = predict_probabilities(checkpoint_path, validation, args.data_dir,
                                                torch.device("cpu"), 16, 2, args.cache_dir)
        weighted = float(member["weight"]) * probabilities
        combined = weighted if combined is None else combined + weighted
    labels = validation["label"].to_numpy(dtype=np.int64)
    predictions = combined.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    if not np.isclose(accuracy, float(report["best_validation_accuracy"]), atol=1e-10):
        raise ValueError("Reproduced ensemble accuracy differs from its report")
    matrix = confusion_matrix(labels, predictions, labels=list(range(16)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "class_name", "samples", "correct",
                                                   "accuracy", "most_confused_with", "confusion_count"],
                                lineterminator="\n")
        writer.writeheader()
        for label, name in enumerate(CLASSES):
            row = matrix[label].copy()
            row[label] = 0
            mistaken = int(row.argmax()) if row.max() else None
            total = int(matrix[label].sum())
            writer.writerow({"label": label, "class_name": name, "samples": total,
                             "correct": int(matrix[label, label]),
                             "accuracy": float(matrix[label, label] / total),
                             "most_confused_with": CLASSES[mistaken] if mistaken is not None else "",
                             "confusion_count": int(row[mistaken]) if mistaken is not None else 0})
    print(f"Reproduced validation accuracy {accuracy:.8f}; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
