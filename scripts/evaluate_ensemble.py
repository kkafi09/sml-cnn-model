"""Evaluate a fixed probability ensemble on the project's validation split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from document_classifier.data import DocumentImageDataset, load_metadata, validate_metadata
from document_classifier.model import DocumentCNN
from document_classifier.submission import write_submission
from document_classifier.transforms import build_image_transform


def predict_probabilities(checkpoint_path, metadata, data_dir, device, batch_size, workers):
    """Return probabilities in metadata order using the checkpoint's own transform."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != 42 or checkpoint.get("num_classes") != 16:
        raise ValueError(f"Unexpected seed or class count in {checkpoint_path}")
    model = DocumentCNN(pretrained=False, architecture=checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_image_transform(
        int(checkpoint["image_size"]),
        preserve_aspect_ratio=bool(checkpoint.get("preserve_aspect_ratio", False)),
    )
    dataset = DocumentImageDataset(metadata, data_dir, transform=transform, include_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    outputs, ids = [], []
    with torch.inference_mode():
        for index, (images, batch_ids) in enumerate(loader, 1):
            outputs.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
            ids.extend(batch_ids)
            if index % 50 == 0 or index == len(loader):
                print(f"{checkpoint_path.name}: {index}/{len(loader)} batches", flush=True)
    expected_ids = metadata["id"].astype(str).tolist()
    if ids != expected_ids:
        raise ValueError("Prediction IDs do not match metadata in order")
    probabilities = np.concatenate(outputs)
    if probabilities.shape != (len(metadata), 16) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model probabilities")
    return probabilities, checkpoint


def main():
    """Evaluate a fixed weighted ensemble and optionally predict the test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-validation", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if (len(args.checkpoint) < 2 or len(args.weight) != len(args.checkpoint)
            or not np.isfinite(args.weight).all() or any(weight < 0 for weight in args.weight)
            or sum(args.weight) <= 0 or not np.isfinite(args.minimum_validation)):
        parser.error("Provide at least two checkpoints and one nonnegative weight per checkpoint")
    weights = np.asarray(args.weight, dtype=np.float64)
    weights /= weights.sum()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_metadata(args.data_dir, "train")
    test = load_metadata(args.data_dir, "test")
    validate_metadata(train, "train")
    validate_metadata(test, "test")
    _, validation = train_test_split(train, test_size=0.2, random_state=42,
                                     stratify=train["label"].astype(int))
    if len(validation) != 6401 or len(test) != 7994:
        raise ValueError("Unexpected validation or test sample count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = validation["label"].to_numpy(dtype=np.int64)
    validation_probabilities = []
    checkpoint_info = []
    for path, weight in zip(args.checkpoint, weights):
        probabilities, checkpoint = predict_probabilities(
            path, validation, args.data_dir, device, args.batch_size, args.workers)
        score = float((probabilities.argmax(axis=1) == labels).mean())
        if abs(score - float(checkpoint["validation_accuracy"])) > 2 / len(labels):
            raise ValueError(f"Validation score does not reproduce checkpoint {path}")
        print(f"Validation {path}: {score:.8f}", flush=True)
        validation_probabilities.append(probabilities)
        checkpoint_info.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "weight": float(weight), "epoch": int(checkpoint["epoch"]),
                                "image_size": int(checkpoint["image_size"]),
                                "validation_accuracy": score})
    combined = sum(weight * probs for weight, probs in zip(weights, validation_probabilities))
    accuracy = float((combined.argmax(axis=1) == labels).mean())
    report = {"model": "weighted probability ensemble", "seed": 42,
              "validation_samples": len(validation), "validation_samples_used": len(validation),
              "best_validation_accuracy": accuracy, "checkpoints": checkpoint_info,
              "minimum_validation": args.minimum_validation, "test_predicted": False,
              "device": str(device)}
    print(f"Ensemble validation accuracy: {accuracy:.8f}", flush=True)
    if accuracy > args.minimum_validation:
        test_probabilities = []
        for path in args.checkpoint:
            probabilities, _ = predict_probabilities(
                path, test, args.data_dir, device, args.batch_size, args.workers)
            test_probabilities.append(probabilities)
        combined_test = sum(weight * probs for weight, probs in zip(weights, test_probabilities))
        submission_path = args.output_dir / "submission.csv"
        write_submission(test["id"].astype(str).tolist(), combined_test.argmax(axis=1).tolist(),
                         submission_path)
        report["test_predicted"] = True
        report["submission_sha256"] = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    (args.output_dir / "validation_result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
