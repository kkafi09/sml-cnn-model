"""Submit a validated candidate only when its accuracy and Kaggle quota allow it."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch


COMPETITION = "document-image-classification-sd234504"
THRESHOLD = 0.80
PUBLIC_TARGET = 0.79164
LIMIT = 5


def kaggle_submissions() -> list[dict]:
    """Fetch submission history or fail closed if it may be incomplete."""
    command = ["kaggle", "competitions", "submissions", "-c", COMPETITION,
               "--format", "json", "--page-size", "200"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) >= 200:
        raise RuntimeError("Cannot establish complete submission history; do not submit")
    return rows


def allowed_now() -> int:
    """Read the team's current allowance from Kaggle or fail closed."""
    command = ["kaggle", "competitions", "submission-limits", COMPETITION, "--json"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    limits = json.loads(result.stdout)
    remaining = limits.get("numAllowedNow") if isinstance(limits, dict) else None
    # Kaggle's JSON omits the zero-valued allowance once the daily limit is used.
    if remaining is None and isinstance(limits, dict):
        today = limits.get("numToday")
        if isinstance(today, int) and not isinstance(today, bool) and today >= LIMIT:
            return 0
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        raise ValueError("Kaggle did not provide an unambiguous submission allowance")
    return remaining


def count_today(rows: list[dict], now: datetime) -> int:
    """Count submissions on the current UTC date, including pending attempts."""
    count = 0
    for row in rows:
        value = row.get("date")
        if not value:
            raise ValueError("Submission history has no date; cannot check quota")
        submitted = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if submitted.tzinfo is None:
            submitted = submitted.replace(tzinfo=timezone.utc)
        submitted = submitted.astimezone(timezone.utc)
        if submitted > now + timedelta(minutes=5):
            raise ValueError("Submission history contains a future date")
        if submitted.date() == now.date():
            count += 1
    return count


def validate_prediction(path: Path, test_csv: Path) -> None:
    """Check the complete ordered IDs and integer labels in a submission CSV."""
    with test_csv.open(newline="") as stream:
        test = list(csv.DictReader(stream))
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError("Submission columns must be exactly id,label")
        rows = list(reader)
    if len(test) != 7994 or len(rows) != len(test):
        raise ValueError("Expected exactly 7,994 test IDs and predictions")
    test_ids = [row.get("id") for row in test]
    if any(not value for value in test_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Test IDs are empty or duplicated")
    if [row.get("id") for row in rows] != test_ids:
        raise ValueError("Submission IDs do not match test.csv in order")
    if any(not (row.get("label") or "").isdigit() or int(row["label"]) > 15 for row in rows):
        raise ValueError("Submission labels must be integers between 0 and 15")


def sync_history(rows: list[dict], history_path: Path) -> None:
    """Update known submission status and scores from Kaggle's response."""
    if not history_path.exists():
        return
    with history_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        history = list(reader)
    remote = {str(row.get("ref")): row for row in rows}
    changed = False
    for entry in history:
        row = remote.get(entry["submission_ref"])
        if row:
            updates = {"status": str(row.get("status", "")).removeprefix("SubmissionStatus."),
                       "public_score": row.get("publicScore") or "",
                       "private_score": row.get("privateScore") or ""}
            for key, value in updates.items():
                if entry[key] != value:
                    entry[key] = value
                    changed = True
    if not changed:
        return
    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    """Validate one experiment, enforce submission limits, and record its Kaggle ref."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        help="Downloaded kernel output containing reports/, checkpoint/, and submission.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--history", type=Path, default=Path("reports/submission_history.csv"))
    parser.add_argument("--sync-only", action="store_true", help="Update scores without submitting")
    parser.add_argument("--ensemble", action="store_true", help="Validate an ensemble evaluation directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate the candidate without submitting")
    parser.add_argument("--minimum-validation", type=float, default=THRESHOLD,
                        help="Additional validation-accuracy gate, e.g. current public #2 score")
    args = parser.parse_args()
    rows = kaggle_submissions()
    sync_history(rows, args.history)
    used = count_today(rows, datetime.now(timezone.utc))
    available = allowed_now()
    print(f"UTC submission count: {used}/{LIMIT}; Kaggle available now: {available}", flush=True)
    scores = [float(row["publicScore"]) for row in rows if row.get("publicScore")]
    if scores:
        print(f"Best public score: {max(scores):.5f}; target: {PUBLIC_TARGET:.5f}", flush=True)
    if args.sync_only:
        return
    if args.experiment_dir is None:
        parser.error("--experiment-dir is required unless --sync-only is specified")
    if (used >= LIMIT or available < 1) and not args.dry_run:
        raise RuntimeError("Daily submission limit reached; stopping")
    if not args.history.exists():
        raise FileNotFoundError("Submission history CSV is missing")
    with args.history.open(newline="") as stream:
        fields = csv.DictReader(stream).fieldnames
    required = {"submission_ref", "submitted_at_utc", "competition", "model", "pretrained_weights",
                "training_setup", "validation_accuracy", "public_score", "private_score", "status",
                "message", "artifact"}
    if fields is None or set(fields) != required:
        raise ValueError("Submission history CSV has an unexpected schema")
    report_path = args.experiment_dir / ("validation_result.json" if args.ensemble else "reports/validation_result.json")
    checkpoint_path = args.experiment_dir / "checkpoint/best.pt"
    candidate = args.experiment_dir / "submission.csv"
    report = json.loads(report_path.read_text())
    score = float(report["best_validation_accuracy"])
    threshold = max(THRESHOLD, args.minimum_validation)
    if not math.isfinite(score) or not math.isfinite(threshold) or not threshold < score <= 1:
        raise ValueError(f"Validation accuracy {score} must be above {threshold:.5f}")
    experiment_id = args.experiment_dir.name
    previous_scores = [float(row["validation_accuracy"]) for row in rows
                       if row.get("validation_accuracy") not in (None, "")]
    for previous_path in Path("reports/experiments").glob("*.json"):
        if previous_path.stem == experiment_id:
            continue
        previous = json.loads(previous_path.read_text())
        if previous.get("status") in {"COMPLETE", "COMPLETED", "SUCCESS"}:
            previous_score = previous.get("best_validation_accuracy")
            if previous_score is not None:
                previous_scores.append(float(previous_score))
    if previous_scores and score <= max(previous_scores):
        raise ValueError(f"Validation accuracy {score:.4f} does not beat the best previous model {max(previous_scores):.4f}")
    if report.get("seed") != 42 or report.get("validation_samples") != 6401 or report.get("validation_samples_used") != 6401:
        raise ValueError("Candidate must use the complete 6,401-sample validation split with seed 42")
    configuration = report.get("configuration", {})
    if args.ensemble:
        members = report.get("checkpoints", [])
        if report.get("model") != "weighted probability ensemble" or not report.get("test_predicted") or len(members) < 2:
            raise ValueError("Ensemble report or test predictions are incomplete")
        weights = [float(member["weight"]) for member in members]
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
            raise ValueError("Invalid ensemble weights")
        architectures = set()
        for member in members:
            path = Path(member["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
                raise ValueError(f"Checkpoint hash does not match ensemble report: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            architecture = checkpoint.get("architecture")
            if (checkpoint.get("seed") != 42 or checkpoint.get("num_classes") != 16
                    or architecture not in {"resnet18", "resnet50", "dit_base", "dit_large", "donut_base"}
                    or int(checkpoint["epoch"]) != int(member["epoch"])
                    or int(checkpoint["image_size"]) != int(member["image_size"])
                    or abs(float(checkpoint["validation_accuracy"]) - float(member["validation_accuracy"])) > 2 / 6401):
                raise ValueError(f"Checkpoint metadata does not match ensemble report: {path}")
            architectures.add(architecture)
        model = "ensemble_" + "_".join(sorted(architectures))
        configuration = {"method": "weighted probability mean", "members": members}
        manifest_checkpoints = {"checkpoints": members}
        description = model + " " + "+".join(
            f"{member['image_size']}x{member['weight']:g}" for member in members)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not math.isclose(float(checkpoint["validation_accuracy"]), score, abs_tol=1e-10):
            raise ValueError("Checkpoint and validation report accuracy differ")
        if int(checkpoint["epoch"]) != int(report["best_epoch"]):
            raise ValueError("Checkpoint and validation report epoch differ")
        if checkpoint.get("seed") != 42 or checkpoint.get("num_classes") != 16:
            raise ValueError("Checkpoint seed or number of classes does not match the experiment")
        if (checkpoint.get("architecture") != configuration.get("architecture") or
                bool(checkpoint.get("preserve_aspect_ratio", False)) !=
                bool(configuration.get("preserve_aspect_ratio", False))):
            raise ValueError("Checkpoint architecture or image transform does not match the validation report")
        model = str(checkpoint["architecture"])
        description = f"{model} epoch {checkpoint['epoch']}"
        manifest_checkpoints = {"checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                                "epoch": checkpoint["epoch"]}
    validate_prediction(candidate, args.data_dir / "test.csv")
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if args.ensemble and report.get("submission_sha256") != candidate_hash:
        raise ValueError("Submission hash does not match ensemble report")
    digest = candidate_hash[:12]
    message = f"{description} val {score:.4f} {digest}"
    if any(row.get("description") == message for row in rows):
        raise RuntimeError("This candidate already appears in Kaggle submission history")
    if args.dry_run:
        print(f"Candidate validated: {message}; UTC quota {used}/{LIMIT}, Kaggle available {available}", flush=True)
        return
    archive = Path("submissions") / f"{model}_{digest}.csv"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.read_bytes() != candidate.read_bytes():
        raise RuntimeError("Candidate archive collision")
    shutil.copyfile(candidate, archive)
    manifest = {**manifest_checkpoints, "validation_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "submission_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "validation_accuracy": score, "model": model}
    archive.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    current_rows = kaggle_submissions()
    if count_today(current_rows, datetime.now(timezone.utc)) >= LIMIT or allowed_now() < 1:
        raise RuntimeError("Submission allowance was exhausted before upload; stopping")
    if any(row.get("description") == message for row in current_rows):
        raise RuntimeError("This candidate appeared in Kaggle submission history before upload")
    command = ["kaggle", "competitions", "submit", "-c", COMPETITION,
               "-f", str(archive), "-m", message]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout, flush=True)
    match = re.search(r"Submission ref:\s*(\d+)", result.stdout)
    if not match:
        raise RuntimeError("Submit returned without a ref; inspect Kaggle before retrying")
    config = configuration
    pretrained_weights = (
        ("Donut-base self-supervised IIT-CDIP+SynthDoG" if model == "donut_base" else
         f"{model.replace('_', '-')} self-supervised IIT-CDIP" if model.startswith("dit_") else "ImageNet")
    )
    record = {"submission_ref": match.group(1), "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
              "competition": COMPETITION, "model": model, "pretrained_weights": pretrained_weights,
              "training_setup": json.dumps(config, sort_keys=True), "validation_accuracy": score,
              "public_score": "", "private_score": "", "status": "PENDING",
              "message": message, "artifact": str(archive)}
    with args.history.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields, lineterminator="\n").writerow(record)
    print(f"Recorded submission ref {match.group(1)} in {args.history}", flush=True)
    sync_history(kaggle_submissions(), args.history)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from error
