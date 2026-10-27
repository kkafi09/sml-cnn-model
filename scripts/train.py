"""Train a reproducible ResNet model and select by validation accuracy."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from document_classifier.data import (
    DocumentImageDataset,
    collate_document_batch,
    load_metadata,
    validate_metadata,
)
from document_classifier.model import DocumentCNN
from document_classifier.transforms import build_image_transform


def run_epoch(
    model, loader, criterion, device, optimizer=None
) -> tuple[float, float, list[str]]:
    """Run one training or validation epoch and return loss and accuracy.

    Args:
        model: Neural network to evaluate or update.
        loader: DataLoader yielding image and label batches.
        criterion: Classification loss function.
        device: CPU or CUDA device used for computation.
        optimizer: Optimizer for training; omit it for validation.

    Returns:
        Mean loss, categorical accuracy, and IDs of unreadable images.
    """
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    skipped_ids = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels, batch_skipped_ids in loader:
            skipped_ids.extend(batch_skipped_ids)
            if images is None:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No readable images were available for this epoch")
    return loss_sum / sample_count, correct / sample_count, skipped_ids


def main() -> None:
    """Train the baseline and save the checkpoint with best validation accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--unfreeze-from", choices=("layer3", "layer4"), default="layer4")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--preserve-aspect-ratio", action="store_true")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without improvement; 0 disables early stopping")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 0 or args.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive; patience must be nonnegative")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frame = load_metadata(args.data_dir, "train")
    validate_metadata(train_frame, "train")
    train_rows, validation_rows = train_test_split(
        train_frame,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_frame["label"].astype(int),
    )
    train_dataset = DocumentImageDataset(
        train_rows,
        args.data_dir,
        transform=build_image_transform(args.image_size, training=True,
                                        preserve_aspect_ratio=args.preserve_aspect_ratio),
    )
    validation_dataset = DocumentImageDataset(
        validation_rows,
        args.data_dir,
        transform=build_image_transform(args.image_size,
                                        preserve_aspect_ratio=args.preserve_aspect_ratio),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_document_batch, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=collate_document_batch,
        **loader_options,
    )

    model = DocumentCNN(pretrained=True, architecture=args.architecture).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad = True
    if args.unfreeze_from == "layer3":
        for parameter in model.backbone.layer3.parameters():
            parameter.requires_grad = True
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=0
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    corrupt_ids = set()
    train_corrupt_ids = set()
    validation_corrupt_ids = set()
    epochs_without_improvement = 0
    print(
        f"Device: {device}; train: {len(train_dataset)}; "
        f"validation: {len(validation_dataset)}; classes: 16",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_corrupt = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy, validation_corrupt = run_epoch(
            model, validation_loader, criterion, device
        )
        train_corrupt_ids.update(train_corrupt)
        validation_corrupt_ids.update(validation_corrupt)
        corrupt_ids.update(train_corrupt_ids)
        corrupt_ids.update(validation_corrupt_ids)
        scheduler.step(validation_accuracy)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} "
            f"corrupt_images={len(corrupt_ids)}",
            flush=True,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 16,
                    "image_size": args.image_size,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "architecture": args.architecture,
                    "unfreeze_from": args.unfreeze_from,
                    "learning_rate": args.learning_rate,
                    "preserve_aspect_ratio": args.preserve_aspect_ratio,
                },
                args.checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        with (args.report_dir / "training_history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs", flush=True)
            break

    with (args.report_dir / "training_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (args.report_dir / "validation_result.json").open("w") as file:
        json.dump(
            {
                "best_validation_accuracy": best_accuracy,
                "best_epoch": max(history, key=lambda row: row["validation_accuracy"])["epoch"],
                "seed": args.seed,
                "train_samples": len(train_dataset),
                "train_samples_used": len(train_dataset) - len(train_corrupt_ids),
                "validation_samples": len(validation_dataset),
                "validation_samples_used": len(validation_dataset) - len(validation_corrupt_ids),
                "device": str(device),
                "model": f"{args.architecture} ImageNet initialization; {args.unfreeze_from} onward fine-tuned",
                "configuration": {"epochs": args.epochs, "batch_size": args.batch_size,
                    "image_size": args.image_size, "learning_rate": args.learning_rate,
                    "unfreeze_from": args.unfreeze_from, "patience": args.patience,
                    "architecture": args.architecture,
                    "preserve_aspect_ratio": args.preserve_aspect_ratio},
                "corrupt_image_count": len(corrupt_ids),
                "corrupt_image_ids": sorted(corrupt_ids),
                "corrupt_train_image_ids": sorted(train_corrupt_ids),
                "corrupt_validation_image_ids": sorted(validation_corrupt_ids),
            },
            file,
            indent=2,
        )
    if corrupt_ids:
        with (args.report_dir / "corrupt_image_ids.txt").open("w") as file:
            file.write("\n".join(sorted(corrupt_ids)) + "\n")
    print(f"Best validation accuracy: {best_accuracy:.4f}; checkpoint saved.", flush=True)


if __name__ == "__main__":
    main()
