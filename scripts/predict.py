"""Run test inference with the selected checkpoint and write a submission CSV."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from document_classifier.data import DocumentImageDataset, load_metadata, validate_metadata
from document_classifier.model import DocumentCNN
from document_classifier.submission import write_submission
from document_classifier.transforms import build_image_transform


def main() -> None:
    """Load a trained model, predict test rows, and save Kaggle submission CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_frame = load_metadata(args.data_dir, "test")
    validate_metadata(test_frame, "test")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 224))
    model = DocumentCNN(pretrained=False, architecture=checkpoint.get("architecture", "resnet18")).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = DocumentImageDataset(
        test_frame,
        args.data_dir,
        transform=build_image_transform(image_size),
        include_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    predicted_labels = []
    sample_ids = []
    with torch.inference_mode():
        for images, batch_ids in loader:
            logits = model(images.to(device, non_blocking=True))
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            sample_ids.extend(batch_ids)
    submission = write_submission(sample_ids, predicted_labels, args.output)
    print(
        f"Wrote {len(submission)} predictions to {args.output}; "
        f"label range: {submission['label'].min()}–{submission['label'].max()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
