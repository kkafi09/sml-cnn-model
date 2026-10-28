"""Cache test probabilities in small resumable chunks for ensemble inference."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader, Subset

import evaluate_ensemble
from document_classifier.data import DocumentImageDataset, load_metadata, validate_metadata
from document_classifier.model import DocumentCNN
from document_classifier.transforms import build_image_transform


def main():
    """Write atomic chunk files and assemble the ensemble evaluator's cache format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=800)
    args = parser.parse_args()
    if (args.chunk_size < 1 or args.batch_size < 1 or args.workers < 0
            or args.chunk_size % args.batch_size):
        parser.error("Chunk size must be a positive multiple of batch size; workers must be nonnegative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cpu")
    metadata = load_metadata(args.data_dir, "test")
    validate_metadata(metadata, "test")
    if len(metadata) != 7994:
        raise ValueError("Unexpected test sample count")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != 42 or checkpoint.get("num_classes") != 16:
        raise ValueError("Unexpected checkpoint seed or class count")
    root = Path(evaluate_ensemble.__file__).resolve().parents[1]
    key = hashlib.sha256()
    for payload in (args.checkpoint.read_bytes(), metadata[["id", "filename"]].to_csv(index=False).encode(),
                    str(args.data_dir.resolve()).encode(), str(device).encode(),
                    torch.__version__.encode(), torchvision.__version__.encode(), pillow_version.encode(),
                    Path(evaluate_ensemble.__file__).read_bytes(),
                    *(root.joinpath("src/document_classifier", name).read_bytes()
                      for name in ("data.py", "model.py", "transforms.py"))):
        key.update(hashlib.sha256(payload).digest())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    digest = key.hexdigest()
    final = args.cache_dir / f"{digest}.npz"
    if final.exists():
        with np.load(final, allow_pickle=False) as data:
            values = data["probabilities"]
        if values.shape != (len(metadata), 16) or not np.isfinite(values).all():
            raise ValueError("Existing full cache is invalid")
        print(f"Full cache already exists: {final}", flush=True)
        return
    model = DocumentCNN(pretrained=False, architecture=checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_image_transform(int(checkpoint["image_size"]),
                                      preserve_aspect_ratio=bool(checkpoint.get("preserve_aspect_ratio", False)))
    dataset = DocumentImageDataset(metadata, args.data_dir, transform=transform, include_labels=False)
    expected = metadata["id"].astype(str).tolist()
    chunks = []
    for start in range(0, len(metadata), args.chunk_size):
        end = min(start + args.chunk_size, len(metadata))
        path = args.cache_dir / f"{digest}.{start}-{end}.part.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                values = data["probabilities"]
            if values.shape != (end - start, 16) or not np.isfinite(values).all():
                raise ValueError(f"Invalid chunk: {path}")
            print(f"Reused test rows {start}-{end}", flush=True)
        else:
            loader = DataLoader(Subset(dataset, range(start, end)), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
            outputs, ids = [], []
            with torch.inference_mode():
                for images, batch_ids in loader:
                    outputs.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
                    ids.extend(batch_ids)
            if ids != expected[start:end]:
                raise ValueError("Test IDs differ from metadata order")
            values = np.concatenate(outputs)
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, probabilities=values)
            temporary.replace(path)
            print(f"Cached test rows {start}-{end}", flush=True)
        chunks.append(values)
    probabilities = np.concatenate(chunks)
    if probabilities.shape != (len(metadata), 16) or not np.isfinite(probabilities).all():
        raise ValueError("Assembled test probabilities are invalid")
    temporary = final.with_suffix(".partial.npz")
    np.savez_compressed(temporary, probabilities=probabilities)
    temporary.replace(final)
    print(f"Assembled complete cache: {final}", flush=True)


if __name__ == "__main__":
    main()
