"""Build a self-contained Kaggle GPU script from the checked-in training code."""

import argparse
import base64
import io
import json
from pathlib import Path
import zipfile


def main() -> None:
    """Package training and inference code into a private Kaggle kernel staging folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--slug", default="document-classification-resnet18-improved")
    parser.add_argument("--output", type=Path, default=Path("kaggle_kernel"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--architecture", choices=("resnet18", "resnet50", "dit_base", "dit_large", "donut_base"), default="resnet18")
    parser.add_argument("--unfreeze-from", choices=("layer2", "layer3", "layer4"), default="layer3")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--preserve-aspect-ratio", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((root / "src" / "document_classifier").glob("*.py")):
            bundle.write(path, path.relative_to(root))
        for filename in ("train.py", "predict.py"):
            path = root / "scripts" / filename
            bundle.write(path, path.relative_to(root))
    encoded = base64.b64encode(archive.getvalue()).decode("ascii")
    runner = '''"""Train the selected model and generate a competition prediction file."""
import base64
import io
from pathlib import Path
import subprocess
import sys
import zipfile
import torch

print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a GPU; stopping before training")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
if ARCHITECTURE.startswith("dit_") or ARCHITECTURE == "donut_base":
    try:
        import transformers
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.46.3"], check=True)
source = Path("/kaggle/working/sml_source")
source.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
    archive.extractall(source)
sys.path.insert(0, str(source / "src"))
import os
os.environ["PYTHONPATH"] = str(source / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
root = Path("/kaggle/input/document-image-classification-sd234504")
if not (root / "train.csv").is_file():
    input_root = Path("/kaggle/input")
    candidates = []
    for pattern in ("*/train.csv", "*/*/train.csv", "*/*/*/train.csv"):
        candidates.extend(input_root.glob(pattern))
    matches = [p.parent for p in candidates if (p.parent / "test.csv").is_file()]
    if len(matches) != 1:
        mounts = [str(path) for path in input_root.iterdir()] if input_root.is_dir() else "missing /kaggle/input"
        raise RuntimeError(f"Cannot uniquely locate competition CSVs: {matches}; input mounts: {mounts}")
    root = matches[0]
work = Path("/kaggle/working")
subprocess.run([sys.executable, str(source / "scripts/train.py"), "--data-dir", str(root),
    "--checkpoint-dir", str(work / "checkpoint"), "--report-dir", str(work / "reports"),
    "--epochs", str(EPOCHS), "--patience", str(PATIENCE), "--batch-size", str(BATCH_SIZE),
    "--image-size", str(IMAGE_SIZE), "--workers", "2", "--seed", "42", "--architecture", ARCHITECTURE,
    "--unfreeze-from", UNFREEZE_FROM, "--learning-rate", str(LEARNING_RATE),
    "--label-smoothing", str(LABEL_SMOOTHING)] +
    (["--preserve-aspect-ratio"] if PRESERVE_ASPECT_RATIO else []), check=True)
subprocess.run([sys.executable, str(source / "scripts/predict.py"), "--data-dir", str(root),
    "--checkpoint", str(work / "checkpoint/best.pt"), "--output", str(work / "submission.csv"),
    "--batch-size", "32", "--workers", "2"], check=True)
'''
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = (f"BUNDLE = {encoded!r}\nEPOCHS = {args.epochs}\nPATIENCE = {args.patience}\n"
              f"ARCHITECTURE = {args.architecture!r}\nUNFREEZE_FROM = {args.unfreeze_from!r}\n"
              f"IMAGE_SIZE = {args.image_size}\nBATCH_SIZE = {args.batch_size}\n"
              f"LEARNING_RATE = {args.learning_rate!r}\n"
              f"LABEL_SMOOTHING = {args.label_smoothing!r}\n"
              f"PRESERVE_ASPECT_RATIO = {args.preserve_aspect_ratio!r}\n")
    (args.output / "run.py").write_text(prefix + runner)
    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": args.slug.replace("-", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": ["document-image-classification-sd234504"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared kernel {metadata['id']} in {args.output}")


if __name__ == "__main__":
    main()
