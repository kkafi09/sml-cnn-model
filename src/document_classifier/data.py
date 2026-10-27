"""Dataset implementation for the competition's image and CSV layout."""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, default_collate


@dataclass(frozen=True)
class SkippedImage:
    """Identifier for a training image that cannot be decoded."""

    sample_id: str


def load_metadata(data_dir: Path, split: str) -> pd.DataFrame:
    """Load the CSV metadata table for a competition split.

    Args:
        data_dir: Directory containing train.csv or test.csv.
        split: Either ``train`` or ``test``.

    Returns:
        The selected metadata table.

    Raises:
        ValueError: If split is not ``train`` or ``test``.
        FileNotFoundError: If the requested CSV is absent.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    csv_path = Path(data_dir) / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
    return pd.read_csv(csv_path)


def validate_metadata(metadata: pd.DataFrame, split: str) -> None:
    """Validate required competition columns and known label bounds.

    Args:
        metadata: Competition CSV contents.
        split: Either ``train`` or ``test``.

    Raises:
        ValueError: If required columns, IDs, filenames, or labels are invalid.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    required = {"id", "filename"}
    if split == "train":
        required.add("label")
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing columns in {split}.csv: {sorted(missing)}")
    if metadata[["id", "filename"]].isna().any().any():
        raise ValueError(f"{split}.csv contains an empty id or filename")
    if metadata["id"].duplicated().any():
        raise ValueError(f"{split}.csv contains duplicate IDs")
    if split == "train":
        labels = pd.to_numeric(metadata["label"], errors="coerce")
        if labels.isna().any() or not labels.between(0, 15).all():
            raise ValueError("Training labels must be integers between 0 and 15")


class DocumentImageDataset(Dataset):
    """Read document images and labels from competition metadata."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        data_dir: Path,
        transform=None,
        include_labels: bool = True,
    ) -> None:
        """Create a dataset backed by a metadata DataFrame.

        Args:
            metadata: Rows from train.csv or test.csv.
            data_dir: Root data directory; filenames are relative to this path.
            transform: Optional image transform, usually from torchvision.
            include_labels: Return labels when true; return IDs when false.
        """
        required = {"id", "filename"}
        if include_labels:
            required.add("label")
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
        self.metadata = metadata.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.include_labels = include_labels

    def __len__(self) -> int:
        """Return the number of metadata rows."""
        return len(self.metadata)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, int] | tuple[Tensor, str] | SkippedImage:
        """Load one RGB page, returning a marker for a corrupt training image."""
        row = self.metadata.iloc[index]
        relative_path = Path(str(row["filename"]))
        image_path = self.data_dir / relative_path
        if not image_path.is_file() and relative_path.parts:
            image_path = self.data_dir / relative_path.parts[0] / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image from metadata not found: {image_path}")
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except (OSError, ValueError) as error:
            sample_id = str(row["id"])
            if self.include_labels:
                return SkippedImage(sample_id)
            raise ValueError(f"Cannot decode test image {sample_id}: {image_path}") from error
        if self.transform is not None:
            image = self.transform(image)
        if self.include_labels:
            return image, int(row["label"])
        return image, str(row["id"])


def collate_document_batch(batch):
    """Batch valid samples and return IDs of corrupt images to the caller.

    Args:
        batch: Dataset outputs containing image/label pairs or skip markers.

    Returns:
        A tuple of collated images, labels, and skipped sample IDs.
    """
    valid_samples = [sample for sample in batch if not isinstance(sample, SkippedImage)]
    skipped_ids = [sample.sample_id for sample in batch if isinstance(sample, SkippedImage)]
    if not valid_samples:
        return None, None, skipped_ids
    images, labels = default_collate(valid_samples)
    return images, labels, skipped_ids
