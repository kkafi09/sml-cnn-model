"""Create and validate Kaggle submission files."""

from pathlib import Path
from collections.abc import Sequence

import pandas as pd


def write_submission(
    sample_ids: Sequence[str], labels: Sequence[int], output_path: Path
) -> pd.DataFrame:
    """Validate predictions and write a CSV with columns ``id,label``.

    Args:
        sample_ids: Test IDs in the exact order used for inference.
        labels: Predicted integer labels from 0 through 15.
        output_path: Destination CSV path.

    Returns:
        The submission DataFrame that was written.

    Raises:
        ValueError: If IDs and predictions differ in length or labels are invalid.
    """
    if len(sample_ids) != len(labels):
        raise ValueError("sample_ids and labels must have equal length")
    if not all(isinstance(label, int) and 0 <= label <= 15 for label in labels):
        raise ValueError("All predicted labels must be integers from 0 through 15")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    frame = pd.DataFrame({"id": sample_ids, "label": labels})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
