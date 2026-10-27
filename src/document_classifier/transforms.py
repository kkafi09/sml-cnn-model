"""Image transforms for training and validation of document classifiers."""

from PIL import ImageOps
from torchvision import transforms


class PadToSquare:
    """Pad a document page with white pixels while preserving its aspect ratio."""

    def __call__(self, image):
        """Center a PIL image on a white square canvas."""
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(image, border=(left, top, side - width - left, side - height - top), fill="white")


def build_image_transform(
    image_size: int = 224, training: bool = False, preserve_aspect_ratio: bool = False
):
    """Build a transform that resizes and normalizes scanned pages.

    Args:
        image_size: Height and width used by the model input.
        training: Add a small rotation augmentation for the training split.
        preserve_aspect_ratio: Pad pages to square before resizing, avoiding page distortion.

    Returns:
        A torchvision composition converting PIL images to normalized tensors.
    """
    operations = []
    if preserve_aspect_ratio:
        operations.append(PadToSquare())
    operations.append(transforms.Resize((image_size, image_size)))
    if training:
        operations.append(transforms.RandomRotation(degrees=2))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)
