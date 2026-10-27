"""Image transforms for training and validation of document classifiers."""

from torchvision import transforms


def build_image_transform(image_size: int = 224, training: bool = False):
    """Build a transform that resizes and normalizes scanned pages.

    Args:
        image_size: Height and width used by the model input.
        training: Add a small rotation augmentation for the training split.

    Returns:
        A torchvision composition converting PIL images to normalized tensors.
    """
    operations = [transforms.Resize((image_size, image_size))]
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
