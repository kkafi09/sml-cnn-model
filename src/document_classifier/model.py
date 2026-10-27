"""ResNet transfer-learning models for scanned document pages."""

from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class DocumentCNN(nn.Module):
    """ImageNet-initialized ResNet that predicts document classes."""

    def __init__(self, num_classes: int = 16, pretrained: bool = True,
                 architecture: str = "resnet18") -> None:
        """Initialize the requested ResNet and replace its final classification layer.

        Args:
            num_classes: Number of document categories to predict.
            pretrained: Start from ImageNet weights when true.
            architecture: Either ``resnet18`` or ``resnet50``.
        """
        super().__init__()
        if architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
        elif architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, num_classes)

    def forward(self, images):
        """Compute unnormalized class scores for a batch of RGB images."""
        return self.backbone(images)
