"""ResNet-18 transfer-learning baseline for scanned document pages."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class DocumentCNN(nn.Module):
    """ImageNet-initialized ResNet-18 that predicts 16 document classes."""

    def __init__(self, num_classes: int = 16, pretrained: bool = True) -> None:
        """Initialize ResNet-18 and replace its final classification layer.

        Args:
            num_classes: Number of document categories to predict.
            pretrained: Start from ImageNet weights when true.
        """
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, num_classes)

    def forward(self, images):
        """Compute unnormalized class scores for a batch of RGB images."""
        return self.backbone(images)
