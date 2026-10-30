"""Neural network models for scanned document pages."""

from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class DocumentCNN(nn.Module):
    """ImageNet-initialized ResNet that predicts document classes."""

    def __init__(self, num_classes: int = 16, pretrained: bool = True,
                 architecture: str = "resnet18") -> None:
        """Initialize an ImageNet ResNet or document-pretrained DiT classifier.

        Args:
            num_classes: Number of document categories to predict.
            pretrained: Start from ImageNet weights when true.
            architecture: ``resnet18``, ``resnet50``, or ``dit_base``.
        """
        super().__init__()
        self.architecture = architecture
        if architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
        elif architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
        elif architecture == "dit_base":
            try:
                from transformers import BeitConfig, BeitForImageClassification
            except ImportError as error:
                raise ImportError("Install transformers to use the DiT-base model") from error
            if pretrained:
                self.backbone = BeitForImageClassification.from_pretrained(
                    "microsoft/dit-base",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            else:
                config = BeitConfig.from_pretrained("microsoft/dit-base")
                config.num_labels = num_classes
                config.id2label = {index: str(index) for index in range(num_classes)}
                config.label2id = {str(index): index for index in range(num_classes)}
                self.backbone = BeitForImageClassification(config)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if architecture in {"resnet18", "resnet50"}:
            input_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(input_features, num_classes)

    def forward(self, images):
        """Compute unnormalized class scores for a batch of RGB images."""
        if self.architecture == "dit_base":
            return self.backbone(pixel_values=images).logits
        return self.backbone(images)
