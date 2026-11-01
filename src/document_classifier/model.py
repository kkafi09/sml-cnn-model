"""Neural network models for scanned document pages."""

from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class DocumentCNN(nn.Module):
    """Image classifier using ResNet or document-pretrained Transformer backbones."""

    def __init__(self, num_classes: int = 16, pretrained: bool = True,
                 architecture: str = "resnet18") -> None:
        """Initialize a ResNet, DiT, or Donut document-image classifier.

        Args:
            num_classes: Number of document categories to predict.
            pretrained: Start from ImageNet weights when true.
            architecture: ``resnet18``, ``resnet50``, ``dit_base``, ``dit_large``, or ``donut_base``.
        """
        super().__init__()
        self.architecture = architecture
        if architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
        elif architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
        elif architecture in {"dit_base", "dit_large"}:
            try:
                from transformers import BeitConfig, BeitForImageClassification
            except ImportError as error:
                raise ImportError("Install transformers to use the DiT-base model") from error
            if pretrained:
                self.backbone = BeitForImageClassification.from_pretrained(
                    "microsoft/dit-large" if architecture == "dit_large" else "microsoft/dit-base",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            else:
                model_name = "microsoft/dit-large" if architecture == "dit_large" else "microsoft/dit-base"
                config = BeitConfig.from_pretrained(model_name)
                config.num_labels = num_classes
                config.id2label = {index: str(index) for index in range(num_classes)}
                config.label2id = {str(index): index for index in range(num_classes)}
                self.backbone = BeitForImageClassification(config)
        elif architecture == "donut_base":
            try:
                from transformers import DonutSwinModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
            except ImportError as error:
                raise ImportError("Install transformers to use the Donut-base model") from error
            model_name = "naver-clova-ix/donut-base"
            if pretrained:
                document_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.backbone = document_model.encoder
                hidden_size = document_model.config.encoder.hidden_size
                del document_model
            else:
                config = VisionEncoderDecoderConfig.from_pretrained(model_name)
                self.backbone = DonutSwinModel(config.encoder)
                hidden_size = config.encoder.hidden_size
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if architecture in {"resnet18", "resnet50"}:
            input_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(input_features, num_classes)

    def forward(self, images):
        """Compute unnormalized class scores for a batch of RGB images."""
        if self.architecture in {"dit_base", "dit_large"}:
            return self.backbone(pixel_values=images).logits
        if self.architecture == "donut_base":
            outputs = self.backbone(pixel_values=images)
            features = outputs.pooler_output
            if features is None:
                features = outputs.last_hidden_state.mean(dim=1)
            return self.classifier(features)
        return self.backbone(images)
