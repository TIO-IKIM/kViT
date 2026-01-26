# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import torch.nn as nn
import torch
import timm
from typing import Optional


class MILModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        model: str = "vit",
        num_features: Optional[int] = None,
        hidden_dim: int = 128,
        device="cpu",
    ):
        super(MILModel, self).__init__()

        self.device = device
        # Feature Extractor
        assert model in [
            "vit",
            "efficientnet",
            "resnet",
        ], "Model must be one of 'vit', 'efficientnet', or 'resnet'"

        if model == "vit":
            if num_features is None:
                num_features = 256
                self.extractor = get_vit_model(
                    num_classes=num_classes, features_only=True
                )
        elif model == "efficientnet":
            if num_features is None:
                num_features = 320
                self.extractor = get_efficientnet_model(
                    num_classes=num_classes, features_only=True
                )
        elif model == "resnet":
            if num_features is None:
                num_features = 256
                self.extractor = get_resnet_model(
                    num_classes=num_classes, features_only=True
                )
        self.num_classes = num_classes
        self.num_features = num_features

        # Aggregation Layer
        self.aggregation = nn.Sequential(
            nn.Linear(num_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, bag):
        bag = bag.to(self.device)

        instance_features = self.extractor(bag)[0].view(-1, self.num_features)

        # Calculate attention weights for each instance
        attention_weights = self.aggregation(instance_features)
        attention_weights = torch.softmax(attention_weights, dim=0)

        # Apply attention to get the aggregated bag-level feature vector
        bag_feature = torch.sum(attention_weights * instance_features, dim=0)

        # Classify the bag
        prediction = self.classifier(bag_feature)

        return prediction


class ClassificationModel2D(torch.nn.Module):
    def __init__(self, num_classes: int, model: str = "vit"):
        """
        Initializes a 2D classification model.

        Args:
            num_classes (int): The number of output classes.
            model (str): The type of model to use. Options are 'vit', 'efficientnet', or 'resnet'.
        """
        super(ClassificationModel2D, self).__init__()

        self.num_classes = num_classes

        assert model in [
            "vit",
            "efficientnet",
            "resnet",
        ], "Model must be one of 'vit', 'efficientnet', or 'resnet'"

        if model == "vit":
            self.extractor = get_vit_model(num_classes=num_classes, features_only=False)
        elif model == "efficientnet":
            self.extractor = get_efficientnet_model(
                num_classes=num_classes, features_only=False
            )
        elif model == "resnet":
            self.extractor = get_resnet_model(
                num_classes=num_classes, features_only=False
            )

    def forward(self, x):
        """
        Forward pass for a single 2D image or a batch of images.

        Args:
            x (torch.Tensor): A tensor representing a single image or a batch of images.
                               Shape could be [batch_size, channels, height, width].

        Returns:
            torch.Tensor: The model's prediction (logits).
        """

        return self.extractor(x)


def get_resnet_model(
    num_classes: int, features_only: bool = False, pretrained: bool = False
):
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        features_only=features_only,
        in_chans=1,
        out_indices=[-1] if features_only else None,  # Use the last layer's features
        num_classes=num_classes,
    )
    return model


def get_efficientnet_model(
    num_classes, features_only: bool = False, pretrained: bool = False
):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        features_only=features_only,
        in_chans=1,
        out_indices=[-1] if features_only else None,  # Use the last layer's features
        num_classes=num_classes,
    )
    return model


def get_vit_model(num_classes, features_only: bool = False, pretrained: bool = False):
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        features_only=features_only,
        in_chans=1,
        out_indices=[-1] if features_only else None,  # Use the last layer's features
        num_classes=num_classes,
    )
    return model
