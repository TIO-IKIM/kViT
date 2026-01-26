# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import unittest
import torch
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from model import image_classification


class TestImageClassification(unittest.TestCase):
    def test_get_resnet_model(self):
        # Use pretrained=False to avoid downloading weights during tests
        model = image_classification.get_resnet_model(
            num_classes=2, features_only=True, pretrained=False
        )
        self.assertIsNotNone(model)
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        self.assertIsInstance(output, list)

    def test_get_efficientnet_model(self):
        model = image_classification.get_efficientnet_model(
            num_classes=2, features_only=True, pretrained=False
        )
        self.assertIsNotNone(model)
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        self.assertIsInstance(output, list)

    def test_get_vit_model(self):
        model = image_classification.get_vit_model(
            num_classes=2, features_only=True, pretrained=False
        )
        self.assertIsNotNone(model)
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        self.assertIsInstance(output, list)

    def test_mil_model_vit(self):
        model = image_classification.MILModel(num_classes=2, model="vit", device="cpu")
        self.assertEqual(model.num_features, 256)

        bag = torch.randn(2, 1, 224, 224)
        output = model(bag)
        self.assertEqual(output.shape, (2,))

    def test_mil_model_resnet(self):
        model = image_classification.MILModel(
            num_classes=2, model="resnet", device="cpu"
        )
        bag = torch.randn(2, 1, 224, 224)
        output = model(bag)
        self.assertEqual(output.shape, (2,))

    def test_classification_model_2d_vit(self):
        model = image_classification.ClassificationModel2D(num_classes=2, model="vit")
        x = torch.randn(2, 1, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))

    def test_classification_model_2d_resnet(self):
        model = image_classification.ClassificationModel2D(
            num_classes=2, model="resnet"
        )
        x = torch.randn(2, 1, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))

    def test_classification_model_2d_efficientnet(self):
        model = image_classification.ClassificationModel2D(
            num_classes=2, model="efficientnet"
        )
        x = torch.randn(2, 1, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
