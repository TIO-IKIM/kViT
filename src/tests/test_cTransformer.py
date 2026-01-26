# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
 
import unittest
import torch
import sys
from pathlib import Path

# Add src to sys.path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from model.cTransformer import (
    LearnablePositionalEncoding,
    ComplexPatchEmbed,
    RadialPatchEmbed,
    ModalityAwareAggregator,
    KSpaceTransformer,
    KSpace2DTransformer,
)


class TestLearnablePositionalEncoding(unittest.TestCase):
    def test_forward(self):
        embed_dim = 32
        seq_len = 10
        batch_size = 2

        pos_encoder = LearnablePositionalEncoding(embed_dim, max_len=20)
        x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.cfloat)

        output = pos_encoder(x)

        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))
        self.assertTrue(output.is_complex())


class TestComplexPatchEmbed(unittest.TestCase):
    def test_forward(self):
        k_space_dim = 64
        patch_size = 8
        embed_dim = 32
        batch_size = 2

        embedder = ComplexPatchEmbed(k_space_dim, patch_size, embed_dim)
        x = torch.randn(batch_size, 1, k_space_dim, k_space_dim, dtype=torch.cfloat)

        output = embedder(x)

        num_patches = (k_space_dim // patch_size) ** 2
        self.assertEqual(output.shape, (batch_size, num_patches, embed_dim))
        self.assertTrue(output.is_complex())


class TestRadialPatchEmbed(unittest.TestCase):
    def test_forward(self):
        k_space_dim = 64
        num_rings = 4
        max_points = 100
        embed_dim = 32
        batch_size = 2

        embedder = RadialPatchEmbed(k_space_dim, num_rings, max_points, embed_dim)
        x = torch.randn(batch_size, 1, k_space_dim, k_space_dim, dtype=torch.cfloat)

        output = embedder(x)

        self.assertEqual(output.shape, (batch_size, num_rings, embed_dim))
        self.assertTrue(output.is_complex())

    def test_visualize(self):
        k_space_dim = 64
        num_rings = 4
        max_points = 100
        embed_dim = 32
        batch_size = 2

        embedder = RadialPatchEmbed(k_space_dim, num_rings, max_points, embed_dim)
        x = torch.randn(batch_size, 1, k_space_dim, k_space_dim, dtype=torch.cfloat)

        embeddings, raw_patches = embedder(x, visualize=True)

        self.assertEqual(embeddings.shape, (batch_size, num_rings, embed_dim))
        self.assertEqual(
            raw_patches.shape, (batch_size, num_rings, 1, k_space_dim, k_space_dim)
        )


class TestModalityAwareAggregator(unittest.TestCase):
    def test_forward(self):
        feature_dim = 32
        num_modalities = 3
        num_slices = 5

        aggregator = ModalityAwareAggregator(feature_dim, num_modalities)
        instance_features = torch.randn(num_slices, feature_dim)
        modality_ids = torch.randint(0, num_modalities, (num_slices,))

        output = aggregator(instance_features, modality_ids)

        self.assertEqual(output.shape, (num_slices, 1))


class TestKSpaceTransformer(unittest.TestCase):
    def setUp(self):
        self.k_space_dim = 64
        self.embed_dim = 32
        self.num_heads = 4
        self.num_layers = 2
        self.ff_dim = 64
        self.num_classes = 2
        self.batch_size = 2

    def test_forward(self):
        model = KSpaceTransformer(
            self.k_space_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
            self.num_classes,
        )
        x = torch.randn(
            self.batch_size, 1, self.k_space_dim, self.k_space_dim, dtype=torch.cfloat
        )

        output = model(x)

        self.assertEqual(output.shape, (1, self.num_classes))  # Bag level output

    def test_forward_instance_label(self):
        model = KSpaceTransformer(
            self.k_space_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
            self.num_classes,
            instance_label=True,
        )
        x = torch.randn(
            self.batch_size, 1, self.k_space_dim, self.k_space_dim, dtype=torch.cfloat
        )

        output, instance_preds = model(x)

        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertEqual(instance_preds.shape, (self.batch_size, self.num_classes))

    def test_forward_modality_embed(self):
        model = KSpaceTransformer(
            self.k_space_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
            self.num_classes,
            modality_embed=True,
        )
        x = torch.randn(
            self.batch_size, 1, self.k_space_dim, self.k_space_dim, dtype=torch.cfloat
        )
        modality_ids = torch.randint(0, 3, (self.batch_size,))

        output = model(x, modality_ids=modality_ids)

        self.assertEqual(output.shape, (1, self.num_classes))


class TestKSpace2DTransformer(unittest.TestCase):
    def setUp(self):
        self.k_space_dim = 64
        self.embed_dim = 32
        self.num_heads = 4
        self.num_layers = 2
        self.ff_dim = 64
        self.num_classes = 2
        self.batch_size = 2

    def test_forward_kspace(self):
        model = KSpace2DTransformer(
            self.k_space_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
            self.num_classes,
            kspace=True,
        )
        x = torch.randn(
            self.batch_size, 1, self.k_space_dim, self.k_space_dim, dtype=torch.cfloat
        )

        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_forward_image(self):
        model = KSpace2DTransformer(
            self.k_space_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
            self.num_classes,
            kspace=False,
        )
        x = torch.randn(
            self.batch_size, 1, self.k_space_dim, self.k_space_dim, dtype=torch.cfloat
        )

        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_forward_hybrid(self):
        model = KSpace2DTransformer(
            self.k_space_dim,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
            self.num_classes,
            hybrid=True,
        )
        # Hybrid expects 2 channels: kspace and image
        x = torch.randn(
            self.batch_size, 2, self.k_space_dim, self.k_space_dim, dtype=torch.cfloat
        )

        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


if __name__ == "__main__":
    unittest.main()
