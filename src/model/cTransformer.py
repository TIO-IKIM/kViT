# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import torch
import torch.nn as nn
import model.complex_layer.spectral_blocks as S
from model.complex_layer.spectral_layer import ComplexLayerNorm


class ComplexRotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE) natively for complex numbers.

    Standard RoPE rotates pairs of real numbers. For complex numbers z = r * e^(i*theta),
    RoPE is simply element-wise multiplication by a frequency phasor e^(i*pos*freq).
    This is mathematically cleaner and more efficient than the real-valued equivalent.
    """

    def __init__(self, embed_dim, max_len=5000, base=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Calculate frequencies: theta_j = 10000^(-2(j-1)/d)
        # Note: We only need embed_dim frequencies because complex numbers are single entities here
        inv_freq = 1.0 / (base ** (torch.arange(0, embed_dim, 1).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache the phasors
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, embed_dim)

        # Create complex phasor: e^(i * freq * pos)
        # Shape: (1, seq_len, embed_dim)
        self.phasor_cache = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: Complex tensor (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)

        # Dynamically update cache if sequence length exceeds current cache
        if seq_len > self.phasor_cache.size(1):
            self._build_cache(seq_len)
            self.phasor_cache = self.phasor_cache.to(x.device)

        # Apply rotation via complex multiplication
        # We slice the cache to the current sequence length
        phasor = self.phasor_cache[:, :seq_len, :].to(x.device)

        return x * phasor


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for complex-valued embeddings, initialized similarly
    to standard Vision Transformer implementations.
    """

    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        # Create a single complex parameter for the positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, embed_dim, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        """
        Adds the learnable positional embedding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor with positional information added.
        """
        # Add the positional embedding to the input tensor x
        # self.pos_embedding is sliced to match the sequence length of x
        pos_emb = self.pos_embedding[:, : x.size(1), :]

        return x + pos_emb


class ComplexPatchEmbed(nn.Module):
    """
    Groups k-space into patches and embeds them.
    This replaces the simple linear embedding layer.
    """

    def __init__(self, k_space_dim, patch_size, embed_dim):
        super().__init__()

        self.patch_size = patch_size

        # Calculate the number of patches, which will be the new sequence length
        self.num_patches = (k_space_dim // patch_size) * (k_space_dim // patch_size)

        # The projection layer takes a flattened patch of size (patch_size * patch_size)
        # and projects it into the embedding dimension (embed_dim).
        self.proj = nn.Linear(patch_size * patch_size, embed_dim, dtype=torch.cfloat)

    def forward(self, x):
        # x shape: (batch_size, k_dim, k_dim)
        batch_size, _, h, w = x.shape

        # Reshape into patches: (batch_size, num_patches_h, num_patches_w, patch_size, patch_size)
        x = x.view(
            batch_size,
            h // self.patch_size,
            self.patch_size,
            w // self.patch_size,
            self.patch_size,
        ).permute(0, 1, 3, 2, 4)

        # Flatten patches: (batch_size, num_patches, patch_size * patch_size)
        x = x.reshape(batch_size, self.num_patches, -1)

        # Project patches to embedding dimension
        return self.proj(x)


class RadialPatchEmbed(nn.Module):
    def __init__(self, k_space_dim, num_rings, max_points_per_patch, embed_dim):
        super().__init__()

        self.k_space_dim = k_space_dim
        self.num_rings = num_rings
        self.max_points_per_patch = max_points_per_patch
        self.embed_dim = embed_dim

        # The largest radius value in normalized grid coordinates ([-1, 1])
        # A small epsilon is added to ensure the max radius is included.
        self.radius_max = torch.sqrt(torch.tensor(2.0)) + 1e-6

        # Learnable ring boundary deltas
        self.radius_deltas = nn.Parameter(torch.ones(num_rings - 1) / (num_rings - 1))

        self.proj = nn.Linear(max_points_per_patch, embed_dim, dtype=torch.cfloat)

        # Precompute radius for each flattened k-space point
        x_coords = torch.linspace(-1, 1, k_space_dim)
        y_coords = torch.linspace(-1, 1, k_space_dim)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        radius_grid = torch.sqrt(grid_x**2 + grid_y**2).flatten()
        self.register_buffer("radius_grid", radius_grid)

    def forward(self, x, visualize=False):
        """
        Args:
            x: complex tensor of shape (batch_size, 1, k_space_dim, k_space_dim)
            visualize: If True, return raw patches as well.

        Returns:
            embeddings: (batch_size, num_rings, embed_dim) tensor of ring embeddings
            raw_patches: (batch_size, num_rings, 1, k_space_dim, k_space_dim) tensor of masked k-space data (if visualize=True)
        """
        batch_size, _, h, w = x.shape
        points = x.view(batch_size, -1)

        # Normalize deltas to define ring boundaries
        deltas = torch.softmax(self.radius_deltas, dim=0)
        # Ensure boundaries span from 0 to radius_max
        boundaries = torch.cat(
            [
                torch.tensor([0.0], device=deltas.device),
                torch.cumsum(deltas * (self.radius_max - 1e-6), dim=0),
            ]
        )

        patches = []
        raw_patches_list = []

        for i in range(self.num_rings):
            low = boundaries[i]
            # Use radius_max for the outer bound of the last ring
            high = boundaries[i + 1] if i < self.num_rings - 1 else self.radius_max

            mask = (self.radius_grid >= low) & (self.radius_grid < high)

            ring_points = points[:, mask]

            current_size = ring_points.shape[1]

            if current_size < self.max_points_per_patch:
                pad_size = self.max_points_per_patch - current_size
                pad = torch.zeros(batch_size, pad_size, dtype=x.dtype, device=x.device)
                ring_points_padded = torch.cat([ring_points, pad], dim=1)
            else:
                ring_points_padded = ring_points[:, : self.max_points_per_patch]

            patch_embedding = self.proj(ring_points_padded)
            patches.append(patch_embedding)

            if visualize:
                # Create a spatial mask for visualization
                spatial_mask = mask.view(1, 1, h, w)
                # Apply mask to the original k-space data
                masked_k_space = x * spatial_mask
                raw_patches_list.append(masked_k_space)

        embeddings = torch.stack(patches, dim=1)

        if visualize:
            raw_patches = torch.stack(raw_patches_list, dim=1)
            return embeddings, raw_patches

        return embeddings


class KSpaceTransformer(nn.Module):
    """
    The main K-Space Transformer model for classification.
    """

    def __init__(
        self, k_space_dim, embed_dim, num_heads, num_layers, ff_dim, num_classes, p=0.2
    ):
        super(KSpaceTransformer, self).__init__()

        num_rings = 16

        self.embedding = RadialPatchEmbed(
            k_space_dim=k_space_dim,
            num_rings=num_rings,
            embed_dim=embed_dim,
            max_points_per_patch=3800,
        )
        self.seq_len = num_rings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim, dtype=torch.cfloat))

        # Input embedding layer
        # Projects each complex k-space point (in_features=1) to a high-dim vector
        self.pos_encoder = LearnablePositionalEncoding(embed_dim, self.seq_len + 1)
        self.embed_dim = embed_dim

        # Stack of Transformer Encoder Blocks
        self.encoder_layers = nn.ModuleList(
            [
                S.ComplexTransformerEncoderBlock(embed_dim, num_heads, ff_dim, p)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = ComplexLayerNorm(embed_dim)

        self.aggregation = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Bag-level Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, k_space_data, visualize=False, modality_ids=None):
        batch_size, _, h, w = k_space_data.shape

        if visualize:
            x, raw_patches = self.embedding(k_space_data, visualize=True)
        else:
            x = self.embedding(k_space_data)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        all_self_attention_maps = []
        # Pass through the encoder stack
        for layer in self.encoder_layers:
            if visualize:
                x, attention_weights = layer(x, return_attention=visualize)
                all_self_attention_maps.append(attention_weights)
            else:
                x = layer(x)

        x_cls = x[:, 0]  # (batch_size, embed_dim)
        x_cls = self.final_norm(x_cls)

        real_features = torch.view_as_real(x_cls)
        instance_features_flat = real_features.flatten(start_dim=1)

        # Calculate attention weights for each instance
        attention_weights = self.aggregation(instance_features_flat)
        attention_weights = torch.softmax(attention_weights, dim=0)

        bag_feature = torch.sum(attention_weights * instance_features_flat, dim=0)

        output = self.classifier(bag_feature)

        if visualize:
            return output, all_self_attention_maps, attention_weights
        else:
            return output


class KSpace2DTransformer(nn.Module):
    """
    Pure 2D Transformer for slice-level classification (no MIL).
    Processes each 2D k-space slice independently with a CLS token.
    """

    def __init__(
        self,
        k_space_dim,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        p=0.2,
        patch_size=16,
        hybrid=False,
        kspace=True,
    ):
        super(KSpace2DTransformer, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.hybrid = hybrid
        self.kspace = kspace
        num_rings = 16

        # Patch embedding
        if self.kspace:
            self.kspace_embedding = RadialPatchEmbed(
                k_space_dim=k_space_dim,
                num_rings=num_rings,
                embed_dim=embed_dim,
                max_points_per_patch=3800,
            )

            self.ring_scale = nn.Parameter(
                torch.ones(num_rings, embed_dim, dtype=torch.cfloat)
            )
            self.ring_bias = nn.Parameter(
                torch.zeros(num_rings, embed_dim, dtype=torch.cfloat)
            )

        else:
            self.image_embedding = ComplexPatchEmbed(
                k_space_dim=k_space_dim, patch_size=patch_size, embed_dim=embed_dim
            )
        self.seq_len = num_rings

        #### Test Case for Complex Patch Embedding ####
        # self.kspace_embedding = ComplexPatchEmbed(
        #         k_space_dim=k_space_dim,
        #         patch_size=patch_size,
        #         embed_dim=embed_dim
        #     )
        # self.seq_len = (k_space_dim // self.patch_size) * (k_space_dim // self.patch_size)

        if self.hybrid:
            self.image_embedding = ComplexPatchEmbed(
                k_space_dim=k_space_dim, patch_size=patch_size, embed_dim=embed_dim
            )
            self.seq_len += (k_space_dim // self.patch_size) * (
                k_space_dim // self.patch_size
            )

            self.type_embed_img = nn.Parameter(
                torch.randn(1, 1, embed_dim, dtype=torch.cfloat)
            )
            self.type_embed_freq = nn.Parameter(
                torch.randn(1, 1, embed_dim, dtype=torch.cfloat)
            )

        # CLS token: Learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim).cfloat())

        # Positional encoding (adjusted for +1 CLS)
        self.pos_encoder = ComplexRotaryPositionalEmbedding(embed_dim)

        self.final_norm = ComplexLayerNorm(embed_dim)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                S.ComplexTransformerEncoderBlock(embed_dim, num_heads, ff_dim, p)
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(embed_dim, num_classes, dtype=torch.cfloat)

        #### Bigger Classification Head ####
        # self.classifier = nn.Sequential(
        #         nn.Linear(embed_dim, 128, dtype=torch.cfloat),
        #         ComplexDropout(p=0.1),
        #         ComplexReLU(),
        #         nn.Linear(128, num_classes, dtype=torch.cfloat)
        #     )
        ####################################

    def forward(self, data, visualize=False):
        """
        Forward pass for a batch of independent 2D k-space slices.

        Args:
            k_space_slices: Tensor of shape (batch_size, k_dim, k_dim) – batch of slices.

        Returns:
            output: Tensor of shape (batch_size, num_classes) – per-slice predictions.
        """
        batch_size = data.shape[0]

        # Embed patches for all slices
        if self.hybrid:
            kspace_tokens = (
                self.kspace_embedding(data[:, 0, ...].unsqueeze(1))
                + self.type_embed_freq
            )
            kspace_tokens = kspace_tokens * self.ring_scale.unsqueeze(
                0
            ) + self.ring_bias.unsqueeze(0)
            image_tokens = (
                self.image_embedding(data[:, 1, ...].unsqueeze(1)) + self.type_embed_img
            )

            x = torch.cat([kspace_tokens, image_tokens], dim=1)

        elif self.kspace:
            x = self.kspace_embedding(data)
            x = x * self.ring_scale.unsqueeze(0) + self.ring_bias.unsqueeze(0)
        else:
            x = self.image_embedding(data)

        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, embed_dim)

        # Prepend CLS to each slice's sequence
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, embed_dim)

        x = self.pos_encoder(x)

        all_self_attention_maps = []
        # Pass through encoder layers
        for layer in self.encoder_layers:
            if visualize:
                x, attention_weights = layer(x, return_attention=visualize)
                all_self_attention_maps.append(attention_weights)
            else:
                x = layer(x)

        # Extract CLS token outputs for each slice
        cls_output = x[:, 0]  # (batch_size, embed_dim)
        cls_output = self.final_norm(cls_output)

        # Classify per slice
        output = self.classifier(cls_output)
        output = (output.real + output.imag) / 2

        if visualize:
            return output, all_self_attention_maps
        else:
            return output
