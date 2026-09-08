"""Native Apple MLX implementation of IndicDocLayout (PPDocLayoutV3 / RT-DETR).

Provides full page-level document layout analysis, 37 class categorizations,
and pairwise reading order matrix on Apple Silicon Metal GPU.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as mnn


def inverse_sigmoid(x: mx.array, eps: float = 1e-5) -> mx.array:
    x = mx.clip(x, eps, 1.0 - eps)
    return mx.log(x / (1.0 - x))


def bilinear_grid_sample_2d(
    val: mx.array,
    grid: mx.array,
) -> mx.array:
    """Bilinear 2D grid sampling in MLX matching torch.nn.functional.grid_sample.

    Args:
        val: Input tensor of shape [B, C, H, W].
        grid: Sampling grid of shape [B, H_out, W_out, 2] with coordinates in [-1, 1] (x, y).

    Returns:
        Sampled values of shape [B, C, H_out, W_out].
    """
    b, c, h, w = val.shape
    b_g, h_out, w_out, _ = grid.shape

    # Normalize [-1, 1] to pixel space [0, W-1], [0, H-1]
    x = ((grid[..., 0] + 1.0) * w - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * h - 1.0) * 0.5

    x0 = mx.floor(x).astype(mx.int32)
    x1 = x0 + 1
    y0 = mx.floor(y).astype(mx.int32)
    y1 = y0 + 1

    wa = (x1.astype(mx.float32) - x) * (y1.astype(mx.float32) - y)
    wb = (x1.astype(mx.float32) - x) * (y - y0.astype(mx.float32))
    wc = (x - x0.astype(mx.float32)) * (y1.astype(mx.float32) - y)
    wd = (x - x0.astype(mx.float32)) * (y - y0.astype(mx.float32))

    mask_x0 = (x0 >= 0) & (x0 < w)
    mask_x1 = (x1 >= 0) & (x1 < w)
    mask_y0 = (y0 >= 0) & (y0 < h)
    mask_y1 = (y1 >= 0) & (y1 < h)

    x0_c = mx.clip(x0, 0, w - 1)
    x1_c = mx.clip(x1, 0, w - 1)
    y0_c = mx.clip(y0, 0, h - 1)
    y1_c = mx.clip(y1, 0, h - 1)

    val_bhwc = mx.transpose(val, (0, 2, 3, 1))
    b_idx = mx.arange(b)[:, None, None]

    ia = val_bhwc[b_idx, y0_c, x0_c] * (mask_x0 & mask_y0)[..., None]
    ib = val_bhwc[b_idx, y1_c, x0_c] * (mask_x0 & mask_y1)[..., None]
    ic = val_bhwc[b_idx, y0_c, x1_c] * (mask_x1 & mask_y0)[..., None]
    id_ = val_bhwc[b_idx, y1_c, x1_c] * (mask_x1 & mask_y1)[..., None]

    out_bhwc = (
        wa[..., None] * ia
        + wb[..., None] * ib
        + wc[..., None] * ic
        + wd[..., None] * id_
    )
    return mx.transpose(out_bhwc, (0, 3, 1, 2))


def mask_to_box_coordinate(mask: mx.array) -> mx.array:
    """Extract bounding box coordinates from binary masks."""
    b, q, h, w = mask.shape
    y_coords, x_coords = mx.meshgrid(
        mx.arange(h, dtype=mx.float32),
        mx.arange(w, dtype=mx.float32),
        indexing="ij",
    )
    x_coords = x_coords[None, None]  # [1, 1, H, W]
    y_coords = y_coords[None, None]

    mask_bool = mask > 0
    x_masked = x_coords * mask_bool
    y_masked = y_coords * mask_bool

    # Max coordinates
    x_max = mx.max(x_masked.reshape(b, q, -1), axis=-1) + 1.0
    y_max = mx.max(y_masked.reshape(b, q, -1), axis=-1) + 1.0

    # Min coordinates
    big_val = 1e6
    x_min_arr = mx.where(mask_bool, x_coords, big_val)
    y_min_arr = mx.where(mask_bool, y_coords, big_val)
    x_min = mx.min(x_min_arr.reshape(b, q, -1), axis=-1)
    y_min = mx.min(y_min_arr.reshape(b, q, -1), axis=-1)

    # Normalize to [0, 1]
    x_min = x_min / w
    y_min = y_min / h
    x_max = x_max / w
    y_max = y_max / h

    # Convert xyxy to cxcywh
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    box_w = mx.clip(x_max - x_min, 1e-4, 1.0)
    box_h = mx.clip(y_max - y_min, 1e-4, 1.0)

    return mx.stack([cx, cy, box_w, box_h], axis=-1)


class MlxConvBNAct(mnn.Module):
    """Fused Conv2d with optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        self.conv = mnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,
        )
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv(x)
        if self.activation == "relu":
            return mnn.relu(y)
        if self.activation == "silu":
            return mnn.silu(y)
        if self.activation == "gelu":
            return mnn.gelu(y)
        return y


class MlxRepVggBlock(mnn.Module):
    """RepVGG block in CSPRepLayer."""

    def __init__(self, channels: int, activation: str = "silu") -> None:
        super().__init__()
        self.conv1 = MlxConvBNAct(channels, channels, 3, 1, 1, activation=None)
        self.conv2 = MlxConvBNAct(channels, channels, 1, 1, 0, activation=None)
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(x) + self.conv2(x)
        if self.activation == "silu":
            return mnn.silu(y)
        if self.activation == "relu":
            return mnn.relu(y)
        return y


class MlxCSPRepLayer(mnn.Module):
    """CSPRepLayer using RepVGG blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.conv1 = MlxConvBNAct(
            in_channels, out_channels, 1, 1, 0, activation=activation
        )
        self.conv2 = MlxConvBNAct(
            in_channels, out_channels, 1, 1, 0, activation=activation
        )
        self.bottlenecks = [
            MlxRepVggBlock(out_channels, activation=activation)
            for _ in range(num_blocks)
        ]
        self.conv3 = (
            MlxConvBNAct(out_channels, out_channels, 1, 1, 0, activation=activation)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        y1 = self.conv1(x)
        for block in self.bottlenecks:
            y1 = block(y1)
        y2 = self.conv2(x)
        out = y1 + y2
        if self.conv3 is not None:
            out = self.conv3(out)
        return out


class MlxHGNetV2Stem(mnn.Module):
    """HGNetV2 stem embedder."""

    def __init__(self) -> None:
        super().__init__()
        self.stem1 = MlxConvBNAct(3, 32, 3, 2, 1, activation="relu")
        self.stem2a = MlxConvBNAct(32, 16, 2, 1, 0, activation="relu")
        self.stem2b = MlxConvBNAct(16, 32, 2, 1, 0, activation="relu")
        self.stem3 = MlxConvBNAct(64, 32, 3, 2, 1, activation="relu")
        self.stem4 = MlxConvBNAct(32, 48, 1, 1, 0, activation="relu")
        self.pool = mnn.MaxPool2d(kernel_size=2, stride=1)

    def __call__(self, x: mx.array) -> mx.array:
        emb = self.stem1(x)
        emb_pad = mx.pad(emb, ((0, 0), (0, 1), (0, 1), (0, 0)))
        emb_2a = self.stem2a(emb_pad)
        emb_2a_pad = mx.pad(emb_2a, ((0, 0), (0, 1), (0, 1), (0, 0)))
        emb_2b = self.stem2b(emb_2a_pad)
        pooled = self.pool(emb_pad)
        cat = mx.concatenate([pooled, emb_2b], axis=-1)
        out = self.stem3(cat)
        out = self.stem4(out)
        return out


class MlxHGNetV2BasicLayer(mnn.Module):
    """Basic layer for HGNetV2 stage."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int = 3,
        light_block: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.layers = []
        for i in range(layer_num):
            temp_in = in_channels if i == 0 else mid_channels
            if light_block:
                conv1 = MlxConvBNAct(temp_in, mid_channels, 1, 1, 0, activation=None)
                conv2 = MlxConvBNAct(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    1,
                    (kernel_size - 1) // 2,
                    groups=mid_channels,
                    activation="relu",
                )
                self.layers.append((conv1, conv2))
            else:
                conv = MlxConvBNAct(
                    temp_in, mid_channels, kernel_size, 1, 1, activation="relu"
                )
                self.layers.append(conv)

        total_channels = in_channels + layer_num * mid_channels
        self.squeeze = MlxConvBNAct(
            total_channels, out_channels // 2, 1, 1, 0, activation="relu"
        )
        self.excitation = MlxConvBNAct(
            out_channels // 2, out_channels, 1, 1, 0, activation="relu"
        )

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        outputs = [x]
        curr = x
        for block in self.layers:
            if isinstance(block, tuple):
                conv1, conv2 = block
                curr = conv2(conv1(curr))
            else:
                curr = block(curr)
            outputs.append(curr)
        fused = mx.concatenate(outputs, axis=-1)
        out = self.excitation(self.squeeze(fused))
        if self.residual:
            out = out + identity
        return out


class MlxHGNetV2Backbone(mnn.Module):
    """Full HGNetV2 backbone producing 4 feature stages."""

    def __init__(self) -> None:
        super().__init__()
        self.embedder = MlxHGNetV2Stem()

        # Stage 0: 48 -> 128 (1 block, mid 48, k=3, standard)
        self.stage0 = MlxHGNetV2BasicLayer(
            48, 48, 128, layer_num=6, kernel_size=3, light_block=False
        )

        # Stage 1: downsample (128 -> 96, s=2), basic (96 -> 512, mid 96, k=3, standard)
        self.downsample1 = MlxConvBNAct(128, 96, 3, 2, 1, activation="relu")
        self.stage1 = MlxHGNetV2BasicLayer(
            96, 96, 512, layer_num=6, kernel_size=3, light_block=False
        )

        # Stage 2: downsample (512 -> 192, s=2), 3 blocks (192 -> 1024, mid 192, k=5, light_block)
        self.downsample2 = MlxConvBNAct(512, 192, 3, 2, 1, activation="relu")
        self.stage2_blocks = [
            MlxHGNetV2BasicLayer(
                192 if i == 0 else 1024,
                192,
                1024,
                layer_num=6,
                kernel_size=5,
                light_block=True,
                residual=(i > 0),
            )
            for i in range(3)
        ]

        # Stage 3: downsample (1024 -> 384, s=2), 1 block (384 -> 2048, mid 384, k=5, light_block)
        self.downsample3 = MlxConvBNAct(1024, 384, 3, 2, 1, activation="relu")
        self.stage3 = MlxHGNetV2BasicLayer(
            384, 384, 2048, layer_num=6, kernel_size=5, light_block=True
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # Input: [B, H, W, 3]
        stem = self.embedder(x)
        s0 = self.stage0(stem)

        d1 = self.downsample1(s0)
        s1 = self.stage1(d1)

        curr = self.downsample2(s1)
        for b in self.stage2_blocks:
            curr = b(curr)
        s2 = curr

        d3 = self.downsample3(s2)
        s3 = self.stage3(d3)

        return s0, s1, s2, s3


class MlxAIFI(mnn.Module):
    """Intra-scale feature interaction transformer layer."""

    def __init__(
        self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024
    ) -> None:
        super().__init__()
        self.self_attn = mnn.MultiHeadAttention(d_model, nhead, bias=True)
        self.norm1 = mnn.LayerNorm(d_model)
        self.linear1 = mnn.Linear(d_model, dim_feedforward)
        self.linear2 = mnn.Linear(dim_feedforward, d_model)
        self.norm2 = mnn.LayerNorm(d_model)

    def __call__(self, x: mx.array, pos: mx.array) -> mx.array:
        q = k = x + pos
        attn_out = self.self_attn(q, k, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.linear2(mnn.gelu(self.linear1(x)))
        x = self.norm2(x + ffn_out)
        return x


class MlxHybridEncoder(mnn.Module):
    """HybridEncoder featuring AIFI and CCFM top-down/bottom-up paths."""

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.input_proj = [
            MlxConvBNAct(512, d_model, 1, 1, 0, activation="silu"),
            MlxConvBNAct(1024, d_model, 1, 1, 0, activation="silu"),
            MlxConvBNAct(2048, d_model, 1, 1, 0, activation="silu"),
        ]
        self.aifi = MlxAIFI(d_model)

        # Top-down FPN
        self.lateral1 = MlxConvBNAct(d_model, d_model, 1, 1, 0, activation="silu")
        self.fpn_block1 = MlxCSPRepLayer(
            d_model * 2, d_model, num_blocks=3, activation="silu"
        )
        self.lateral2 = MlxConvBNAct(d_model, d_model, 1, 1, 0, activation="silu")
        self.fpn_block2 = MlxCSPRepLayer(
            d_model * 2, d_model, num_blocks=3, activation="silu"
        )

        # Bottom-up PAN
        self.downsample1 = MlxConvBNAct(d_model, d_model, 3, 2, 1, activation="silu")
        self.pan_block1 = MlxCSPRepLayer(
            d_model * 2, d_model, num_blocks=3, activation="silu"
        )
        self.downsample2 = MlxConvBNAct(d_model, d_model, 3, 2, 1, activation="silu")
        self.pan_block2 = MlxCSPRepLayer(
            d_model * 2, d_model, num_blocks=3, activation="silu"
        )

        # Mask feature FPN
        self.mask_conv1 = MlxConvBNAct(d_model, 64, 3, 1, 1, activation="silu")
        self.mask_conv2 = MlxConvBNAct(d_model, 64, 3, 1, 1, activation="silu")
        self.mask_conv3 = MlxConvBNAct(d_model, 64, 3, 1, 1, activation="silu")
        self.mask_fusion = MlxConvBNAct(64, 64, 1, 1, 0, activation="silu")
        self.encoder_mask_lateral = MlxConvBNAct(128, 64, 3, 1, 1, activation="silu")
        self.encoder_mask_output = mnn.Conv2d(64, 32, 1, 1, 0, bias=True)

    def _build_2d_sincos_pos(self, h: int, w: int, d_model: int = 256) -> mx.array:
        pos_dim = d_model // 4
        omega = 1.0 / (10000.0 ** (mx.arange(pos_dim, dtype=mx.float32) / pos_dim))
        grid_h, grid_w = mx.meshgrid(
            mx.arange(h, dtype=mx.float32),
            mx.arange(w, dtype=mx.float32),
            indexing="ij",
        )
        emb_h = grid_h.reshape(-1, 1) * omega[None, :]
        emb_w = grid_w.reshape(-1, 1) * omega[None, :]
        pos = mx.concatenate(
            [mx.sin(emb_h), mx.cos(emb_h), mx.sin(emb_w), mx.cos(emb_w)], axis=-1
        )
        return pos[None, :]  # [1, H*W, D]

    def __call__(
        self,
        s0: mx.array,
        s1: mx.array,
        s2: mx.array,
        s3: mx.array,
    ) -> tuple[list[mx.array], mx.array]:
        # Project backbone features to d_model=256
        f0 = self.input_proj[0](s1)  # stride 8,  [B, 128, 128, 256]
        f1 = self.input_proj[1](s2)  # stride 16, [B, 64, 64, 256]
        f2 = self.input_proj[2](s3)  # stride 32, [B, 32, 32, 256]

        # AIFI on f2
        b, h2, w2, c = f2.shape
        pos = self._build_2d_sincos_pos(h2, w2, c)
        f2_flat = f2.reshape(b, h2 * w2, c)
        f2_aifi = self.aifi(f2_flat, pos).reshape(b, h2, w2, c)

        # Top-down FPN
        # Level 1 (stride 16)
        lat1 = self.lateral1(f2_aifi)
        lat1_up = mx.repeat(mx.repeat(lat1, 2, axis=1), 2, axis=2)
        cat1 = mx.concatenate([lat1_up, f1], axis=-1)
        fpn1 = self.fpn_block1(cat1)

        # Level 0 (stride 8)
        lat2 = self.lateral2(fpn1)
        lat2_up = mx.repeat(mx.repeat(lat2, 2, axis=1), 2, axis=2)
        cat0 = mx.concatenate([lat2_up, f0], axis=-1)
        fpn0 = self.fpn_block2(cat0)

        # Bottom-up PAN
        # Level 1 (stride 16)
        down1 = self.downsample1(fpn0)
        pan_cat1 = mx.concatenate([down1, fpn1], axis=-1)
        pan1 = self.pan_block1(pan_cat1)

        # Level 2 (stride 32)
        down2 = self.downsample2(pan1)
        pan_cat2 = mx.concatenate([down2, f2_aifi], axis=-1)
        pan2 = self.pan_block2(pan_cat2)

        # Mask feature generation
        m0 = self.mask_conv1(fpn0)  # stride 8
        m1 = self.mask_conv2(pan1)  # stride 16
        m2 = self.mask_conv3(pan2)  # stride 32

        m2_up = mx.repeat(mx.repeat(m2, 4, axis=1), 4, axis=2)
        m1_up = mx.repeat(mx.repeat(m1, 2, axis=1), 2, axis=2)
        mask_fused = self.mask_fusion(m0 + m1_up + m2_up)

        # Upsample to stride 4
        mask_fused_up = mx.repeat(mx.repeat(mask_fused, 2, axis=1), 2, axis=2)
        mask_lateral = self.encoder_mask_lateral(s0)
        mask_proto = self.encoder_mask_output(mask_fused_up + mask_lateral)

        return [fpn0, pan1, pan2], mask_proto


class MlxDeformableAttention(mnn.Module):
    """Multi-scale deformable cross-attention in MLX."""

    def __init__(
        self, d_model: int = 256, n_heads: int = 8, n_levels: int = 3, n_points: int = 4
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = mnn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = mnn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = mnn.Linear(d_model, d_model)
        self.output_proj = mnn.Linear(d_model, d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_features: list[mx.array],
        reference_points: mx.array,
    ) -> mx.array:
        b, num_queries, _ = hidden_states.shape
        head_dim = self.d_model // self.n_heads

        # Sampling offsets and attention weights
        offsets = self.sampling_offsets(hidden_states).reshape(
            b, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn_w = self.attention_weights(hidden_states).reshape(
            b, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attn_w = mnn.softmax(attn_w, axis=-1).reshape(
            b, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        sampled_levels = []
        for lvl, feat in enumerate(encoder_features):
            _, h_l, w_l, _ = feat.shape
            # feat is [B, H, W, D] -> transpose to [B, D, H, W] for grid sampling
            feat_nchw = mx.transpose(self.value_proj(feat), (0, 3, 1, 2))
            feat_heads = feat_nchw.reshape(b * self.n_heads, head_dim, h_l, w_l)

            # Sampling locations in [0, 1]
            ref_lvl = reference_points[:, :, None, None, :2]
            offset_lvl = offsets[:, :, :, lvl, :, :]
            normalizer = mx.array([w_l, h_l], dtype=mx.float32)[
                None, None, None, None, :
            ]
            loc = ref_lvl + offset_lvl / normalizer

            # Convert to grid [-1, 1]
            grid_lvl = 2.0 * loc - 1.0
            grid_flat = mx.transpose(grid_lvl, (0, 2, 1, 3, 4)).reshape(
                b * self.n_heads, num_queries, self.n_points, 2
            )

            # Bilinear sampling
            sampled = bilinear_grid_sample_2d(feat_heads, grid_flat)
            sampled = sampled.reshape(
                b, self.n_heads, head_dim, num_queries, self.n_points
            )
            sampled_levels.append(sampled)

        # Stack over levels: [B, n_heads, head_dim, num_queries, n_levels, n_points]
        stacked = mx.stack(sampled_levels, axis=4)
        attn_weights_t = mx.transpose(attn_w, (0, 2, 1, 3, 4))[:, :, None, :, :, :]
        weighted = (stacked * attn_weights_t).sum(
            axis=(4, 5)
        )  # [B, n_heads, head_dim, num_queries]

        out = mx.transpose(weighted, (0, 3, 1, 2)).reshape(b, num_queries, self.d_model)
        return self.output_proj(out)


class MlxDecoderLayer(mnn.Module):
    """RT-DETR Decoder layer."""

    def __init__(
        self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024
    ) -> None:
        super().__init__()
        self.self_attn = mnn.MultiHeadAttention(d_model, nhead, bias=True)
        self.self_attn_layer_norm = mnn.LayerNorm(d_model)

        self.encoder_attn = MlxDeformableAttention(d_model, nhead)
        self.encoder_attn_layer_norm = mnn.LayerNorm(d_model)

        self.mlp_fc1 = mnn.Linear(d_model, dim_feedforward)
        self.mlp_fc2 = mnn.Linear(dim_feedforward, d_model)
        self.final_layer_norm = mnn.LayerNorm(d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        query_pos: mx.array,
        encoder_features: list[mx.array],
        reference_points: mx.array,
    ) -> mx.array:
        # Self-attention
        q = k = hidden_states + query_pos
        sa_out = self.self_attn(q, k, hidden_states)
        hidden_states = self.self_attn_layer_norm(hidden_states + sa_out)

        # Cross-attention (Deformable attention)
        ca_out = self.encoder_attn(hidden_states, encoder_features, reference_points)
        hidden_states = self.encoder_attn_layer_norm(hidden_states + ca_out)

        # FFN
        ffn_out = self.mlp_fc2(mnn.relu(self.mlp_fc1(hidden_states)))
        hidden_states = self.final_layer_norm(hidden_states + ffn_out)
        return hidden_states


class MlxPPDocLayout(mnn.Module):
    """End-to-end PPDocLayoutV3 model in MLX."""

    def __init__(
        self, num_queries: int = 300, num_classes: int = 37, d_model: int = 256
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.d_model = d_model

        self.backbone = MlxHGNetV2Backbone()
        self.encoder = MlxHybridEncoder(d_model)

        # Query selection heads
        self.enc_output = mnn.Conv2d(d_model, d_model, 1, 1, 0, bias=True)
        self.enc_score_head = mnn.Linear(d_model, num_classes)
        self.enc_bbox_head = [
            mnn.Linear(d_model, d_model),
            mnn.Linear(d_model, d_model),
            mnn.Linear(d_model, 4),
        ]

        # Query position embedding head
        self.query_pos_head = [
            mnn.Linear(4, 512),
            mnn.Linear(512, d_model),
        ]

        # Decoder layers (6 layers)
        self.decoder_layers = [MlxDecoderLayer(d_model) for _ in range(6)]
        self.decoder_norm = mnn.LayerNorm(d_model)

        # Prediction heads
        self.class_embed = mnn.Linear(d_model, num_classes)
        self.bbox_embed = [
            mnn.Linear(d_model, d_model),
            mnn.Linear(d_model, d_model),
            mnn.Linear(d_model, 4),
        ]
        self.mask_query_head = [
            mnn.Linear(d_model, d_model),
            mnn.Linear(d_model, d_model),
            mnn.Linear(d_model, 32),
        ]

        # Reading order heads
        self.decoder_order_head = [mnn.Linear(d_model, d_model) for _ in range(6)]
        self.decoder_global_pointer_dense = mnn.Linear(d_model, 128)

    def _forward_mlp(self, layers: list[mnn.Linear], x: mx.array) -> mx.array:
        for i, layer in enumerate(layers):
            x = layer(x)
            if i < len(layers) - 1:
                x = mnn.relu(x)
        return x

    def __call__(self, pixel_values: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Execute layout detection forward pass.

        Args:
            pixel_values: Input image tensor [B, 1024, 1024, 3] normalized to [0, 1].

        Returns:
            logits: Class logits [B, 300, 37].
            pred_boxes: Bounding boxes [B, 300, 4] in [cx, cy, w, h].
            order_logits: Reading order affinity matrix [B, 300, 300].
        """
        b = pixel_values.shape[0]

        # 1. Backbone
        s0, s1, s2, s3 = self.backbone(pixel_values)

        # 2. Encoder
        encoder_features, mask_proto = self.encoder(s0, s1, s2, s3)

        # Flatten encoder features: [B, H*W, D]
        feats_flat = [f.reshape(b, -1, self.d_model) for f in encoder_features]
        source_flatten = mx.concatenate(feats_flat, axis=1)

        # 3. Query selection
        output_memory = self.enc_output(source_flatten)
        enc_class_logits = self.enc_score_head(output_memory)

        # Top-k query selection
        scores = mx.max(enc_class_logits, axis=-1)  # [B, N]
        topk_indices = mx.argpartition(scores, -self.num_queries, axis=-1)[
            :, -self.num_queries :
        ]

        # Gather target queries
        b_idx = mx.arange(b)[:, None]
        target = output_memory[b_idx, topk_indices]  # [B, 300, D]

        # Initial reference points from masks
        out_query = self.decoder_norm(target)
        mask_embed = self._forward_mlp(self.mask_query_head, out_query)  # [B, 300, 32]

        # Compute mask logits: mask_embed @ mask_proto.flatten
        _, h_m, w_m, c_m = mask_proto.shape
        proto_flat = mx.transpose(mask_proto, (0, 3, 1, 2)).reshape(b, c_m, h_m * w_m)
        enc_masks = (mask_embed @ proto_flat).reshape(b, self.num_queries, h_m, w_m)
        init_ref_boxes = mask_to_box_coordinate(enc_masks)  # [B, 300, 4]

        # 4. Decoder layers with iterative box refinement
        hidden_states = target
        ref_points = init_ref_boxes

        for idx, layer in enumerate(self.decoder_layers):
            # Positional embeddings
            pos_emb = self._forward_mlp(self.query_pos_head, ref_points)

            # Layer forward
            hidden_states = layer(hidden_states, pos_emb, encoder_features, ref_points)

            # Box refinement
            box_deltas = self._forward_mlp(self.bbox_embed, hidden_states)
            ref_points = mnn.sigmoid(box_deltas + inverse_sigmoid(ref_points))

        # 5. Output predictions
        final_norm = self.decoder_norm(hidden_states)
        logits = self.class_embed(final_norm)
        pred_boxes = ref_points

        # Reading order from last layer
        last_order_query = self.decoder_order_head[-1](final_norm)
        qk = self.decoder_global_pointer_dense(last_order_query)  # [B, 300, 128]
        q_vec = qk[..., :64]
        k_vec = qk[..., 64:]

        order_logits = (q_vec @ mx.transpose(k_vec, (0, 2, 1))) / math.sqrt(64.0)
        mask = mx.tril(mx.ones((self.num_queries, self.num_queries)))
        order_logits = mx.where(mask[None, :, :], -1e4, order_logits)

        return logits, pred_boxes, order_logits
