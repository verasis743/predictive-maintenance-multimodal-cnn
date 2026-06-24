# =============================================================================
# model.py
# Early-Fusion Multimodal CNN for electromechanical fault classification.
#
# Architecture overview:
#   Input: 4-channel fused tensor [thermal_RGB + log-mel spectrogram]
#          Shape: [batch, 4, IMAGE_SIZE, IMAGE_SIZE]
#
#   Stem:  Standard Conv2d (4→32, stride 2) + BN + ReLU
#   Body:  Stack of depthwise separable convolution blocks (MobileNet-style)
#          progressively expanding channels: 32→64→128→256→512
#   Head:  Conv2d (512→1024) + GlobalAvgPool + Dropout + Linear → NUM_CLASSES
#
# Design rationale:
#   Depthwise separable convolutions reduce parameter count and computational
#   cost compared to standard convolutions, making the model suitable for
#   deployment on resource-constrained edge devices (e.g. Raspberry Pi 4).
#   The 4-channel input allows joint learning of thermal and acoustic features
#   from the first layer, enabling the network to discover cross-modal
#   correlations that unimodal models cannot exploit.
# =============================================================================

import torch
import torch.nn as nn

from config import NUM_CLASSES, IN_CHANNELS, DROPOUT_RATE


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.

    Replaces a standard Conv2d with two operations:
      1. Depthwise conv: one filter per input channel (spatial mixing).
      2. Pointwise conv: 1x1 conv to mix channels (channel projection).

    This reduces FLOPs by a factor of ~8-9x compared to a standard 3x3 conv
    while preserving representational capacity.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        s:  Stride for the depthwise convolution (default 1).
    """

    def __init__(self, c1: int, c2: int, s: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(c1, c1, kernel_size=3, stride=s, padding=1,
                             groups=c1, bias=False)   # depthwise
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw  = nn.Conv2d(c1, c2, kernel_size=1, bias=False)  # pointwise
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class EarlyFusionNet(nn.Module):
    """
    Lightweight early-fusion CNN for multimodal fault classification.

    Accepts a 4-channel fused input (3 thermal RGB + 1 log-mel spectrogram)
    and classifies electromechanical machines into health states:
    Healthy, Mild_Fault, or Heavy_Fault.

    Architecture stages:
        Stage       Output shape (224x224 input)    Channels
        ─────────── ──────────────────────────────  ────────
        Stem        [B, 32,  112, 112]              4  → 32
        Block 1     [B, 64,   56,  56]              32 → 64   (1 DS-conv, stride 2)
        Block 2     [B, 128,  28,  28]              64 → 128  (3 DS-convs, stride 2)
        Block 3     [B, 256,  14,  14]              128→ 256  (4 DS-convs, stride 2)
        Block 4     [B, 512,   7,   7]              256→ 512  (3 DS-convs, stride 2)
        Conv head   [B, 1024,  4,   4]              512→ 1024 (stride 2)
        GAP + FC    [B, NUM_CLASSES]

    Args:
        nc: Number of output classes (default: NUM_CLASSES from config).
    """

    # Block config: (in_channels, out_channels, num_repeats, stride)
    _BLOCK_CFG = [
        (32,  64,  2, 2),
        (64,  128, 3, 2),
        (128, 256, 4, 2),
        (256, 512, 3, 2),
    ]

    def __init__(self, nc: int = NUM_CLASSES):
        super().__init__()

        # Stem: adapts 4-channel fused input to 32-channel feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Body: depthwise separable convolution blocks
        layers = []
        for c1, c2, repeats, stride in self._BLOCK_CFG:
            layers.append(DepthwiseSeparableConv(c1, c2, s=stride))
            for _ in range(repeats - 1):
                layers.append(DepthwiseSeparableConv(c2, c2))
        self.blocks = nn.Sequential(*layers)

        # Final conv to expand to 1024 channels before global pooling
        self.conv_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Classification head: GlobalAvgPool → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, nc),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 4, H, W].

        Returns:
            Logits tensor of shape [B, NUM_CLASSES].
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps from the final convolutional layer.
        Used for UMAP visualisation and Grad-CAM saliency mapping.

        Args:
            x: Input tensor of shape [B, 4, H, W].

        Returns:
            Feature map tensor of shape [B, 1024, H', W'] before global pooling.
        """
        x = self.stem(x)
        x = self.blocks(x)
        # Apply only the Conv2d layer of conv_head (index 0), not BN+ReLU,
        # to get raw spatial activations for visualisation purposes.
        x = self.conv_head[0](x)
        return x


# =============================================================================
# IoT / Edge Deployment — ONNX Export
# =============================================================================
# Uncomment and run this block to export the trained model to ONNX format
# for deployment on edge devices (Raspberry Pi 4, Jetson Nano, etc.).
#
# The exported .onnx file can be:
#   - Loaded by ONNX Runtime on the edge device for inference
#   - Integrated with Node-RED via a Python node for real-time fault detection
#   - Served over MQTT: edge device publishes predictions, Node-RED dashboard
#     subscribes and visualises fault state in real time
#
# Typical end-to-end IoT pipeline:
#   INMP441 mic + thermal camera
#       → Raspberry Pi 4 (ONNX Runtime inference, <200ms latency)
#           → MQTT broker (Mosquitto)
#               → Node-RED dashboard (real-time fault visualisation)
#
# Usage:
#   from model import EarlyFusionNet
#   import torch, torch.onnx
#
#   model = EarlyFusionNet()
#   model.load_state_dict(torch.load("checkpoints/best_fusion.pth", map_location="cpu"))
#   model.eval()
#
#   dummy_input = torch.randn(1, 4, 224, 224)  # [batch=1, channels=4, H=224, W=224]
#
#   torch.onnx.export(
#       model,
#       dummy_input,
#       "checkpoints/fusion_model.onnx",
#       export_params=True,
#       opset_version=11,          # ONNX opset 11 is supported by ONNX Runtime 1.x+
#       input_names=["fused_input"],
#       output_names=["fault_logits"],
#       dynamic_axes={
#           "fused_input":  {0: "batch_size"},
#           "fault_logits": {0: "batch_size"},
#       },
#   )
#   print("Model exported to checkpoints/fusion_model.onnx")
#
# To run inference on the edge device:
#   import onnxruntime as ort
#   import numpy as np
#
#   session = ort.InferenceSession("fusion_model.onnx")
#   input_name = session.get_inputs()[0].name
#   logits = session.run(None, {input_name: fused_input_numpy})[0]
#   predicted_class = CLASS_NAMES[logits.argmax()]
# =============================================================================
