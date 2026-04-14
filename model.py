import torch
import torch.nn as nn


# ─── BUILDING BLOCK ───────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """3D Conv → BatchNorm → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ─── SINGLE STREAM ENCODER ────────────────────────────────────────────────────
class StreamEncoder(nn.Module):
    """
    3D CNN encoder for one modality (MRI or PET).
    Input  : (B, 1, 128, 128, 128)
    Output : (B, 256, 8, 8, 8)
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1 → (B, 32, 64, 64, 64)
            ConvBlock(1, 32),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2 → (B, 64, 32, 32, 32)
            ConvBlock(32, 64),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3 → (B, 128, 16, 16, 16)
            ConvBlock(64, 128),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4 → (B, 256, 8, 8, 8)
            ConvBlock(128, 256),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.encoder(x)


# ─── DUAL STREAM CNN ──────────────────────────────────────────────────────────
class DualStreamCNN(nn.Module):
    """
    Two separate encoders for MRI and PET.
    Features are concatenated then classified.

    Input  : mri (B, 1, 128, 128, 128)
             pet (B, 1, 128, 128, 128)
    Output : (B, 3)  → logits for CN, MCI, AD
    """
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        # Two separate streams
        self.mri_stream = StreamEncoder()
        self.pet_stream = StreamEncoder()

        # After concat: 256 + 256 = 512 channels
        # Global average pooling → (B, 512)
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, mri, pet):
        # Extract features from each stream
        mri_features = self.mri_stream(mri)   # (B, 256, 8, 8, 8)
        pet_features = self.pet_stream(pet)   # (B, 256, 8, 8, 8)

        # Fuse by concatenation along channel dim
        fused = torch.cat([mri_features, pet_features], dim=1)  # (B, 512, 8, 8, 8)

        # Global average pooling
        fused = self.gap(fused)   # (B, 512, 1, 1, 1)

        # Classify
        out = self.classifier(fused)  # (B, 3)
        return out


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create model
    model = DualStreamCNN(num_classes=3, dropout=0.5).to(device)

    # Dummy input
    mri_dummy = torch.randn(2, 1, 128, 128, 128).to(device)
    pet_dummy = torch.randn(2, 1, 128, 128, 128).to(device)

    # Forward pass
    out = model(mri_dummy, pet_dummy)

    print(f"Input MRI shape  : {mri_dummy.shape}")
    print(f"Input PET shape  : {pet_dummy.shape}")
    print(f"Output shape     : {out.shape}")   # should be (2, 3)
    print(f"Output logits    : {out}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total_params:,}")
    print("\n✅ Model working correctly!")