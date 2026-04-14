import torch
import torch.nn as nn
import torchvision.models as models


class BrainCNN2D(nn.Module):
    """
    2D CNN for brain disease classification using MRI+PET fused slices.
    Input  : (B, 2, 128, 128) — 2 channels: MRI + PET
    Output : (B, 3) — logits for CN, MCI, AD

    Uses pretrained ResNet18 with modified first conv layer
    to accept 2 channels instead of 3.
    """

    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        # Load pretrained ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv to accept 2 channels instead of 3
        # Average the pretrained weights across the 3 input channels → 2 channels
        old_conv   = backbone.conv1
        new_conv   = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
            )
        backbone.conv1 = new_conv

        # Freeze early layers, fine-tune layer3, layer4, classifier
        for name, param in backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        # Replace final FC
        backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BrainCNN2D(num_classes=3).to(device)
    dummy = torch.randn(4, 2, 128, 128).to(device)
    out   = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable    : {trainable:,} / {total:,}")
    print("\n✅ 2D Model working correctly!")