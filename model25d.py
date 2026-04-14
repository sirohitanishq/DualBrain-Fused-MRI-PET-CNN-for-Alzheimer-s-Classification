import torch
import torch.nn as nn
import torchvision.models as models


class DualStream25D(nn.Module):
    """
    2.5D Dual Stream CNN using pretrained ResNet18 backbones.
    Backbone frozen except layer4 to prevent overfitting on small dataset.

    Input  : mri (B, 3, 128, 128), pet (B, 3, 128, 128)
    Output : (B, 3) logits for CN, MCI, AD
    """

    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        # ── MRI stream ────────────────────────────────────────────────────────
        mri_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mri_backbone.fc = nn.Identity()
        for name, param in mri_backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        self.mri_stream = mri_backbone

        # ── PET stream ────────────────────────────────────────────────────────
        pet_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pet_backbone.fc = nn.Identity()
        for name, param in pet_backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        self.pet_stream = pet_backbone

        # ── Fusion + Classifier ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, mri, pet):
        mri_feat = self.mri_stream(mri)
        pet_feat = self.pet_stream(pet)
        fused    = torch.cat([mri_feat, pet_feat], dim=1)
        return self.classifier(fused)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DualStream25D(num_classes=3).to(device)

    mri_dummy = torch.randn(2, 3, 128, 128).to(device)
    pet_dummy = torch.randn(2, 3, 128, 128).to(device)
    out = model(mri_dummy, pet_dummy)

    print(f"Input MRI shape  : {mri_dummy.shape}")
    print(f"Input PET shape  : {pet_dummy.shape}")
    print(f"Output shape     : {out.shape}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total:,}")
    print("\n✅ Model working correctly!")