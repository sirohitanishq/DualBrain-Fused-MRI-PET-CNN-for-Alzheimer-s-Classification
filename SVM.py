import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import joblib

from dataset25d import BrainDataset25D

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABELS_CSV       = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
MRI_DIR          = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
PET_DIR          = r"C:\JupyterNotebook\MRI_PET\Data\PET"
SAVE_DIR         = r"C:\JupyterNotebook\MRI_PET\SVM"
SLICES_PER_PLANE = 10
BATCH_SIZE       = 32
RANDOM_SEED      = 42
# ──────────────────────────────────────────────────────────────────────────────

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ─── FEATURE EXTRACTOR ────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    """
    Dual stream ResNet18 feature extractor.
    Fully frozen — only used to extract features, not trained.
    """
    def __init__(self):
        super().__init__()
        mri_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mri_backbone.fc = nn.Identity()
        self.mri_stream = mri_backbone

        pet_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pet_backbone.fc = nn.Identity()
        self.pet_stream = pet_backbone

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, mri, pet):
        mri_feat = self.mri_stream(mri)                      # (B, 512)
        pet_feat = self.pet_stream(pet)                      # (B, 512)
        return torch.cat([mri_feat, pet_feat], dim=1)        # (B, 1024)


def extract_features(model, loader):
    """Extract features from all batches and return as numpy arrays."""
    model.eval()
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for mri, pet, labels in tqdm(loader, desc="  Extracting features"):
            mri  = mri.to(device)
            pet  = pet.to(device)
            feat = model(mri, pet)
            all_features.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Splits
    df = pd.read_csv(LABELS_CSV)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["Numeric_Label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["Numeric_Label"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

    # Datasets — no augmentation for feature extraction
    train_dataset = BrainDataset25D(train_df, MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)
    val_dataset   = BrainDataset25D(val_df,   MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)
    test_dataset  = BrainDataset25D(test_df,  MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Feature extractor
    extractor = FeatureExtractor().to(device)
    print("Extracting features...\n")

    print("→ Train set")
    X_train, y_train = extract_features(extractor, train_loader)
    print("→ Val set")
    X_val,   y_val   = extract_features(extractor, val_loader)
    print("→ Test set")
    X_test,  y_test  = extract_features(extractor, test_loader)

    print(f"\nTrain features shape : {X_train.shape}")
    print(f"Val features shape   : {X_val.shape}")
    print(f"Test features shape  : {X_test.shape}\n")

    # Normalize features
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_val    = scaler.transform(X_val)
    X_test   = scaler.transform(X_test)

    # ── Train SVM ─────────────────────────────────────────────────────────────
    print("Training SVM...\n")
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",   # handles AD imbalance
        probability=True,
        random_state=RANDOM_SEED
    )
    svm.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    label_names = ["CN", "MCI", "AD"]

    for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = svm.predict(X)
        acc   = accuracy_score(y, preds)
        print(f"{'='*50}")
        print(f"{split_name} Accuracy: {acc:.4f}")
        print(f"{'='*50}")
        print(classification_report(y, preds, target_names=label_names))
        cm = confusion_matrix(y, preds)
        print(f"Confusion Matrix ({split_name}):")
        print(pd.DataFrame(cm, index=label_names, columns=label_names))
        print()

    # Save SVM and scaler
    joblib.dump(svm,    Path(SAVE_DIR) / "svm_model.pkl")
    joblib.dump(scaler, Path(SAVE_DIR) / "scaler.pkl")
    print(f"✅ SVM model saved to: {SAVE_DIR}\\svm_model.pkl")