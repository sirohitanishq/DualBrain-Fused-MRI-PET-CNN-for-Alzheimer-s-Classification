import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
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


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        mri_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mri_backbone.fc = nn.Identity()
        self.mri_stream = mri_backbone

        pet_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pet_backbone.fc = nn.Identity()
        self.pet_stream = pet_backbone

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, mri, pet):
        mri_feat = self.mri_stream(mri)
        pet_feat = self.pet_stream(pet)
        return torch.cat([mri_feat, pet_feat], dim=1)


def extract_features(model, loader):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for mri, pet, labels in tqdm(loader, desc="  Extracting"):
            mri  = mri.to(device)
            pet  = pet.to(device)
            feat = model(mri, pet)
            all_features.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


if __name__ == "__main__":
    df = pd.read_csv(LABELS_CSV)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["Numeric_Label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["Numeric_Label"]
    )

    train_dataset = BrainDataset25D(train_df, MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)
    val_dataset   = BrainDataset25D(val_df,   MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)
    test_dataset  = BrainDataset25D(test_df,  MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    extractor = FeatureExtractor().to(device)
    print("Extracting features...\n")
    X_train, y_train = extract_features(extractor, train_loader)
    X_val,   y_val   = extract_features(extractor, val_loader)
    X_test,  y_test  = extract_features(extractor, test_loader)

    # Normalize
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # PCA — reduce to 256 dims, removes noise and speeds up SVM
    print("Applying PCA...")
    pca     = PCA(n_components=256, random_state=RANDOM_SEED)
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}\n")

    # Grid search for best SVM params
    print("Running Grid Search for best SVM params...")
    param_grid = {
        "C"      : [1, 10, 100],
        "gamma"  : ["scale", "auto"],
        "kernel" : ["rbf", "poly"]
    }
    svm_base = SVC(class_weight="balanced", probability=True, random_state=RANDOM_SEED)
    grid     = GridSearchCV(svm_base, param_grid, cv=3, scoring="balanced_accuracy",
                            verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}\n")

    best_svm = grid.best_estimator_

    # Evaluate
    label_names = ["CN", "MCI", "AD"]
    for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = best_svm.predict(X)
        acc   = accuracy_score(y, preds)
        print(f"{'='*50}")
        print(f"{split_name} Accuracy: {acc:.4f}")
        print(f"{'='*50}")
        print(classification_report(y, preds, target_names=label_names))
        cm = confusion_matrix(y, preds)
        print(f"Confusion Matrix ({split_name}):")
        print(pd.DataFrame(cm, index=label_names, columns=label_names))
        print()

    # Save
    joblib.dump(best_svm, Path(SAVE_DIR) / "svm_tuned.pkl")
    joblib.dump(scaler,   Path(SAVE_DIR) / "scaler.pkl")
    joblib.dump(pca,      Path(SAVE_DIR) / "pca.pkl")
    print(f"✅ Tuned SVM saved to: {SAVE_DIR}\\svm_tuned.pkl")