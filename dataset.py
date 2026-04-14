import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABELS_CSV  = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
MRI_DIR     = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
PET_DIR     = r"C:\JupyterNotebook\MRI_PET\Data\PET"
BATCH_SIZE  = 4
NUM_WORKERS = 0   # keep 0 on Windows to avoid multiprocessing issues
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


class BrainDataset(Dataset):
    def __init__(self, df, mri_dir, pet_dir, augment=False):
        """
        df       : dataframe with columns [MRI_File, PET_File, Numeric_Label]
        mri_dir  : path to MRI .npy files
        pet_dir  : path to PET .npy files
        augment  : if True, apply random flips (for training only)
        """
        self.df      = df.reset_index(drop=True)
        self.mri_dir = Path(mri_dir)
        self.pet_dir = Path(pet_dir)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load MRI and PET arrays
        mri = np.load(self.mri_dir / row["MRI_File"]).astype(np.float32)
        pet = np.load(self.pet_dir / row["PET_File"]).astype(np.float32)
        label = int(row["Numeric_Label"])

        # Add channel dimension → (1, 128, 128, 128)
        mri = np.expand_dims(mri, axis=0)
        pet = np.expand_dims(pet, axis=0)

        # Augmentation (training only) — random flips along each axis
        if self.augment:
            if np.random.rand() > 0.5:
                mri = np.flip(mri, axis=1).copy()
                pet = np.flip(pet, axis=1).copy()
            if np.random.rand() > 0.5:
                mri = np.flip(mri, axis=2).copy()
                pet = np.flip(pet, axis=2).copy()
            if np.random.rand() > 0.5:
                mri = np.flip(mri, axis=3).copy()
                pet = np.flip(pet, axis=3).copy()

        # Convert to tensors
        mri   = torch.tensor(mri)
        pet   = torch.tensor(pet)
        label = torch.tensor(label, dtype=torch.long)

        return mri, pet, label


def get_dataloaders():
    # Load labels
    df = pd.read_csv(LABELS_CSV)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['Class_Label'].value_counts()}\n")

    # ── Train / Val / Test split (70 / 15 / 15) ──────────────────────────────
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["Numeric_Label"]   # preserve class ratio in each split
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["Numeric_Label"]
    )

    print(f"Train size : {len(train_df)}")
    print(f"Val size   : {len(val_df)}")
    print(f"Test size  : {len(test_df)}\n")

    # ── Weighted Sampler for training (handles AD imbalance) ──────────────────
    class_counts  = train_df["Numeric_Label"].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = train_df["Numeric_Label"].map(
        {i: class_weights[i] for i in range(len(class_weights))}
    ).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_df),
        replacement=True
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = BrainDataset(train_df, MRI_DIR, PET_DIR, augment=True)
    val_dataset   = BrainDataset(val_df,   MRI_DIR, PET_DIR, augment=False)
    test_dataset  = BrainDataset(test_df,  MRI_DIR, PET_DIR, augment=False)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,          # use weighted sampler instead of shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True           # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    # Load one batch and verify shapes
    mri_batch, pet_batch, labels = next(iter(train_loader))
    print(f"MRI batch shape   : {mri_batch.shape}")   # (4, 1, 128, 128, 128)
    print(f"PET batch shape   : {pet_batch.shape}")   # (4, 1, 128, 128, 128)
    print(f"Labels            : {labels}")
    print(f"MRI value range   : [{mri_batch.min():.2f}, {mri_batch.max():.2f}]")
    print(f"PET value range   : [{pet_batch.min():.2f}, {pet_batch.max():.2f}]")
    print("\n✅ Dataloader working correctly!")