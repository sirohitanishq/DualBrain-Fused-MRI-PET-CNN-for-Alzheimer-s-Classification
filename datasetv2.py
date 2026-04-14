import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BrainDataset2D(Dataset):
    """
    2D Dataset — extracts every slice from all 3 planes (axial, coronal, sagittal).
    MRI and PET slices are stacked as 2 channels → (2, 128, 128).

    Per subject:
        108 axial + 108 coronal + 108 sagittal = 324 slices
    Total training samples:
        325 subjects × 324 = ~105,300 slices
    """

    def __init__(self, df, mri_dir, pet_dir, augment=False, skip_empty=0.15):
        self.mri_dir    = Path(mri_dir)
        self.pet_dir    = Path(pet_dir)
        self.augment    = augment
        self.samples    = []

        print(f"Building 2D dataset from {len(df)} subjects (3 planes)...")
        for _, row in df.iterrows():
            mri_vol  = np.load(self.mri_dir / row["MRI_File"])
            n_slices = mri_vol.shape[0]  # 128

            start = int(n_slices * skip_empty)
            end   = int(n_slices * (1 - skip_empty))

            # Add slices from all 3 planes
            for plane in ["axial", "coronal", "sagittal"]:
                for slice_idx in range(start, end):
                    self.samples.append({
                        "MRI_File"     : row["MRI_File"],
                        "PET_File"     : row["PET_File"],
                        "Numeric_Label": int(row["Numeric_Label"]),
                        "Class_Label"  : row["Class_Label"],
                        "Subject_ID"   : row["Subject_ID"],
                        "plane"        : plane,
                        "slice_idx"    : slice_idx,
                    })

        print(f"Total 2D slice samples: {len(self.samples):,}\n")

    def __len__(self):
        return len(self.samples)

    def get_slice(self, volume, plane, slice_idx):
        """Extract a single slice from the given plane."""
        if plane == "axial":
            return volume[slice_idx, :, :]        # top-down
        elif plane == "coronal":
            return volume[:, slice_idx, :]        # front-back
        else:  # sagittal
            return volume[:, :, slice_idx]        # left-right

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        label     = sample["Numeric_Label"]
        slice_idx = sample["slice_idx"]
        plane     = sample["plane"]

        mri_vol = np.load(self.mri_dir / sample["MRI_File"]).astype(np.float32)
        pet_vol = np.load(self.pet_dir / sample["PET_File"]).astype(np.float32)

        mri_slice = self.get_slice(mri_vol, plane, slice_idx)  # (128, 128)
        pet_slice = self.get_slice(pet_vol, plane, slice_idx)  # (128, 128)

        # Stack as 2 channels → (2, 128, 128)
        img = np.stack([mri_slice, pet_slice], axis=0)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
            if np.random.rand() > 0.3:
                k   = np.random.choice([1, 2, 3])
                img = np.rot90(img, k=k, axes=(1, 2)).copy()
            img[0] = img[0] + np.random.uniform(-0.1, 0.1)
            img[1] = np.clip(img[1] + np.random.uniform(-0.05, 0.05), 0, 1)

        img   = torch.tensor(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    LABELS_CSV = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
    MRI_DIR    = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
    PET_DIR    = r"C:\JupyterNotebook\MRI_PET\Data\PET"

    df = pd.read_csv(LABELS_CSV)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["Numeric_Label"])

    ds = BrainDataset2D(train_df, MRI_DIR, PET_DIR, augment=False)
    img, label = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Label       : {label}")
    print(f"MRI range   : [{img[0].min():.2f}, {img[0].max():.2f}]")
    print(f"PET range   : [{img[1].min():.2f}, {img[1].max():.2f}]")