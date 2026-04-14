import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BrainDataset25D(Dataset):
    """
    2.5D Dataset — extracts multiple overlapping triplets of slices
    from each MRI and PET volume.

    For each volume we sample N triplets per plane (axial, coronal, sagittal).
    Each triplet = 3 consecutive slices stacked as channels → (3, 128, 128)

    With N=10 triplets × 3 planes = 30 samples per brain volume.
    465 volumes × 30 = 13,950 total training samples.
    """

    def __init__(self, df, mri_dir, pet_dir, augment=False, slices_per_plane=10):
        self.mri_dir         = Path(mri_dir)
        self.pet_dir         = Path(pet_dir)
        self.augment         = augment
        self.slices_per_plane = slices_per_plane

        # Expand dataset: each row becomes multiple slice samples
        self.samples = []
        for _, row in df.iterrows():
            # Generate slice indices for each plane
            # We skip the first and last 10% to avoid empty border regions
            for plane in ["axial", "coronal", "sagittal"]:
                for i in range(slices_per_plane):
                    self.samples.append({
                        "MRI_File"     : row["MRI_File"],
                        "PET_File"     : row["PET_File"],
                        "Numeric_Label": row["Numeric_Label"],
                        "Class_Label"  : row["Class_Label"],
                        "plane"        : plane,
                        "slice_idx"    : i,   # will be mapped to actual index at load time
                    })

    def __len__(self):
        return len(self.samples)

    def get_triplet(self, volume, plane, slice_idx, n_slices):
        """
        Extract a triplet of consecutive slices from a given plane.
        Evenly spaces n_slices triplets across the middle 80% of the volume.
        """
        d = volume.shape[0]  # depth of this plane

        # Use middle 80% of volume to avoid empty borders
        start = int(d * 0.10)
        end   = int(d * 0.90) - 2   # -2 to allow triplet

        # Evenly space slice positions
        positions = np.linspace(start, end, n_slices, dtype=int)
        pos = positions[slice_idx]

        # Extract triplet: pos-1, pos, pos+1
        s0 = volume[max(0, pos-1)]
        s1 = volume[pos]
        s2 = volume[min(d-1, pos+1)]

        return np.stack([s0, s1, s2], axis=0)  # (3, H, W)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label  = int(sample["Numeric_Label"])
        plane  = sample["plane"]
        s_idx  = sample["slice_idx"]

        # Load volumes
        mri_vol = np.load(self.mri_dir / sample["MRI_File"]).astype(np.float32)
        pet_vol = np.load(self.pet_dir / sample["PET_File"]).astype(np.float32)

        # Rotate volume so correct plane is along axis 0
        if plane == "axial":
            mri_plane = mri_vol              # (D, H, W) — already axial
            pet_plane = pet_vol
        elif plane == "coronal":
            mri_plane = mri_vol.transpose(1, 0, 2)   # (H, D, W)
            pet_plane = pet_vol.transpose(1, 0, 2)
        else:  # sagittal
            mri_plane = mri_vol.transpose(2, 0, 1)   # (W, D, H)
            pet_plane = pet_vol.transpose(2, 0, 1)

        # Extract triplet → (3, 128, 128)
        mri = self.get_triplet(mri_plane, plane, s_idx, self.slices_per_plane)
        pet = self.get_triplet(pet_plane, plane, s_idx, self.slices_per_plane)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                mri = np.flip(mri, axis=2).copy()
                pet = np.flip(pet, axis=2).copy()
            if np.random.rand() > 0.5:
                mri = np.flip(mri, axis=1).copy()
                pet = np.flip(pet, axis=1).copy()
            if np.random.rand() > 0.5:
                mri = np.rot90(mri, k=1, axes=(1, 2)).copy()
                pet = np.rot90(pet, k=1, axes=(1, 2)).copy()
            mri = mri + np.random.uniform(-0.1, 0.1)
            pet = np.clip(pet + np.random.uniform(-0.05, 0.05), 0, 1)

        mri   = torch.tensor(mri)
        pet   = torch.tensor(pet)
        label = torch.tensor(label, dtype=torch.long)

        return mri, pet, label


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(r"C:\JupyterNotebook\MRI_PET\Data\labels.csv")
    ds = BrainDataset25D(df,
                         r"C:\JupyterNotebook\MRI_PET\Data\MRI",
                         r"C:\JupyterNotebook\MRI_PET\Data\PET",
                         augment=False, slices_per_plane=10)
    print(f"Total samples : {len(ds)}")
    mri, pet, label = ds[0]
    print(f"MRI shape     : {mri.shape}")
    print(f"PET shape     : {pet.shape}")
    print(f"Label         : {label}")
    print(f"\n465 volumes × 10 slices × 3 planes = {465*10*3} total samples")