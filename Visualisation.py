import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABELS_CSV = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
MRI_DIR    = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
PET_DIR    = r"C:\JupyterNotebook\MRI_PET\Data\PET"
SAVE_DIR   = r"C:\JupyterNotebook\MRI_PET\Visualizations"
# ──────────────────────────────────────────────────────────────────────────────

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
df = pd.read_csv(LABELS_CSV)


def load_pair(row):
    mri = np.load(Path(MRI_DIR) / row["MRI_File"])
    pet = np.load(Path(PET_DIR) / row["PET_File"])
    return mri, pet


# ─── PLOT 1: MRI Slices for one subject per class ────────────────────────────
def plot_mri_slices():
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("MRI Brain Slices — One Subject per Class", fontsize=16, fontweight="bold")

    classes = {"CN": 0, "MCI": 1, "AD": 2}
    for row_idx, (class_name, _) in enumerate(classes.items()):
        sample = df[df["Class_Label"] == class_name].iloc[0]
        mri, _ = load_pair(sample)

        d = mri.shape[0]
        axial    = mri[d // 2, :, :]
        coronal  = mri[:, d // 2, :]
        sagittal = mri[:, :, d // 2]

        for col_idx, (plane, img) in enumerate(zip(
            ["Axial", "Coronal", "Sagittal"],
            [axial, coronal, sagittal]
        )):
            ax = axes[row_idx][col_idx]
            ax.imshow(img, cmap="gray", origin="lower")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=13, fontweight="bold", rotation=90, labelpad=10)
                ax.yaxis.set_label_coords(-0.1, 0.5)
            if row_idx == 0:
                ax.set_title(plane, fontsize=12)

    plt.tight_layout()
    plt.savefig(Path(SAVE_DIR) / "mri_slices_per_class.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: mri_slices_per_class.png")


# ─── PLOT 2: MRI vs PET side by side for same subject ────────────────────────
def plot_mri_vs_pet():
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    fig.suptitle("MRI vs PET — Side by Side per Class", fontsize=16, fontweight="bold")

    classes = ["CN", "MCI", "AD"]
    planes  = ["Axial", "Coronal", "Sagittal"]

    for row_idx, class_name in enumerate(classes):
        sample   = df[df["Class_Label"] == class_name].iloc[0]
        mri, pet = load_pair(sample)
        d = mri.shape[0]

        slices_mri = [mri[d//2, :, :], mri[:, d//2, :], mri[:, :, d//2]]
        slices_pet = [pet[d//2, :, :], pet[:, d//2, :], pet[:, :, d//2]]

        for col_idx in range(3):
            # MRI
            ax_mri = axes[row_idx][col_idx]
            ax_mri.imshow(slices_mri[col_idx], cmap="gray", origin="lower")
            ax_mri.axis("off")
            if row_idx == 0:
                ax_mri.set_title(f"MRI\n{planes[col_idx]}", fontsize=11)
            if col_idx == 0:
                ax_mri.set_ylabel(class_name, fontsize=13, fontweight="bold")

            # PET
            ax_pet = axes[row_idx][col_idx + 3]
            ax_pet.imshow(slices_pet[col_idx], cmap="hot", origin="lower")
            ax_pet.axis("off")
            if row_idx == 0:
                ax_pet.set_title(f"PET\n{planes[col_idx]}", fontsize=11)

    plt.tight_layout()
    plt.savefig(Path(SAVE_DIR) / "mri_vs_pet_per_class.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: mri_vs_pet_per_class.png")


# ─── PLOT 3: Slice intensity across depth ────────────────────────────────────
def plot_intensity_profile():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Mean MRI Intensity Across Axial Slices per Class", fontsize=14, fontweight="bold")

    colors  = {"CN": "green", "MCI": "orange", "AD": "red"}
    classes = ["CN", "MCI", "AD"]

    for ax, class_name in zip(axes, classes):
        class_samples = df[df["Class_Label"] == class_name].head(5)
        for _, row in class_samples.iterrows():
            mri, _ = load_pair(row)
            mean_intensity = mri.mean(axis=(1, 2))   # mean per axial slice
            ax.plot(mean_intensity, alpha=0.6, color=colors[class_name])
        ax.set_title(class_name, fontsize=13, color=colors[class_name], fontweight="bold")
        ax.set_xlabel("Slice index")
        ax.set_ylabel("Mean intensity")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(SAVE_DIR) / "intensity_profile.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: intensity_profile.png")


# ─── PLOT 4: Dataset class distribution ──────────────────────────────────────
def plot_class_distribution():
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Class_Label"].value_counts()[["CN", "MCI", "AD"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars   = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of subjects")
    ax.set_xlabel("Class")
    ax.set_ylim(0, max(counts.values) + 30)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(SAVE_DIR) / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: class_distribution.png")


# ─── PLOT 5: MRI scroll through all axial slices for one subject ─────────────
def plot_all_axial_slices():
    sample   = df[df["Class_Label"] == "AD"].iloc[0]
    mri, _   = load_pair(sample)
    n_slices = mri.shape[0]

    cols = 10
    rows = (n_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2))
    fig.suptitle(f"All Axial MRI Slices — {sample['Subject_ID']} ({sample['Class_Label']})",
                 fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < n_slices:
            ax.imshow(mri[i], cmap="gray", origin="lower")
            ax.set_title(str(i), fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(Path(SAVE_DIR) / "all_axial_slices.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("✅ Saved: all_axial_slices.png")


# ─── RUN ALL ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating visualizations...\n")
    plot_class_distribution()
    plot_mri_slices()
    plot_mri_vs_pet()
    plot_intensity_profile()
    plot_all_axial_slices()
    print(f"\n✅ All plots saved to: {SAVE_DIR}")