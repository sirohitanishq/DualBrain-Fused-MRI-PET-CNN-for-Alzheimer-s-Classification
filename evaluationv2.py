import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from datasetv2 import BrainDataset2D
from modelv2 import BrainCNN2D

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABELS_CSV  = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
MRI_DIR     = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
PET_DIR     = r"C:\JupyterNotebook\MRI_PET\Data\PET"
SAVE_DIR    = r"C:\JupyterNotebook\MRI_PET\v2\Checkpoints"
RESULTS_DIR = r"C:\JupyterNotebook\MRI_PET\v2\Results"
MODEL_PATH  = r"C:\JupyterNotebook\MRI_PET\v2\Checkpoints\best_model.pth"
HISTORY_CSV = r"C:\JupyterNotebook\MRI_PET\v2\Checkpoints\training_history.csv"

BATCH_SIZE  = 64
NUM_WORKERS = 2
# ──────────────────────────────────────────────────────────────────────────────

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

CLASS_NAMES = ["CN", "MCI", "AD"]


def evaluate_subjects(model, dataset):
    """Subject level voting — average softmax probs across all slices."""
    model.eval()
    subject_probs  = defaultdict(list)
    subject_labels = {}

    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_probs = []

    with torch.no_grad():
        for img, labels in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS):
            img = img.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)

    for i, sample in enumerate(dataset.samples):
        sid   = sample["Subject_ID"]
        label = sample["Numeric_Label"]
        subject_probs[sid].append(all_probs[i])
        subject_labels[sid] = label

    y_true, y_pred, y_probs = [], [], []
    for sid, probs_list in subject_probs.items():
        avg_prob = np.mean(probs_list, axis=0)
        pred     = np.argmax(avg_prob)
        y_true.append(subject_labels[sid])
        y_pred.append(pred)
        y_probs.append(avg_prob)

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix (counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title("Confusion Matrix (%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: confusion_matrix.png")


def plot_roc_curves(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ["green", "orange", "red"]

    plt.figure(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Subject Level", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: roc_curves.png")


def plot_training_history():
    if not Path(HISTORY_CSV).exists():
        print("No training history found, skipping...")
        return

    df = pd.read_csv(HISTORY_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="orange")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Slice accuracy
    axes[1].plot(df["epoch"], df["train_acc"],  label="Train Acc",  color="blue")
    axes[1].plot(df["epoch"], df["slice_acc"],  label="Val Slice Acc", color="orange")
    axes[1].set_title("Slice Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subject accuracy
    axes[2].plot(df["epoch"], df["subject_acc"], label="Subject Acc", color="green", lw=2)
    axes[2].plot(df["epoch"], df["cn_subject_acc"],  label="CN",  color="blue",   linestyle="--")
    axes[2].plot(df["epoch"], df["mci_subject_acc"], label="MCI", color="orange", linestyle="--")
    axes[2].plot(df["epoch"], df["ad_subject_acc"],  label="AD",  color="red",    linestyle="--")
    axes[2].set_title("Subject Accuracy", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / "training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: training_history.png")


if __name__ == "__main__":
    # Load test split
    test_df = pd.read_csv(Path(SAVE_DIR) / "test_split.csv")
    print(f"Test subjects: {len(test_df)}")
    print(f"Class distribution:\n{test_df['Class_Label'].value_counts()}\n")

    # Dataset
    test_dataset = BrainDataset2D(test_df, MRI_DIR, PET_DIR, augment=False)

    # Load model
    model = BrainCNN2D(num_classes=3, dropout=0.5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded model from: {MODEL_PATH}\n")

    # Evaluate
    print("Running subject-level voting on test set...")
    y_true, y_pred, y_probs = evaluate_subjects(model, test_dataset)

    # Results
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"TEST RESULTS — Subject Level Voting")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Per class accuracy
    print("Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        mask     = (y_true == i)
        cls_acc  = (y_pred[mask] == i).mean()
        print(f"  {cls}: {cls_acc:.3f} ({cls_acc*100:.1f}%)")

    # Save results
    results_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_CN" : y_probs[:, 0],
        "prob_MCI": y_probs[:, 1],
        "prob_AD" : y_probs[:, 2],
    })
    results_df.to_csv(Path(RESULTS_DIR) / "test_results.csv", index=False)

    # Plots
    print("\nGenerating plots...")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_probs)
    plot_training_history()

    print(f"\n✅ All results saved to: {RESULTS_DIR}")