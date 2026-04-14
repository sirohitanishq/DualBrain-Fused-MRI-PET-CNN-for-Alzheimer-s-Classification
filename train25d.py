import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset25d import BrainDataset25D
from model25d import DualStream25D

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABELS_CSV       = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
MRI_DIR          = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
PET_DIR          = r"C:\JupyterNotebook\MRI_PET\Data\PET"
SAVE_DIR         = r"C:\JupyterNotebook\MRI_PET\Checkpoints25D_v4"

BATCH_SIZE       = 32
NUM_EPOCHS       = 50
LR               = 5e-5        # lower LR
WEIGHT_DECAY     = 5e-4        # stronger L2
GRAD_CLIP        = 1.0
SLICES_PER_PLANE = 20
EARLY_STOP_PAT   = 7           # stop if no improvement for 7 epochs
NUM_WORKERS      = 0
RANDOM_SEED      = 42
# ──────────────────────────────────────────────────────────────────────────────

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def get_splits():
    df = pd.read_csv(LABELS_CSV)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["Numeric_Label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["Numeric_Label"]
    )
    return train_df, val_df, test_df


def get_dataloaders(train_df, val_df):
    class_counts   = train_df["Numeric_Label"].value_counts().sort_index().values
    class_weights  = 1.0 / np.sqrt(class_counts)
    sample_weights = train_df["Numeric_Label"].map(
        {i: class_weights[i] for i in range(len(class_weights))}
    ).values

    train_dataset = BrainDataset25D(train_df, MRI_DIR, PET_DIR,
                                    augment=True, slices_per_plane=SLICES_PER_PLANE)
    val_dataset   = BrainDataset25D(val_df,   MRI_DIR, PET_DIR,
                                    augment=False, slices_per_plane=SLICES_PER_PLANE)

    expanded_weights = np.repeat(sample_weights, SLICES_PER_PLANE * 3)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(expanded_weights, dtype=torch.float),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for mri, pet, labels in tqdm(loader, desc="  Train", leave=False):
        mri    = mri.to(device)
        pet    = pet.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(mri, pet)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]

    with torch.no_grad():
        for mri, pet, labels in tqdm(loader, desc="  Val  ", leave=False):
            mri    = mri.to(device)
            pet    = pet.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(mri, pet)
                loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            for i in range(3):
                mask = (labels == i)
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                class_total[i]   += mask.sum().item()

    class_acc = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(3)
    ]
    return total_loss / len(loader), correct / total, class_acc


if __name__ == "__main__":
    train_df, val_df, test_df = get_splits()
    train_loader, val_loader  = get_dataloaders(train_df, val_df)

    train_samples = len(train_df) * SLICES_PER_PLANE * 3
    val_samples   = len(val_df)   * SLICES_PER_PLANE * 3
    print(f"Train subjects : {len(train_df)} → {train_samples} slice samples")
    print(f"Val subjects   : {len(val_df)}   → {val_samples} slice samples")
    print(f"Test subjects  : {len(test_df)}\n")

    test_df.to_csv(Path(SAVE_DIR) / "test_split.csv", index=False)

    # Higher dropout to fight overfitting
    model = DualStream25D(num_classes=3, dropout=0.6).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total_p:,}\n")

    class_counts  = train_df["Numeric_Label"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / np.sqrt(class_counts), dtype=torch.float).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda")

    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    print("Starting training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        train_loss, train_acc        = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, class_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), Path(SAVE_DIR) / "best_model.pth")
            saved = "✅ saved"
        else:
            patience_count += 1
            saved = f"(patience {patience_count}/{EARLY_STOP_PAT})"

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f}  {saved}")
        print(f"  Per-class Val Acc → CN: {class_acc[0]:.3f} | MCI: {class_acc[1]:.3f} | AD: {class_acc[2]:.3f}\n")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "cn_acc": class_acc[0], "mci_acc": class_acc[1], "ad_acc": class_acc[2],
        })

        # Early stopping
        if patience_count >= EARLY_STOP_PAT:
            print(f"⛔ Early stopping at epoch {epoch} — no improvement for {EARLY_STOP_PAT} epochs")
            break

    pd.DataFrame(history).to_csv(Path(SAVE_DIR) / "training_history.csv", index=False)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {SAVE_DIR}\\best_model.pth")