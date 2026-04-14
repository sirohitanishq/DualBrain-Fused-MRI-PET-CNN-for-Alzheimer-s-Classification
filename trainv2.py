import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasetv2 import BrainDataset2D
from modelv2 import BrainCNN2D

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABELS_CSV   = r"C:\JupyterNotebook\MRI_PET\Data\labels.csv"
MRI_DIR      = r"C:\JupyterNotebook\MRI_PET\Data\MRI"
PET_DIR      = r"C:\JupyterNotebook\MRI_PET\Data\PET"
SAVE_DIR     = r"C:\JupyterNotebook\MRI_PET\v2\Checkpoints"

BATCH_SIZE   = 64
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
EARLY_STOP   = 7
NUM_WORKERS  = 8
RANDOM_SEED  = 42

# Gradual unfreezing schedule
# epoch : layers to unfreeze
UNFREEZE_SCHEDULE = {
    6  : "layer2",   # unfreeze layer2 at epoch 6
    11 : "layer1",   # unfreeze layer1 at epoch 11
}
# ──────────────────────────────────────────────────────────────────────────────

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}\n")


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
    train_dataset = BrainDataset2D(train_df, MRI_DIR, PET_DIR, augment=True)
    val_dataset   = BrainDataset2D(val_df,   MRI_DIR, PET_DIR, augment=False)

    labels        = [s["Numeric_Label"] for s in train_dataset.samples]
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / np.sqrt(class_counts)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, val_dataset


def unfreeze_layer(model, layer_name):
    """Unfreeze a specific layer by name."""
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  🔓 Unfroze {layer_name} — trainable params now: {trainable:,}")


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for img, labels in tqdm(loader, desc="  Train", leave=False):
        img    = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(img)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate_slices(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]

    with torch.no_grad():
        for img, labels in tqdm(loader, desc="  Val  ", leave=False):
            img    = img.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(img)
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


def evaluate_subjects(model, dataset):
    """Subject-level voting — 324 slices per subject vote for final prediction."""
    model.eval()
    subject_probs  = defaultdict(list)
    subject_labels = {}

    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs = []

    with torch.no_grad():
        for img, labels in tqdm(loader, desc="  Voting", leave=False):
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

    y_true, y_pred = [], []
    for sid, probs_list in subject_probs.items():
        avg_prob = np.mean(probs_list, axis=0)
        pred     = np.argmax(avg_prob)
        y_true.append(subject_labels[sid])
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)

    # Per class subject accuracy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_acc = [
        (y_pred[y_true == i] == i).mean() if (y_true == i).sum() > 0 else 0
        for i in range(3)
    ]
    return acc, class_acc


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df, val_df, test_df = get_splits()
    train_loader, val_loader, val_dataset = get_dataloaders(train_df, val_df)

    print(f"Train subjects : {len(train_df)} → {len(train_df) * 108 * 3:,} slices")
    print(f"Val subjects   : {len(val_df)}   → {len(val_df) * 108 * 3:,} slices")
    print(f"Test subjects  : {len(test_df)}\n")

    test_df.to_csv(Path(SAVE_DIR) / "test_split.csv", index=False)

    model = BrainCNN2D(num_classes=3, dropout=0.5).to(device)
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
    scheduler         = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler            = torch.amp.GradScaler("cuda")
    best_subject_acc  = 0.0        # ← save by subject accuracy now
    patience_count    = 0
    history           = []

    print("Starting training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        # Gradual unfreezing
        if epoch in UNFREEZE_SCHEDULE:
            layer_name = UNFREEZE_SCHEDULE[epoch]
            unfreeze_layer(model, layer_name)
            # Recreate optimizer to include newly unfrozen params
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR * 0.1,        # lower LR for newly unfrozen layers
                weight_decay=WEIGHT_DECAY
            )

        train_loss, train_acc        = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, slice_class_acc = evaluate_slices(model, val_loader, criterion)
        subject_acc, subject_class_acc     = evaluate_subjects(model, val_dataset)
        scheduler.step()

        # Save by subject accuracy ← key change
        if subject_acc > best_subject_acc:
            best_subject_acc = subject_acc
            patience_count   = 0
            torch.save(model.state_dict(), Path(SAVE_DIR) / "best_model.pth")
            saved = "✅ saved"
        else:
            patience_count += 1
            saved = f"(patience {patience_count}/{EARLY_STOP})"

        print(f"  Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}   | Slice Acc : {val_acc:.4f}")
        print(f"  Subject Acc: {subject_acc:.4f}  ← TRUE accuracy  {saved}")
        print(f"  Slice  class → CN: {slice_class_acc[0]:.3f} | MCI: {slice_class_acc[1]:.3f} | AD: {slice_class_acc[2]:.3f}")
        print(f"  Subject class → CN: {subject_class_acc[0]:.3f} | MCI: {subject_class_acc[1]:.3f} | AD: {subject_class_acc[2]:.3f}\n")

        history.append({
            "epoch"          : epoch,
            "train_loss"     : train_loss,
            "train_acc"      : train_acc,
            "val_loss"       : val_loss,
            "slice_acc"      : val_acc,
            "subject_acc"    : subject_acc,
            "cn_subject_acc" : subject_class_acc[0],
            "mci_subject_acc": subject_class_acc[1],
            "ad_subject_acc" : subject_class_acc[2],
        })

        if patience_count >= EARLY_STOP:
            print(f"⛔ Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(Path(SAVE_DIR) / "training_history.csv", index=False)
    print(f"\nTraining complete! Best subject acc: {best_subject_acc:.4f}")
    print(f"Model saved to: {SAVE_DIR}\\best_model.pth")