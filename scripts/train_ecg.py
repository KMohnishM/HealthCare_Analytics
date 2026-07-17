"""
scripts/train_ecg.py
---------------------
Training script for the 1D ResNet ECG branch.

Run from project root:
    python scripts/train_ecg.py

On Google Colab:
    !pip install wfdb scipy
    !python scripts/train_ecg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.ecg.dataset import ECGDataset, make_ecg_loader
from src.ecg.model import ECGResNet, mc_predict_ecg
from src.ecg.preprocess import build_ecg_index
from src.evaluation.metrics import evaluate_all
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, init_mlflow
from src.utils.seed import set_seed

log = get_logger(__name__)


def train_one_epoch(
    model: ECGResNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        wav  = batch["waveform"].to(device)
        lbl  = batch["label"].to(device)
        avail = batch["available"].to(device)

        # Only compute loss for patients with available ECG
        if avail.sum() == 0:
            continue

        logit, _ = model(wav)
        logit = logit.squeeze(-1)

        # Mask loss to available patients only
        loss = (criterion(logit, lbl) * avail).sum() / avail.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(
    model: ECGResNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    all_logits, all_labels, all_avail = [], [], []
    total_loss = 0.0

    for batch in loader:
        wav   = batch["waveform"].to(device)
        lbl   = batch["label"].to(device)
        avail = batch["available"].to(device)
        logit, _ = model(wav)
        logit = logit.squeeze(-1)
        loss = (criterion(logit, lbl) * avail).sum() / avail.sum().clamp(min=1)
        total_loss += loss.item()
        all_logits.append(logit.cpu().numpy())
        all_labels.append(lbl.cpu().numpy())
        all_avail.append(avail.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    avail  = np.concatenate(all_avail)

    # AUROC only on available ECG patients
    from sklearn.metrics import roc_auc_score
    mask = avail > 0.5
    try:
        auc = roc_auc_score(labels[mask], 1 / (1 + np.exp(-logits[mask])))
    except Exception:
        auc = 0.5

    return total_loss / max(len(loader), 1), auc


def main() -> None:
    cfg    = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.ecg.random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    run_id = init_mlflow(cfg, run_name="train_ecg")

    cohort_dir = Path(cfg.paths.cohort_dir)
    train_df   = pd.read_parquet(cohort_dir / "train.parquet")
    val_df     = pd.read_parquet(cohort_dir / "val.parquet")
    test_df    = pd.read_parquet(cohort_dir / "test.parquet")

    # ── Build ECG index ───────────────────────────────────────────────────────
    log.info("Building ECG index …")
    all_cohort = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ecg_index  = build_ecg_index(all_cohort, cfg)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ECGDataset(ecg_index, train_df, cfg)
    val_ds   = ECGDataset(ecg_index, val_df,   cfg)
    test_ds  = ECGDataset(ecg_index, test_df,  cfg)
    # ── Out-Of-Fold (OOF) Generation for Stacking (Fusion) ───────────────────
    log.info("Generating Out-Of-Fold ECG predictions via 5-fold CV to prevent Stacking Leakage …")
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.ecg.random_seed)
    
    oof_probs  = np.zeros(len(train_df), dtype=np.float32)
    oof_confs  = np.zeros(len(train_df), dtype=np.float32)
    oof_embeds = np.zeros((len(train_df), cfg.ecg.embed_dim), dtype=np.float32)
    oof_avails = np.zeros(len(train_df), dtype=np.float32)

    # Convert train_df to reset index for indexing convenience in folds
    train_df_reset = train_df.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_reset)):
        log.info(f"--- Fold {fold + 1}/5 ---")
        train_sub = train_df_reset.iloc[train_idx]
        val_sub   = train_df_reset.iloc[val_idx]
        
        fold_train_ds = ECGDataset(ecg_index, train_sub, cfg)
        fold_val_ds   = ECGDataset(ecg_index, val_sub,   cfg)
        
        fold_train_loader = make_ecg_loader(fold_train_ds, batch_size=cfg.ecg.batch_size, shuffle=True)
        fold_val_loader   = make_ecg_loader(fold_val_ds,   batch_size=cfg.ecg.batch_size)
        
        fold_model = ECGResNet(cfg).to(device)
        fold_optimizer = optim.AdamW(fold_model.parameters(), lr=cfg.ecg.lr, weight_decay=cfg.ecg.weight_decay)
        fold_criterion = nn.BCEWithLogitsLoss()
        
        # Fast training (5 epochs) for CV fold to gather OOF embeddings quickly
        for epoch in range(1, 6):
            _ = train_one_epoch(fold_model, fold_train_loader, fold_optimizer, fold_criterion, device)
            
        # Collect OOF predictions
        fold_model.eval()
        for batch in fold_val_loader:
            wav   = batch["waveform"]
            avail = batch["available"].numpy()
            hadm_ids = batch["hadm_id"].numpy()
            
            result = mc_predict_ecg(fold_model, wav, n_passes=10, device=device) # fewer passes for speed in CV
            
            # Map back to train_df_reset indices
            for idx_in_batch, h_id in enumerate(hadm_ids):
                # find index in train_df_reset
                orig_idx = train_df_reset[train_df_reset["hadm_id"] == h_id].index[0]
                oof_probs[orig_idx]  = result["prob"][idx_in_batch]
                oof_confs[orig_idx]  = result["confidence"][idx_in_batch]
                oof_embeds[orig_idx] = result["embed"][idx_in_batch]
                oof_avails[orig_idx] = avail[idx_in_batch]

    log.info("ECG OOF prediction generation complete.")

    # ── Train final model on full training set ───────────────────────────────
    log.info("Training final ECGResNet model on full training set …")
    model = ECGResNet(cfg).to(device)
    log.info(
        "ECGResNet params: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    n_pos = int(train_df["readmitted_30d"].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = optim.AdamW(model.parameters(), lr=cfg.ecg.lr, weight_decay=cfg.ecg.weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.ecg.epochs)

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(1, cfg.ecg.epochs + 1):
        train_loss             = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc      = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_AUROC=%.4f",
                epoch, cfg.ecg.epochs, train_loss, val_loss, val_auc,
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    log.info("Best val AUROC: %.4f", best_val_auc)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = Path(cfg.paths.models_dir) / "ecg_resnet.pt"
    torch.save({"model_state": model.state_dict(), "cfg": dict(cfg.ecg)}, model_path)
    log.info("ECG model saved → %s", model_path)

    # ── Generate embeddings and predictions for fusion ────────────────────────
    proc_dir = Path(cfg.paths.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF train split outputs
    train_out_df = pd.DataFrame({
        "hadm_id":    train_df_reset["hadm_id"].values,
        "score":      oof_probs,
        "confidence": oof_confs,
        "available":  oof_avails,
        "label":      train_df_reset["readmitted_30d"].values,
    })
    train_out_df.to_csv(proc_dir / "ecg_preds_train.csv", index=False)
    np.save(proc_dir / "ecg_embed_train.npy", oof_embeds.astype("float32"))
    log.info("Saved train ECG OOF outputs → %s", proc_dir)

    # Save val & test outputs using the final trained model
    for split_name, loader, split_df in [
        ("val",   val_loader,   val_df),
        ("test",  test_loader,  test_df),
    ]:
        log.info("Generating MC-Dropout predictions for %s split …", split_name)
        all_probs, all_confs, all_embeds, all_avail_flags = [], [], [], []

        model.eval()
        for batch in loader:
            wav   = batch["waveform"]
            avail = batch["available"].numpy()

            result = mc_predict_ecg(model, wav, n_passes=cfg.ecg.mc_passes, device=device)
            all_probs.append(result["prob"])
            all_confs.append(result["confidence"])
            all_embeds.append(result["embed"])
            all_avail_flags.append(avail)

        probs  = np.concatenate(all_probs)
        confs  = np.concatenate(all_confs)
        embeds = np.concatenate(all_embeds, axis=0)
        avails = np.concatenate(all_avail_flags)

        hadm_ids = split_df["hadm_id"].values
        out_df = pd.DataFrame({
            "hadm_id":    hadm_ids,
            "score":      probs,
            "confidence": confs,
            "available":  avails,
            "label":      split_df["readmitted_30d"].values,
        })
        out_df.to_csv(proc_dir / f"ecg_preds_{split_name}.csv", index=False)
        np.save(proc_dir / f"ecg_embed_{split_name}.npy", embeds.astype("float32"))
        log.info("Saved %s ECG outputs → %s", split_name, proc_dir)

    # ── Test evaluation (on available ECG patients) ──────────────────────────
    test_preds_df = pd.read_csv(proc_dir / "ecg_preds_test.csv")
    avail_mask    = test_preds_df["available"] > 0.5
    metrics = evaluate_all(
        test_preds_df.loc[avail_mask, "label"].values,
        test_preds_df.loc[avail_mask, "score"].values,
    )
    log.info("=" * 60)
    log.info("TEST RESULTS — ECG Branch (available ECG only, n=%d)", avail_mask.sum())
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)
    log.info("=" * 60)

    log.info("ECG branch training complete ✓")


if __name__ == "__main__":
    main()
