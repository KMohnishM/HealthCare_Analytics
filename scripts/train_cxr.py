"""
scripts/train_cxr.py
---------------------
Training script for the DenseNet-121 CXR branch.

Run from project root:
    python scripts/train_cxr.py

On Google Colab:
    !pip install timm Pillow
    !python scripts/train_cxr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.cxr.dataset import CXRDataset, make_cxr_loader
from src.cxr.model import CXREncoder, mc_predict_cxr
from src.cxr.preprocess import build_cxr_index
from src.evaluation.metrics import evaluate_all
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, init_mlflow
from src.utils.seed import set_seed

log = get_logger(__name__)


def train_one_epoch(
    model: CXREncoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        img   = batch["image"].to(device)
        lbl   = batch["label"].to(device)
        avail = batch["available"].to(device)

        if avail.sum() == 0:
            continue

        logit, _ = model(img)
        logit = logit.squeeze(-1)
        loss = (criterion(logit, lbl) * avail).sum() / avail.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(
    model: CXREncoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    all_logits, all_labels, all_avail = [], [], []
    total_loss = 0.0

    for batch in loader:
        img   = batch["image"].to(device)
        lbl   = batch["label"].to(device)
        avail = batch["available"].to(device)
        logit, _ = model(img)
        logit = logit.squeeze(-1)
        loss = (criterion(logit, lbl) * avail).sum() / avail.sum().clamp(min=1)
        total_loss += loss.item()
        all_logits.append(logit.cpu().numpy())
        all_labels.append(lbl.cpu().numpy())
        all_avail.append(avail.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    avail  = np.concatenate(all_avail)

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
    set_seed(cfg.cxr.random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    run_id = init_mlflow(cfg, run_name="train_cxr")

    cohort_dir = Path(cfg.paths.cohort_dir)
    train_df   = pd.read_parquet(cohort_dir / "train.parquet")
    val_df     = pd.read_parquet(cohort_dir / "val.parquet")
    test_df    = pd.read_parquet(cohort_dir / "test.parquet")

    # ── Build CXR index ───────────────────────────────────────────────────────
    log.info("Building CXR index …")
    all_cohort = pd.concat([train_df, val_df, test_df], ignore_index=True)
    cxr_index  = build_cxr_index(all_cohort, cfg)
    cxr_index.to_csv(Path(cfg.paths.processed_dir) / "cxr_index.csv", index=False)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = CXRDataset(cxr_index, train_df, cfg, is_train=True)
    val_ds   = CXRDataset(cxr_index, val_df,   cfg, is_train=False)
    test_ds  = CXRDataset(cxr_index, test_df,  cfg, is_train=False)

    # num_workers=0 is safe on Colab
    train_loader = make_cxr_loader(train_ds, batch_size=cfg.cxr.batch_size, shuffle=True,  num_workers=0)
    val_loader   = make_cxr_loader(val_ds,   batch_size=cfg.cxr.batch_size, shuffle=False, num_workers=0)
    test_loader  = make_cxr_loader(test_ds,  batch_size=cfg.cxr.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CXREncoder(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("CXREncoder trainable params: %d", trainable)

    n_pos = int(train_df["readmitted_30d"].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train only head initially (backbone frozen)
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.cxr.lr, weight_decay=cfg.cxr.weight_decay,
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cxr.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_state   = None

    for epoch in range(1, cfg.cxr.epochs + 1):
        train_loss             = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc      = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_AUROC=%.4f",
                epoch, cfg.cxr.epochs, train_loss, val_loss, val_auc,
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    log.info("Best val AUROC: %.4f", best_val_auc)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = Path(cfg.paths.models_dir) / "cxr_densenet.pt"
    torch.save({"model_state": model.state_dict()}, model_path)
    log.info("CXR model saved → %s", model_path)

    # ── Generate MC-Dropout embeddings for fusion ─────────────────────────────
    proc_dir = Path(cfg.paths.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    for split_name, loader, split_df in [
        ("train", train_loader, train_df),
        ("val",   val_loader,   val_df),
        ("test",  test_loader,  test_df),
    ]:
        log.info("Generating MC-Dropout predictions for %s split …", split_name)
        all_probs, all_confs, all_embeds, all_avail_flags = [], [], [], []

        for batch in loader:
            img   = batch["image"]
            avail = batch["available"].numpy()
            result = mc_predict_cxr(model, img, n_passes=cfg.cxr.mc_passes, device=device)
            all_probs.append(result["prob"])
            all_confs.append(result["confidence"])
            all_embeds.append(result["embed"])
            all_avail_flags.append(avail)

        probs  = np.concatenate(all_probs)
        confs  = np.concatenate(all_confs)
        embeds = np.concatenate(all_embeds, axis=0)
        avails = np.concatenate(all_avail_flags)

        out_df = pd.DataFrame({
            "hadm_id":    split_df["hadm_id"].values,
            "score":      probs,
            "confidence": confs,
            "available":  avails,
            "label":      split_df["readmitted_30d"].values,
        })
        out_df.to_csv(proc_dir / f"cxr_preds_{split_name}.csv", index=False)
        np.save(proc_dir / f"cxr_embed_{split_name}.npy", embeds.astype("float32"))
        log.info("Saved %s CXR outputs → %s", split_name, proc_dir)

    # ── Test metrics ──────────────────────────────────────────────────────────
    test_preds_df = pd.read_csv(proc_dir / "cxr_preds_test.csv")
    avail_mask    = test_preds_df["available"] > 0.5
    metrics = evaluate_all(
        test_preds_df.loc[avail_mask, "label"].values,
        test_preds_df.loc[avail_mask, "score"].values,
    )
    log.info("=" * 60)
    log.info("TEST RESULTS — CXR Branch (available CXR only, n=%d)", avail_mask.sum())
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)
    log.info("=" * 60)

    log.info("CXR branch training complete ✓")


if __name__ == "__main__":
    main()
