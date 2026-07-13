"""
src/explainability/gradcam_cxr.py
-----------------------------------
Grad-CAM visualization for the DenseNet-121 CXR branch.

Highlights which regions of the chest X-ray most influence the
model's readmission risk prediction.

Target layer: last dense block's final convolution
(backbone.features.denseblock4.denselayer16.conv2 for DenseNet-121).

Usage:
    from src.explainability.gradcam_cxr import generate_gradcam
    heatmaps = generate_gradcam(cxr_model, image_paths, hadm_ids, cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Retrieve the target convolutional layer for Grad-CAM.

    For DenseNet-121 from timm, the last dense block's last conv is used.
    Adjusts automatically if the backbone name changes.

    Parameters
    ----------
    model : CXREncoder

    Returns
    -------
    torch.nn.Module — the target layer.
    """
    backbone = model.backbone

    # Try DenseNet-121 specific path
    try:
        target = backbone.features.denseblock4.denselayer16.conv2
        return target
    except AttributeError:
        pass

    # Generic fallback: find the last Conv2d in the backbone
    conv_layers = [
        m for m in backbone.modules()
        if isinstance(m, torch.nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("No Conv2d layer found in backbone for Grad-CAM.")
    log.warning(
        "Could not find DenseNet-121 target layer — using last Conv2d as fallback."
    )
    return conv_layers[-1]


def generate_gradcam(
    cxr_model: torch.nn.Module,
    image_paths: List[str],
    hadm_ids: List[int],
    cfg: DictConfig,
    save_dir: Optional[str] = None,
    top_k: int = 4,
) -> Dict[int, np.ndarray]:
    """
    Generate Grad-CAM heatmaps for a list of CXR images.

    Parameters
    ----------
    cxr_model : CXREncoder
        Trained CXR model.
    image_paths : list of str
        Absolute paths to JPEG images.
    hadm_ids : list of int
        Corresponding admission IDs (for saving/logging).
    cfg : DictConfig
        Project config.
    save_dir : str, optional
        Directory to save overlay images.
    top_k : int
        Maximum number of heatmaps to generate.

    Returns
    -------
    dict : hadm_id → heatmap array (H, W) float32, values in [0, 1].
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError("Install grad-cam: pip install grad-cam")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Install Pillow: pip install Pillow")

    import matplotlib.pyplot as plt
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cxr_model.to(device).eval()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    image_size  = cfg.cxr.image_size
    target_layer = _get_target_layer(cxr_model)

    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    to_rgb = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    # Wrap model so GradCAM targets the backbone's output before the head
    class _BackboneWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x):
            logit, _ = self.encoder(x)
            return logit

    wrapped = _BackboneWrapper(cxr_model)
    cam = GradCAM(model=wrapped, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(0)]

    heatmaps: dict[int, np.ndarray] = {}

    for i, (img_path, hadm_id) in enumerate(zip(image_paths[:top_k], hadm_ids[:top_k])):
        try:
            raw_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning("Failed to load CXR %s for hadm_id %s: %s", img_path, hadm_id, e)
            continue

        # Preprocessed tensor for model
        input_tensor = preprocess(raw_img).unsqueeze(0).to(device)
        # Original RGB (0–1) for overlay
        rgb_np = to_rgb(raw_img).permute(1, 2, 0).numpy()

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]  # (H, W) float32, values in [0,1]

        heatmaps[hadm_id] = grayscale_cam
        overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

        # Get model prediction for title
        with torch.no_grad():
            logit, _ = cxr_model(input_tensor)
            prob = float(torch.sigmoid(logit).item())

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_np)
        axes[0].set_title("Original CXR", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM Overlay | Risk Score: {prob:.3f}", fontsize=11)
        axes[1].axis("off")

        plt.suptitle(f"Patient hadm_id={hadm_id}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_dir:
            fname = Path(save_dir) / f"gradcam_{hadm_id}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            log.info("Grad-CAM saved → %s", fname)

        plt.show()
        plt.close()

    log.info("Grad-CAM generated for %d patients.", len(heatmaps))
    return heatmaps
