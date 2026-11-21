"""
Grad-CAM and Integrated Gradients implementation for leaf nutrient
deficiency models.

Grad-CAM (Gradient-weighted Class Activation Mapping) generates coarse
heatmaps showing which leaf regions the model focuses on.

Integrated Gradients provides pixel-level attribution (SHAP-like) that
satisfies the completeness and sensitivity axioms.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ------------------------------------------------------------------ #
#                          Grad-CAM                                    #
# ------------------------------------------------------------------ #

class GradCAM:
    """Grad-CAM: Visual Explanations from Deep Networks.

    Registers forward/backward hooks on a target convolutional layer
    to capture feature-map activations and their gradients, then
    combines them into a class-discriminative heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    # -- hooks --------------------------------------------------------
    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    # -- main API -----------------------------------------------------
    def generate(self, input_tensor, class_idx):
        """Return a [H, W] numpy heatmap in [0, 1] for *class_idx*.

        Parameters
        ----------
        input_tensor : Tensor [1, C, H, W]
            Pre-processed image (must allow grad propagation).
        class_idx : int
            Index of the target class in the classifier head.
        """
        self.model.eval()
        input_tensor = input_tensor.detach().requires_grad_(True)

        output = self.model(input_tensor)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM: global-average-pool the gradients, weight the
        # activations, sum, ReLU, normalise.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Up-sample to input resolution
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode="bilinear", align_corners=False,
        )

        cam = cam[0, 0]
        lo, hi = cam.min(), cam.max()
        if (hi - lo) > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ------------------------------------------------------------------ #
#                   Target-layer detection                             #
# ------------------------------------------------------------------ #

def get_target_layer(backbone):
    """Find the last spatial feature-map layer in a timm EfficientNet.

    We look for ``bn2`` (BatchNormAct2d after the final 1×1 conv) which
    retains full spatial resolution before global-average pooling.
    """
    for attr in ("bn2", "act2", "conv_head"):
        layer = getattr(backbone, attr, None)
        if layer is not None:
            return layer
    # Fallback: last inverted-residual block
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        return backbone.blocks[-1]
    raise ValueError("Could not find a suitable Grad-CAM target layer")


# ------------------------------------------------------------------ #
#                     Heatmap helpers                                  #
# ------------------------------------------------------------------ #

def overlay_heatmap(image_np, heatmap, alpha=0.45, colormap="jet"):
    """Blend a [0-1] heatmap onto an RGB image using a matplotlib cmap.

    Parameters
    ----------
    image_np : ndarray [H, W, 3]  in [0, 1]
    heatmap  : ndarray [H_cam, W_cam] in [0, 1]
    alpha    : float – blending factor for the heatmap
    colormap : str – matplotlib colormap name

    Returns
    -------
    ndarray [H, W, 3] in [0, 1]
    """
    h, w = image_np.shape[:2]

    # Resize heatmap to image dims
    hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    hm_resized = np.array(hm_pil.resize((w, h), Image.BILINEAR)) / 255.0

    cmap = mpl_cm.get_cmap(colormap)
    coloured = cmap(hm_resized)[:, :, :3]  # drop alpha

    blended = (1 - alpha) * image_np + alpha * coloured
    return np.clip(blended, 0, 1)


def create_combined_heatmap(cam_maps, ranked_top):
    """Probability-weighted combination of per-class Grad-CAM maps."""
    combined = None
    total_w = 0.0
    for cls, prob in ranked_top:
        if cls in cam_maps:
            w = float(prob)
            combined = w * cam_maps[cls] if combined is None else combined + w * cam_maps[cls]
            total_w += w
    if combined is not None and total_w > 0:
        combined /= total_w
        lo, hi = combined.min(), combined.max()
        if (hi - lo) > 1e-8:
            combined = (combined - lo) / (hi - lo)
    else:
        combined = np.zeros((384, 384))
    return combined


# ------------------------------------------------------------------ #
#          Integrated Gradients  (SHAP-like attribution)               #
# ------------------------------------------------------------------ #

def compute_integrated_gradients(model, input_tensor, class_idx,
                                  n_steps=50, device=None):
    """Pixel-level attribution via Integrated Gradients.

    This method satisfies the SHAP completeness and sensitivity axioms
    and does **not** require the ``shap`` package.

    Parameters
    ----------
    model : nn.Module
    input_tensor : Tensor [1, C, H, W]
    class_idx : int
    n_steps : int – interpolation steps (higher = more precise)

    Returns
    -------
    attr_map : ndarray [H, W] – signed attribution per pixel
    """
    if device is None:
        device = input_tensor.device

    baseline = torch.zeros_like(input_tensor)
    diff = input_tensor - baseline

    total_grads = torch.zeros_like(input_tensor)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interp = (baseline + alpha * diff).detach().requires_grad_(True)

        out = model(interp)
        model.zero_grad()
        out[0, class_idx].backward()

        total_grads += interp.grad.detach()

    avg_grads = total_grads / (n_steps + 1)
    ig = diff.detach() * avg_grads  # Integrated Gradients

    # Collapse channels → spatial map
    attr_map = ig[0].sum(dim=0).cpu().numpy()
    return attr_map


def save_attribution_image(original_img, attr_map, output_path,
                            title="Integrated Gradients (SHAP-like) Attribution"):
    """Save a three-panel attribution visualisation.

    Panels: Original | Signed attribution map | Positive-overlay.
    """
    if not HAS_MPL:
        print("[WARN] matplotlib not available – skipping attribution image")
        return

    BG = "#0d1117"
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)

    img_np = np.array(original_img.resize((384, 384), Image.LANCZOS)) / 255.0

    # 1) Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Leaf", color="white", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # 2) Signed attribution (diverging cmap)
    abs_max = max(abs(attr_map.min()), abs(attr_map.max()), 1e-8)
    # Resize attr_map to 384x384
    attr_resized = np.array(
        Image.fromarray(attr_map).resize((384, 384), Image.BILINEAR)
    )
    im = axes[1].imshow(attr_resized, cmap="seismic",
                         vmin=-abs_max, vmax=abs_max)
    axes[1].set_title("Signed Attribution Map", color="white",
                       fontsize=14, fontweight="bold")
    axes[1].axis("off")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # 3) Positive-attribution overlay
    pos = np.clip(attr_resized, 0, None)
    pos_max = pos.max()
    if pos_max > 1e-8:
        pos = pos / pos_max
    overlay = overlay_heatmap(img_np, pos, alpha=0.5, colormap="hot")
    axes[2].imshow(overlay)
    axes[2].set_title("Positive Attributions Overlay", color="white",
                       fontsize=14, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=18, color="#58a6ff", fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
