"""
Dashboard visualisation for crop nutrient deficiency analysis.

Creates a dark-themed composite output image inspired by clinical
radiology dashboards, adapted for agricultural leaf analysis:

  Row 1 — Original leaf | Combined Grad-CAM | Top-3 class heatmaps
  Row 2 — Overall severity banner
  Row 3 — Probability bar chart (left) | Deficiency summary (right)
"""

import os
import textwrap
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import cm as mpl_cm
    from PIL import Image
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .gradcam import overlay_heatmap

# ------------------------------------------------------------------ #
#  Dark-theme colour palette                                           #
# ------------------------------------------------------------------ #
BG_COLOR     = "#0d1117"
BG_CARD      = "#161b22"
TEXT_PRIMARY  = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
ACCENT_BLUE  = "#58a6ff"
ACCENT_RED   = "#f85149"
ACCENT_ORANGE = "#d29922"
ACCENT_GREEN = "#3fb950"
ACCENT_YELLOW = "#e3b341"
BORDER       = "#30363d"


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _bar_color(prob):
    if prob >= 0.60:
        return ACCENT_RED
    if prob >= 0.40:
        return ACCENT_ORANGE
    if prob >= 0.20:
        return ACCENT_YELLOW
    return ACCENT_GREEN


def _severity(ranked, has_healthy=True):
    """Return (label, colour, one-liner)."""
    top_cls, top_p = ranked[0]
    if top_cls == "Healthy":
        if top_p >= 0.70:
            return "LOW", ACCENT_GREEN, "Leaf appears healthy — no deficiency detected"
        if top_p >= 0.50:
            return "LOW–MODERATE", ACCENT_YELLOW, "Likely healthy, minor uncertainty"
        return "MODERATE", ACCENT_ORANGE, "Uncertain — possible early deficiency"
    if not has_healthy:          # rice (no Healthy class)
        if top_p >= 0.70:
            return "HIGH", ACCENT_RED, f"Strong {top_cls} deficiency signal"
        if top_p >= 0.50:
            return "MODERATE", ACCENT_ORANGE, f"Moderate {top_cls} deficiency signal"
        return "UNCERTAIN", ACCENT_YELLOW, "Mixed deficiency signals — retest advised"
    # deficiency is top class and Healthy exists
    if top_p >= 0.70:
        return "HIGH", ACCENT_RED, f"Strong {top_cls} deficiency detected"
    if top_p >= 0.50:
        return "MODERATE–HIGH", ACCENT_ORANGE, f"Likely {top_cls} deficiency"
    if top_p >= 0.35:
        return "MODERATE", ACCENT_YELLOW, f"Possible {top_cls} deficiency"
    return "UNCERTAIN", ACCENT_YELLOW, "Inconclusive — multiple possible conditions"


def _title_color(prob):
    if prob >= 0.50:
        return ACCENT_RED
    if prob >= 0.30:
        return ACCENT_ORANGE
    return ACCENT_YELLOW


# ------------------------------------------------------------------ #
#  Main dashboard builder                                              #
# ------------------------------------------------------------------ #

def create_dashboard(original_img, cam_maps, combined_cam, ranked,
                     crop_name, image_name, output_path,
                     recommendations_dict=None):
    """Create and save a composite analysis dashboard.

    Parameters
    ----------
    original_img : PIL.Image
    cam_maps : dict  class_name → ndarray [H, W] in [0, 1]
    combined_cam : ndarray [H, W]
    ranked : list of (class_name, probability)
    crop_name : str   display name like "Ash Gourd"
    image_name : str  original file name
    output_path : str where to write the PNG
    recommendations_dict : dict  class_name → dict with keys
        region, description, action, severity_note
    """
    if not HAS_MPL:
        print("[WARN] matplotlib not available — cannot create dashboard")
        return None

    display = (384, 384)
    img_resized = original_img.resize(display, Image.LANCZOS)
    img_np = np.array(img_resized) / 255.0

    has_healthy = any(c == "Healthy" for c, _ in ranked)
    sev_label, sev_color, sev_desc = _severity(ranked, has_healthy)
    top_n = min(3, len(ranked))

    # ---- matplotlib rcParams ----------------------------------------
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_CARD,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "font.family": "sans-serif",
    })

    fig = plt.figure(figsize=(28, 18))
    fig.suptitle(
        f"{crop_name} Nutrient Deficiency Analysis — {image_name}",
        fontsize=22, fontweight="bold", color=TEXT_PRIMARY, y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 5,
        height_ratios=[4, 0.6, 4],
        hspace=0.30, wspace=0.12,
        left=0.03, right=0.97, top=0.93, bottom=0.04,
    )

    # ---- ROW 0 — images --------------------------------------------
    # 0-0: original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title("Original Leaf Image", fontsize=13,
                 color=TEXT_PRIMARY, fontweight="bold", pad=10)
    ax.axis("off")

    # 0-1: combined Grad-CAM
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(overlay_heatmap(img_np, combined_cam, alpha=0.45))
    ax.set_title("Combined Attention Map\n(All Findings)", fontsize=12,
                 color=TEXT_PRIMARY, fontweight="bold", pad=10)
    ax.axis("off")

    # 0-2..0-4: top class heatmaps
    for i in range(3):
        ax = fig.add_subplot(gs[0, 2 + i])
        if i < top_n:
            cls_name, cls_prob = ranked[i]
            if cls_name in cam_maps:
                ax.imshow(overlay_heatmap(img_np, cam_maps[cls_name], 0.50))
            else:
                ax.imshow(img_np)
            ax.set_title(f"{cls_name}\n({cls_prob:.1%})",
                         fontsize=13, color=_title_color(cls_prob),
                         fontweight="bold", pad=10)
        else:
            ax.set_facecolor(BG_COLOR)
        ax.axis("off")

    # ---- ROW 1 — severity banner -----------------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(BG_CARD)

    urgent = [c for c, p in ranked if c != "Healthy" and p >= 0.40]
    banner = f"OVERALL DEFICIENCY SEVERITY: {sev_label}"
    if urgent:
        banner += f"  |  !! ATTENTION: {', '.join(urgent)}"
    ax.text(
        0.5, 0.5, banner, fontsize=16, fontweight="bold", color=sev_color,
        ha="center", va="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_CARD,
                  edgecolor=sev_color, linewidth=2.5),
    )
    ax.axis("off")

    # ---- ROW 2 LEFT — bar chart -------------------------------------
    ax_bars = fig.add_subplot(gs[2, :2])
    ax_bars.set_facecolor(BG_CARD)

    classes = [c for c, _ in ranked]
    probs   = [float(p) for _, p in ranked]
    colors  = [_bar_color(p) for p in probs]

    y_pos = np.arange(len(classes))[::-1]
    bars  = ax_bars.barh(y_pos, probs, color=colors,
                          height=0.6, edgecolor="none", alpha=0.90)

    for bar, prob in zip(bars, probs):
        w = bar.get_width()
        lx    = w + 0.02 if w < 0.85 else w - 0.10
        lha   = "left"  if w < 0.85 else "right"
        lcol  = TEXT_PRIMARY if w < 0.85 else BG_COLOR
        ax_bars.text(lx, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1%}", va="center", ha=lha,
                     fontsize=12, fontweight="bold", color=lcol)

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(classes, fontsize=12, fontweight="bold")
    ax_bars.set_xlim(0, 1.15)
    ax_bars.set_title("Deficiency Probability Distribution",
                      fontsize=14, color=TEXT_PRIMARY,
                      fontweight="bold", pad=15)
    ax_bars.axvline(x=0.50, color=ACCENT_RED, ls="--", alpha=0.5, lw=1)
    ax_bars.text(0.51, y_pos[-1] - 0.6, "Threshold (50%)",
                 fontsize=9, color=ACCENT_RED, alpha=0.7)
    for spine in ("top", "right"):
        ax_bars.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax_bars.spines[spine].set_color(BORDER)

    # ---- ROW 2 RIGHT — summary panel --------------------------------
    ax_sum = fig.add_subplot(gs[2, 2:])
    ax_sum.set_facecolor(BG_CARD)
    ax_sum.axis("off")
    for sp in ax_sum.spines.values():
        sp.set_visible(True)
        sp.set_color(BORDER)

    lines = []  # (text, size, weight, colour)
    lines.append(("Deficiency Summary", 18, "bold", ACCENT_BLUE))
    lines.append((f"Severity: {sev_label}", 15, "bold", sev_color))
    lines.append((sev_desc, 11, "normal", TEXT_SECONDARY))

    positive = [(c, p) for c, p in ranked if c != "Healthy" and p >= 0.15]
    lines.append((f"Positive Findings: {len(positive)} / {len(ranked)}"
                   + (f"  |  Urgent: {len(urgent)}" if urgent else ""),
                  12, "normal", TEXT_SECONDARY))
    lines.append(("", 6, "normal", TEXT_PRIMARY))  # spacer
    lines.append(("Active Findings:", 14, "bold", ACCENT_BLUE))

    for cls, prob in ranked[:5]:
        marker = "!!" if prob >= 0.50 else ("*" if prob >= 0.25 else "·")
        col = ACCENT_RED if prob >= 0.50 else (
              ACCENT_ORANGE if prob >= 0.25 else TEXT_SECONDARY)
        lines.append((f"  {marker} {cls} — {prob:.1%}", 12, "normal", col))

        if recommendations_dict and cls in recommendations_dict and prob >= 0.15:
            rec = recommendations_dict[cls]
            if isinstance(rec, dict):
                if "region" in rec:
                    lines.append((f"      Region: {rec['region']}", 10,
                                  "normal", TEXT_SECONDARY))
                if "action" in rec:
                    # wrap long action text
                    wrapped = textwrap.wrap(rec["action"], width=62)
                    for j, wl in enumerate(wrapped):
                        prefix = "      Action: " if j == 0 else "              "
                        lines.append((prefix + wl, 10, "normal", TEXT_SECONDARY))
                if "severity_note" in rec:
                    lines.append((f"      Note: {rec['severity_note']}", 9,
                                  "italic", TEXT_SECONDARY))

    # Render text
    y = 0.96
    for txt, sz, wt, col in lines:
        if y < 0.02:
            break
        # 'wt' might be "italic" which is a fontstyle, not fontweight
        fw = wt if wt in ("bold", "normal") else "normal"
        fs = "italic" if wt == "italic" else "normal"
        ax_sum.text(0.04, y, txt, fontsize=sz, fontweight=fw, color=col,
                    transform=ax_sum.transAxes, va="top",
                    fontfamily="sans-serif", fontstyle=fs)
        y -= 0.038 + (sz - 8) * 0.003

    # ---- save -------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    return output_path
