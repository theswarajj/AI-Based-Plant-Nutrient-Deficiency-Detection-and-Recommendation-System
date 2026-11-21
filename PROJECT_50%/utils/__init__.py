"""
Utility modules for crop nutrient deficiency analysis.

Provides Grad-CAM visualization, Integrated Gradients (SHAP-like)
attribution, dashboard generation, and crop-specific recommendations.
"""

from .gradcam import GradCAM, get_target_layer, overlay_heatmap, create_combined_heatmap
from .gradcam import compute_integrated_gradients, save_attribution_image
from .visualization import create_dashboard
from .recommendations import get_recommendations, get_all_recommendations
from .selection import (
    select_image_interactive,
    list_input_images,
    SPECIES_MAP,
    select_species_interactive,
)
