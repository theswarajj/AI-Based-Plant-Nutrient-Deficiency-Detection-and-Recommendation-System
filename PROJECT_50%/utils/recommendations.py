"""
Crop-specific nutrient deficiency recommendations.

Each crop maps every predicted class to a dict with keys:
    region          – which part of the leaf / plant is affected
    description     – visual symptoms and physiological explanation
    action          – corrective fertiliser / treatment recommendation
    severity_note   – extra context (mobility, urgency, etc.)
"""

RECOMMENDATIONS = {
    # ================================================================
    #  ASH GOURD
    # ================================================================
    "ashgourd": {
        "Healthy": {
            "region": "Entire leaf",
            "description": "No visible nutrient deficiency symptoms.",
            "action": "Continue current fertilisation schedule. Monitor regularly.",
            "severity_note": "No corrective action required.",
        },
        "Nitrogen": {
            "region": "Older / lower leaves first",
            "description": (
                "General yellowing (chlorosis) starting from older leaves. "
                "Stunted vine growth and reduced leaf size."
            ),
            "action": (
                "Apply urea (46-0-0) at 5-10 g/plant or foliar spray of 2 % "
                "urea solution for rapid relief. Split application recommended."
            ),
            "severity_note": (
                "Nitrogen is mobile — symptoms always appear on oldest leaves first."
            ),
        },
        "Potassium": {
            "region": "Leaf margins and tips",
            "description": (
                "Brown scorching and curling of leaf tips; interveinal chlorosis "
                "on older leaves."
            ),
            "action": (
                "Apply potassium sulfate (K₂SO₄) or muriate of potash (KCl). "
                "Soil application at 10-15 g/plant."
            ),
            "severity_note": (
                "Potassium is critical for fruit development and disease resistance."
            ),
        },
        "Nitrogen_Potassium": {
            "region": "Widespread — older leaves and margins",
            "description": (
                "Combined N + K deficiency: yellowing plus marginal scorch. "
                "Severely impacts yield."
            ),
            "action": (
                "Apply balanced NPK fertiliser (19-19-19). Supplement with "
                "urea and potash in split doses."
            ),
            "severity_note": "Combined deficiency requires urgent balanced fertilisation.",
        },
        "Potassium_Magnesium": {
            "region": "Interveinal areas and leaf margins",
            "description": (
                "K + Mg deficiency: interveinal chlorosis with marginal necrosis."
            ),
            "action": (
                "Apply K-Mag (langbeinite) or MOP + 1 % MgSO₄ foliar spray."
            ),
            "severity_note": (
                "Mg is central to chlorophyll — expect photosynthesis reduction."
            ),
        },
        "Powdery_Mildew": {
            "region": "Upper leaf surface",
            "description": (
                "Fungal disease: white powdery patches on leaves. NOT a nutrient "
                "deficiency."
            ),
            "action": (
                "Apply sulfur-based fungicide or potassium bicarbonate spray. "
                "Improve air circulation; remove badly affected leaves."
            ),
            "severity_note": "Disease — requires fungicide, not fertiliser.",
        },
        "Nitrogen_Magnesium": {
            "region": "Older leaves — interveinal areas",
            "description": (
                "Combined N + Mg deficiency: yellowing with distinct interveinal "
                "chlorosis pattern."
            ),
            "action": (
                "Apply ammonium nitrate + Epsom salt (MgSO₄) foliar spray at 2 %."
            ),
            "severity_note": "Both nutrients affect chlorophyll production.",
        },
    },

    # ================================================================
    #  BANANA
    # ================================================================
    "banana": {
        "Healthy": {
            "region": "Entire leaf",
            "description": "No visible nutrient deficiency.",
            "action": "Continue current fertilisation schedule. Monitor regularly.",
            "severity_note": "No corrective action required.",
        },
        "Boron": {
            "region": "Young leaves and growing points",
            "description": (
                "Deformed / curled new leaves, white streaks, thickened lamina."
            ),
            "action": (
                "Apply borax (Na₂B₄O₇) at 10-20 g/plant, or foliar spray "
                "of 0.2 % boric acid."
            ),
            "severity_note": "Boron is immobile — symptoms appear on newest growth.",
        },
        "Calcium": {
            "region": "Young leaves and leaf margins",
            "description": (
                "Distorted new growth, marginal necrosis, weak cell walls."
            ),
            "action": (
                "Apply gypsum (CaSO₄) or calcium nitrate. Foliar spray of "
                "0.5 % CaCl₂ for fast correction."
            ),
            "severity_note": "Calcium is immobile — check soil pH (optimal 5.5-7.0).",
        },
        "Iron": {
            "region": "Young / new leaves",
            "description": (
                "Interveinal chlorosis on young leaves (veins stay green, "
                "lamina turns yellow)."
            ),
            "action": (
                "Foliar spray of 0.5 % ferrous sulfate (FeSO₄). Check soil "
                "pH — iron lock-out occurs above pH 7."
            ),
            "severity_note": "Iron is immobile — symptoms always on youngest leaves.",
        },
        "Magnesium": {
            "region": "Older leaves — interveinal areas",
            "description": (
                "Interveinal chlorosis on older leaves; green veins, yellow lamina."
            ),
            "action": (
                "Apply Epsom salt (MgSO₄) foliar spray at 2 % or soil drench "
                "(50-100 g/plant)."
            ),
            "severity_note": "Mg is mobile — older leaves sacrifice Mg for new growth.",
        },
        "Potassium": {
            "region": "Older leaf margins",
            "description": (
                "Rapid yellowing of older leaves, marginal scorch, premature "
                "senescence. Orange / yellow leaf margins."
            ),
            "action": (
                "Apply MOP (KCl) at 200-300 g/plant/year in splits. Banana is "
                "a very heavy K feeder."
            ),
            "severity_note": "K is critical for bunch development in banana.",
        },
        "Sulphur": {
            "region": "Young leaves (uniform yellowing)",
            "description": (
                "Uniform pale yellow on young leaves (unlike N which affects old "
                "leaves)."
            ),
            "action": (
                "Apply ammonium sulfate or gypsum. Foliar spray of 0.5 % "
                "potassium sulfate."
            ),
            "severity_note": "Sulphur is semi-mobile — symptoms on newer leaves.",
        },
        "Zinc": {
            "region": "Young leaves — interveinal areas",
            "description": (
                "Small, narrow leaves with interveinal chlorosis ('little leaf')."
            ),
            "action": (
                "Foliar spray of 0.5 % ZnSO₄. Soil application 25 g ZnSO₄/plant."
            ),
            "severity_note": "Zinc is immobile — symptoms appear on youngest growth.",
        },
    },

    # ================================================================
    #  BITTER GOURD
    # ================================================================
    "bitter_gourd": {
        "Healthy": {
            "region": "Entire leaf",
            "description": "No visible nutrient deficiency.",
            "action": "Continue current fertilisation schedule. Monitor regularly.",
            "severity_note": "No corrective action required.",
        },
        "Nitrogen": {
            "region": "Older / lower leaves",
            "description": (
                "General yellowing of older leaves, stunted vine growth, "
                "reduced fruiting."
            ),
            "action": (
                "Apply urea (46-0-0) at 5-10 g/plant or 2 % urea foliar spray. "
                "Split application recommended."
            ),
            "severity_note": "Nitrogen is critical for vine growth in bitter gourd.",
        },
        "Nitrogen_Potassium": {
            "region": "Widespread — older leaves and margins",
            "description": (
                "Combined N + K deficiency: vine vigour and fruit development "
                "are severely impacted."
            ),
            "action": (
                "Apply balanced NPK (19-19-19). Supplement with urea and MOP."
            ),
            "severity_note": "Combined deficiency — urgent balanced fertilisation.",
        },
        "Potassium": {
            "region": "Leaf margins and tips",
            "description": "Marginal scorch on older leaves, poor fruit quality.",
            "action": (
                "Apply K₂SO₄ or KCl at 10-15 g/plant."
            ),
            "severity_note": "K essential for fruit quality and disease resistance.",
        },
        "Magnesium": {
            "region": "Older leaves — interveinal areas",
            "description": "Interveinal chlorosis, reduced photosynthesis.",
            "action": "Foliar spray of 2 % MgSO₄ (Epsom salt). Repeat every 2 weeks.",
            "severity_note": "Mg is mobile — symptoms on oldest leaves first.",
        },
        "Phosphorus": {
            "region": "Older leaves, roots",
            "description": (
                "Dark green to purplish leaves, stunted growth, poor root "
                "development."
            ),
            "action": (
                "Apply single superphosphate (SSP) or DAP at 40-60 kg P₂O₅/ha "
                "as basal."
            ),
            "severity_note": "P is critical for root and early growth. Apply early.",
        },
    },

    # ================================================================
    #  RICE
    # ================================================================
    "rice": {
        "Nitrogen": {
            "region": "Older / lower leaves, whole plant",
            "description": (
                "Pale green to yellow leaves starting from older leaves, "
                "reduced tillering, stunted growth."
            ),
            "action": (
                "Apply urea at 30-50 kg/ha in standing water. Split into "
                "3 doses: basal, tillering, panicle initiation."
            ),
            "severity_note": (
                "Most common rice deficiency. Critical for tillering & grain filling."
            ),
        },
        "Phosphorus": {
            "region": "Older leaves, root zone",
            "description": (
                "Dark green to purplish leaves, reduced root development, "
                "delayed maturity."
            ),
            "action": (
                "Apply SSP or DAP at 40-60 kg P₂O₅/ha as basal dose before "
                "transplanting."
            ),
            "severity_note": "P is critical for root development. Apply early.",
        },
        "Potassium": {
            "region": "Leaf tips and margins of older leaves",
            "description": (
                "Brown / scorched leaf tips, lodging susceptibility, increased "
                "disease incidence."
            ),
            "action": (
                "Apply MOP (KCl) at 40-60 kg K₂O/ha. Split: 50 % basal + "
                "50 % at panicle initiation."
            ),
            "severity_note": (
                "K improves disease resistance and grain quality. Critical later."
            ),
        },
    },
}


def get_recommendations(crop_key, class_name):
    """Return the recommendation dict for one (crop, class) pair."""
    crop = RECOMMENDATIONS.get(crop_key, {})
    if class_name in crop:
        return crop[class_name]
    return {
        "region": "See leaf analysis",
        "description": (
            f"{class_name} condition detected. Consult an agronomist."
        ),
        "action": "Perform soil testing and consult local extension services.",
        "severity_note": "Refer to crop-specific nutrient management guidelines.",
    }


def get_all_recommendations(crop_key):
    """Return the full recommendation dict for a crop."""
    return RECOMMENDATIONS.get(crop_key, {})
