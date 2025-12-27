# -*- coding: utf-8 -*-
"""
Unified Feature Definition Management File
Function: Centralize storage of various feature lists, feature type mappings, and total feature lists for cross-script reuse
"""

# ==========================
# Various Feature List Definitions (retain original format, including trailing spaces)
# ==========================
# Vegetation Indices Features
Vegetation_Index = [
    'RVI', 'CIgreen', 'Cired', 'MDD', 'Int1', 'Int2 ',
    'Red-Edge NDVl ', 'GARl ', 'SlPI', 'ARVl', 'EVI',
    'GNDVl', 'NDVl', 'SAVI', 'VARI_1', 'NDI',
]

coverage = ['coverage']

# Color Indices Features
Color_Index = ['ExG ', 'ExR ', 'ExGR', 'VARI_2', 'GLI ', 'WI ']

# Texture Indices Features
Texture_Feature = [
    'R_Mean', 'R_Variance', 'R_Homogeneity', 'R_Contrast',
    'R_Dissimilarity', 'R_Entropy', 'R_Second Moment', 'R_Correlation',
    'G_Mean', 'G_Variance', 'G_Homogeneity', 'G_Contrast',
    'G_Dissimilarity', 'G_Entropy', 'G_Second Moment', 'G_Correlation',
    'B_Mean', 'B_Variance', 'B_Homogeneity', 'B_Contrast',
    'B_Dissimilarity', 'B_Entropy', 'B_Second Moment', 'B_Correlation'
]

# Climate Features (keep Chinese content in the list as required)
Meteorological_Factor = [
    "2m Average Temperature (C)",
    "2m Maximum Temperature (C)",
    "2m Minimum Temperature (C)",
    "Precipitation (mm)",
    "Growing Degree Days (GDD)",
    "Relative Humidity (%)",
    "Total Solar Radiation (downward, J/m²)",
    "Peak Sunshine Duration (h)"
]

# ==========================
# Feature Type Mapping Dictionary (feature name → feature category)
# ==========================
feature_types = {
    **{col: 'Vegetation Index' for col in Vegetation_Index},
    **{col: 'Coverage Index' for col in coverage},
    **{col: 'Color Index' for col in Color_Index},
    **{col: 'Texture Feature' for col in Texture_Feature},
    **{col: 'Meteorological Factor' for col in Meteorological_Factor}
}

# ==========================
# Total Feature List (merged in category order)
# ==========================
all_features = Vegetation_Index + coverage + Color_Index + Texture_Feature + Meteorological_Factor

# ==========================
# Optional: Validation and Information Printing (executed when running this file directly)
# ==========================
