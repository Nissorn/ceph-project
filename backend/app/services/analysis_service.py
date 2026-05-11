import sys
import os
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.phase3.biomechanics import BoneThicknessCalculator, generate_mock_masks, mock_landmarks
import numpy as np

class AnalysisService:
    def __init__(self):
        pass

    def analyze_image(self, request_data: dict) -> dict:
        """
        Process the image data and calculate biomechanics.
        Uses mocked data for now, waiting for actual integration.
        """
        # Generate mock dependencies locally so Uvicorn does not fail at startup
        landmarks = mock_landmarks()
        tooth_mask, labial_mask, palatal_mask = generate_mock_masks()
        
        upper_tip = landmarks["Upper_tip"]
        upper_apex = landmarks["Upper_apex"]
        u1_axis_vector = (upper_tip, upper_apex)
        
        calc = BoneThicknessCalculator(
            u1_axis_vector=u1_axis_vector,
            upper_apex_point=upper_apex,
            tooth_mask=tooth_mask,
            labial_bone_mask=labial_mask,
            palatal_bone_mask=palatal_mask
        )
        
        plan_c_min = calc.calculate_plan_c_min_with_offset(offset_mm=2.0)
        
        result = {
            "maxillary": {
                "bone_thickness_mm": plan_c_min
            },
            "mandibular": {
                "bone_thickness_mm": 3.1
            },
            "interpretation": "Normal thickness parameters"
        }
        
        return result

# Singleton instance
analysis_service = AnalysisService()
