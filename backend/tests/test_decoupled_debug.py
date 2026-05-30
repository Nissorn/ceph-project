import numpy as np
import pytest
from app.services.analysis_service import calculate_global_minimum

def test_decoupled_anchors_debug():
    tip = np.array([50.0, 100.0])
    apex = np.array([50.0, 0.0])
    u1_unit = np.array([0.0, -1.0])
    u1_perp = np.array([1.0, 0.0])

    labial_crest_pt = np.array([50.0, 80.0])
    palatal_crest_pt = np.array([50.0, 40.0])
    
    tooth_mask = np.zeros((150, 100), dtype=np.uint8)
    tooth_mask[0:150, 45:55] = 255
    
    labial_mask = np.zeros((150, 100), dtype=np.uint8)
    labial_mask[80, 60:] = 255
    labial_mask[0:80, 80:] = 255
    
    palatal_mask = np.zeros((150, 100), dtype=np.uint8)
    palatal_mask[40, 0:35] = 255
    palatal_mask[0:40, 0:10] = 255
    palatal_mask[41:150, 0:10] = 255

    masks = [None]*3
    masks[0] = tooth_mask
    masks[1] = labial_mask
    masks[2] = palatal_mask
    
    global_min_lines = calculate_global_minimum(
        tip=tip,
        apex=apex,
        labial_crest_pt=labial_crest_pt,
        palatal_crest_pt=palatal_crest_pt,
        u1_unit=u1_unit,
        u1_perp=u1_perp,
        masks=masks,
        mm_per_pixel=1.0
    )
    print("RES_0_0 LABIAL_MM:", global_min_lines["0.0"]["labial_mm"])
    print("RES_0_0 LABIAL_LINE:", global_min_lines["0.0"]["labial_line"])
    
if __name__ == "__main__":
    test_decoupled_anchors_debug()
