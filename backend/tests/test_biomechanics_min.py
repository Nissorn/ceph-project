import pytest
from app.services.biomechanics import calculate_clinical_assessment

def test_calculate_clinical_assessment_min_logic():
    # If labial_crest is 0.41, labial_midroot is 1.77, labial_apex is 4.26
    # Then min_labial = 0.41 (which is <= 0.5).
    # And palatal_crest is 1.2, palatal_midroot is 1.5, palatal_apex is 1.0 (so min_palatal = 1.0 > 0.5)
    # The phenotype should be Type 2.
    
    true_min_labial_standard = min(0.41, 1.77, 4.26)
    true_min_palatal_standard = min(1.2, 1.5, 1.0)
    
    res = calculate_clinical_assessment(
        labial_min_mm=true_min_labial_standard,
        palatal_min_mm=true_min_palatal_standard,
        labial_apex_mm=4.26,
        palatal_apex_mm=1.0,
        upper_tip=(0, 0),
        upper_apex=(0, 100),
        ans=(100, 50),
        pns=(200, 50)
    )
    
    assert "Type 2" in res["bone_thickness_type"], f"Expected Type 2, got {res['bone_thickness_type']}"

