import numpy as np
import math

def calculate_clinical_assessment(
    labial_min_mm: float,
    palatal_min_mm: float,
    labial_apex_mm: float,
    palatal_apex_mm: float,
    upper_tip: tuple,
    upper_apex: tuple,
    ans: tuple,
    pns: tuple,
) -> dict:
    """
    Computes clinical assessment based on exact geometry and measurements.
    """
    # ── Step 1: U1-PP Angle Math ──
    # U1 Vector
    u1_vec = np.array(upper_apex) - np.array(upper_tip)
    # PP Vector
    pp_vec = np.array(pns) - np.array(ans)
    
    def angle_between(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    # Compute interior angle
    angle = angle_between(u1_vec, pp_vec)
    # The ceph vectors might need adjusting to get the ~90-130 range.
    # U1 generally points UP (tip to apex). PP generally points LEFT (ANS to PNS).
    # Usually we take the supplementary angle if it's obtuse, or standard cephalometric definitions.
    # The user specifically mentioned interior angle. Let's make sure it's in the standard range.
    if angle > 180:
        angle = 360 - angle
    if angle > 90 and angle > 150: # sometimes they intersect at a sharp angle
        angle = 180 - angle
    
    # Just in case, let's also compute based on the previous u1_pp_angle logic if we need to.
    # We'll use the precise geometric angle computed here, but if the previous logic was different,
    # we should check it. Actually, `angle` is the true geometric angle. Let's just use it, or fallback.
    # But wait, `angle` computed simply as dot product of (apex-tip) and (pns-ans).
    
    u1_pp_angle_deg = angle
    if u1_pp_angle_deg < 90:
        u1_pp_angle_deg = 180 - u1_pp_angle_deg # Typical U1-PP is ~110. If it's < 90, it's the wrong supplementary side.

    if u1_pp_angle_deg < 105.0:
        u1_pp_angle_class = "Retroclined"
        angle_state = "Retroclined"
        general_retraction = "Root torque + retraction (Maximum movement limited by PB distance)"
    elif u1_pp_angle_deg <= 115.0:
        u1_pp_angle_class = "Normal Inclination"
        angle_state = "Normal"
        general_retraction = "Translation movement (Maximum movement limited by PB distance)"
    else:
        u1_pp_angle_class = "Proclined"
        angle_state = "Proclined"
        general_retraction = "Controlled tipping (Maximum movement limited by PB distance)"

    # ── Step 2: Root Apex Position ──
    # Compare LB (labial_apex_mm) and PB (palatal_apex_mm) at apex
    lb = labial_apex_mm
    pb = palatal_apex_mm
    
    if lb < pb - 1.0:
        root_apex_position_type = "Labial"
    elif pb < lb - 1.0:
        root_apex_position_type = "Palatal"
    else:
        root_apex_position_type = "Midway"

    # ── Step 3: Alveolar Bone Phenotype ──
    if labial_min_mm > 0.5 and palatal_min_mm > 0.5:
        bone_thickness_type = "Type 1 - Thick"
        bone_thickness_interpretation = "Thick alveolar bone; Favorable bone support."
    elif (labial_min_mm <= 0.5 and palatal_min_mm > 0.5) or (labial_min_mm > 0.5 and palatal_min_mm <= 0.5):
        bone_thickness_type = "Type 2 - Relatively thick with mono-plate concavity"
        bone_thickness_interpretation = "Represents unilateral cortical thinning."
    else:
        bone_thickness_type = "Type 3/4 - Vulnerably thin"
        bone_thickness_interpretation = "Very thin alveolar bone; High-risk morphology; compromised phenotype requiring extreme caution."

    # ── Step 4: Orthodontic & Biomechanical Plan (Lookup Table) ──
    matrix_table = {
        "Labial": {
            "Retroclined": {
                "Preferred biomechanics": "Light controlled tipping with torque control",
                "Biomechanics to avoid": "Uncontrolled proclination, labial root torque",
                "Clinical implication": "Uprighting is possible but labial cortical bone must be preserved",
            },
            "Normal": {
                "Preferred biomechanics": "Light controlled tipping or torque maintenance",
                "Biomechanics to avoid": "Bodily movement forward, uncontrolled tipping",
                "Clinical implication": "Avoid further labial displacement of the apex",
            },
            "Proclined": {
                "Preferred biomechanics": "Controlled tipping during retraction with strict torque control",
                "Biomechanics to avoid": "Uncontrolled tipping, labial root torque",
                "Clinical implication": "High risk; strict torque control is required",
            },
        },
        "Midway": {
            "Retroclined": {
                "Preferred biomechanics": "Controlled proclination or bodily movement if bone allows",
                "Biomechanics to avoid": "Uncontrolled tipping",
                "Clinical implication": "Favorable prognosis",
            },
            "Normal": {
                "Preferred biomechanics": "Bodily movement (translation)",
                "Biomechanics to avoid": "Uncontrolled tipping",
                "Clinical implication": "Most favorable condition",
            },
            "Proclined": {
                "Preferred biomechanics": "Controlled tipping with torque control during retraction",
                "Biomechanics to avoid": "Uncontrolled tipping",
                "Clinical implication": "Safe if torque is well controlled",
            },
        },
        "Palatal": {
            "Retroclined": {
                "Preferred biomechanics": "Careful movement; labial crown/root control may be required",
                "Biomechanics to avoid": "Palatal root torque, further retroclination",
                "Clinical implication": "Risk of palatal cortical perforation",
            },
            "Normal": {
                "Preferred biomechanics": "Bodily movement with caution",
                "Biomechanics to avoid": "Excessive palatal root torque",
                "Clinical implication": "Monitor palatal bone limits",
            },
            "Proclined": {
                "Preferred biomechanics": "Controlled tipping during retraction with apex control",
                "Biomechanics to avoid": "Retraction causing further palatal displacement of apex",
                "Clinical implication": "Retraction possible but avoid excessive palatal pressure",
            },
        },
    }

    b_matrix = matrix_table.get(root_apex_position_type, {}).get(angle_state, {
        "Preferred biomechanics": "",
        "Biomechanics to avoid": "",
        "Clinical implication": ""
    })

    return {
        "u1_pp_angle_class": u1_pp_angle_class,
        "bone_thickness_type": bone_thickness_type,
        "bone_thickness_interpretation": bone_thickness_interpretation,
        "root_apex_position_type": root_apex_position_type,
        "general_retraction_strategy": general_retraction,
        "preferred_biomechanics": b_matrix["Preferred biomechanics"],
        "biomechanics_to_avoid": b_matrix["Biomechanics to avoid"],
        "clinical_implication": b_matrix["Clinical implication"],
    }
