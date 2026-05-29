import json
import logging
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)

def build_splits(records, n_folds=5, holdout_ratio=0.15, random_state=42):
    """
    Build patient-aware, stratified cross-validation splits and holdout set.
    Ensures T1+T2 of the same patient always stay together.
    """
    # 1. Group records by patient_id
    patient_to_treatment = {}
    for r in records:
        pid = r.get("patient_id")
        if not pid:
            continue
        treatments = r.get("treatment", [])
        if pid not in patient_to_treatment:
            patient_to_treatment[pid] = set()
        patient_to_treatment[pid].update(treatments)
        
    patient_ids = sorted(list(patient_to_treatment.keys()))
    if not patient_ids:
        log.warning("No patients found in records to create splits.")
        return {"holdout": [], "folds": []}

    # 2. Extract multi-label treatment vector per patient
    treatment_lists = [list(patient_to_treatment[pid]) for pid in patient_ids]
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(treatment_lists)
    X = np.zeros((len(patient_ids), 1)) # dummy features
    
    has_labels = y.shape[1] > 0
    
    holdout_ids = []
    cv_patient_ids = []
    cv_y = []
    
    # 3. Isolate holdout patients
    if holdout_ratio > 0:
        if has_labels:
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=random_state)
            try:
                # iterstrat expects X and y
                for cv_idx, holdout_idx in msss.split(X, y):
                    holdout_ids = [patient_ids[i] for i in holdout_idx]
                    cv_patient_ids = [patient_ids[i] for i in cv_idx]
                    cv_y = y[cv_idx]
            except Exception as e:
                log.warning(f"MultilabelStratifiedShuffleSplit failed: {e}. Falling back to non-stratified holdout.")
                np.random.seed(random_state)
                indices = np.random.permutation(len(patient_ids))
                holdout_size = max(1, int(len(patient_ids) * holdout_ratio))
                holdout_idx = indices[:holdout_size]
                cv_idx = indices[holdout_size:]
                holdout_ids = [patient_ids[i] for i in holdout_idx]
                cv_patient_ids = [patient_ids[i] for i in cv_idx]
                cv_y = y[cv_idx]
        else:
            np.random.seed(random_state)
            indices = np.random.permutation(len(patient_ids))
            holdout_size = max(1, int(len(patient_ids) * holdout_ratio))
            holdout_idx = indices[:holdout_size]
            cv_idx = indices[holdout_size:]
            holdout_ids = [patient_ids[i] for i in holdout_idx]
            cv_patient_ids = [patient_ids[i] for i in cv_idx]
            cv_y = np.zeros((len(cv_idx), 0))
    else:
        cv_patient_ids = patient_ids
        cv_y = y
        
    # 4. StratifiedKFold on remaining patients
    folds = []
    if n_folds > 1 and len(cv_patient_ids) >= n_folds:
        X_cv = np.zeros((len(cv_patient_ids), 1))
        if has_labels:
            mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            try:
                for train_idx, val_idx in mskf.split(X_cv, cv_y):
                    folds.append({
                        "train": [cv_patient_ids[i] for i in train_idx],
                        "val": [cv_patient_ids[i] for i in val_idx]
                    })
            except Exception as e:
                log.warning(f"MultilabelStratifiedKFold failed: {e}. Falling back to KFold.")
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                for train_idx, val_idx in kf.split(X_cv):
                    folds.append({
                        "train": [cv_patient_ids[i] for i in train_idx],
                        "val": [cv_patient_ids[i] for i in val_idx]
                    })
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            for train_idx, val_idx in kf.split(X_cv):
                folds.append({
                    "train": [cv_patient_ids[i] for i in train_idx],
                    "val": [cv_patient_ids[i] for i in val_idx]
                })
    elif n_folds == 1 or len(cv_patient_ids) < n_folds:
        # Not enough patients for n_folds
        folds.append({"train": cv_patient_ids, "val": []})
        
    return {
        "holdout": holdout_ids,
        "folds": folds
    }

def get_lopo_splits(records: list[dict]) -> list[tuple[list[str], list[str]]]:
    """
    Build Leave-One-Patient-Out splits from image records.

    Returns list of (train_image_ids, test_image_ids) — one tuple per patient.
    T1 and T2 of the same patient are always held out together.
    """
    patient_ids = sorted({r["patient_id"] for r in records})
    pid_to_iids: dict[str, list[str]] = {}
    for r in records:
        pid_to_iids.setdefault(r["patient_id"], []).append(r["image_id"])

    splits = []
    for test_pid in patient_ids:
        test_ids = pid_to_iids[test_pid]
        train_ids = [
            iid
            for pid, imgs in pid_to_iids.items()
            if pid != test_pid
            for iid in imgs
        ]
        splits.append((train_ids, test_ids))
    return splits


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    records_path = project_root / "data" / "processed" / "landmarks_clean.json"
    splits_path = project_root / "data" / "processed" / "splits.json"
    
    if not records_path.exists():
        log.error("Records file not found: %s", records_path)
        return
        
    with open(records_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        records = data["images"]
        
    splits = build_splits(records)
    
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
        
    log.info("Saved splits to %s", splits_path)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
