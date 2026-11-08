import numpy as np
import pandas as pd
import pytest
from dermatoai.settings import CLASSES, ROOT
from dermatoai.evaluation import evaluate_frame, audit_split

def frame(y,p):
    return pd.DataFrame({"y_true":y,**{f"p_{c}":p[:,i] for i,c in enumerate(CLASSES)}})

def test_perfect_metrics():
    r=evaluate_frame(frame(np.arange(6),np.eye(6)))
    assert r["accuracy"]==1 and r["auc_cancer"]==1
    assert all(c["sensitivity"]==1 and c["specificity"]==1 and c["auc_ovr"]==1 for c in r["per_class"])

def test_missing_classes_not_falsely_zero_sensitivity():
    r=evaluate_frame(frame([0,0],np.eye(6)[[0,0]]))
    assert r["per_class"][2]["sensitivity"] is None
    assert r["per_class"][2]["auc_ovr"] is None
    assert r["auc_cancer"] is None
    assert r["macro_f1_present_classes"]==1
    assert r["macro_f1_six_classes"]==pytest.approx(1/6)

def test_bad_scores_and_fractional_label():
    with pytest.raises(ValueError):
        evaluate_frame(frame([0],np.array([[1,1,0,0,0,0]])))
    with pytest.raises(ValueError):
        evaluate_frame(frame([0.1],np.eye(6)[[0]]))

def test_original_split_audit():
    r=audit_split(ROOT/"data/splits/pad_split.csv")
    assert r["images"]==2298 and r["patients"]==1373
    assert r["duplicate_image_ids"]==0 and r["patients_in_multiple_splits"]==0
    assert r["patient_lesion_pairs_in_multiple_splits"]==0

def test_historical_default_matches():
    d=pd.read_csv(ROOT/"results/preds/vits16_robust_s1.csv")
    r=evaluate_frame(d[(d.split=="test")&(d.corruption=="clean")])
    assert r["n"]==329
    assert r["accuracy"]==pytest.approx(0.7386018237082067)
    assert r["auc_cancer"]==pytest.approx(0.9288570712190501,abs=1e-6)
