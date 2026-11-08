"""Métricas independientes; nunca sobrescribe results/tables del repositorio."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score, f1_score
from .settings import CLASSES

def evaluate_frame(frame):
    required = {"y_true", *(f"p_{c}" for c in CLASSES)}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Se requieren y_true y las seis columnas p_CLASE.")
    raw = pd.to_numeric(frame.y_true, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Etiquetas inválidas: use enteros 0 a 5.")
    y = raw.astype(int)
    p = frame[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
    if not np.isin(y, range(6)).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError("Etiquetas o puntuaciones inválidas.")
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(6))
    pr, rec, f1, support = precision_recall_fscore_support(y, pred, labels=range(6), zero_division=0)
    rows = []
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp); fp = int(cm[:, i].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
        binary = y == i
        rows.append(dict(class_name=cls, support=int(support[i]), precision=float(pr[i]) if support[i] else None,
                         recall=float(rec[i]) if support[i] else None, f1=float(f1[i]) if support[i] else None,
                         sensitivity=tp/(tp+fn) if tp+fn else None, specificity=tn/(tn+fp) if tn+fp else None,
                         auc_ovr=float(roc_auc_score(binary, p[:, i])) if binary.any() and (~binary).any() else None,
                         tp=tp, fp=fp, tn=tn, fn=fn))
    target = np.isin(y, [1, 2, 4]); aggregate = p[:, [1, 2, 4]].sum(1)
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(np.mean(rec[support > 0])),
                macro_f1_present_classes=float(f1_score(y, pred, labels=np.flatnonzero(support), average="macro", zero_division=0)),
                macro_f1_six_classes=float(np.mean(f1)),
                auc_cancer=float(roc_auc_score(target, aggregate)) if target.any() and (~target).any() else None,
                per_class=rows, confusion=cm.tolist())

def audit_split(path):
    d = pd.read_csv(path)
    required = {"img_id", "patient_id", "lesion_id", "diagnostic", "split"}
    if not required.issubset(d.columns) or d[list(required)].isna().any().any() or set(d.split) != {"train", "val", "test"}:
        raise ValueError("Partición incompleta o inválida.")
    return dict(images=len(d), patients=int(d.patient_id.nunique()), duplicate_image_ids=int(d.img_id.duplicated().sum()),
                patients_in_multiple_splits=int(d.groupby("patient_id").split.nunique().gt(1).sum()),
                patient_lesion_pairs_in_multiple_splits=int(d.groupby(["patient_id", "lesion_id"]).split.nunique().gt(1).sum()),
                ambiguous_raw_lesion_ids_in_multiple_patients=int(d.groupby("lesion_id").patient_id.nunique().gt(1).sum()),
                sizes=d.groupby("split").size().to_dict(),
                caveat="Sin imágenes no es posible comprobar duplicados visuales ni validar el origen de las etiquetas.")

def historical_reports(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(root / "data/splits/pad_split.csv")
    expected = split.set_index("img_id")
    reports, summary, mismatch = {}, [], []
    for path in sorted((root / "results/preds").glob("*.csv")):
        df = pd.read_csv(path)
        clean = df[(df.split == "test") & (df.corruption == "clean")].copy()
        if clean.img_id.duplicated().any():
            raise ValueError(f"IDs duplicados en test limpio: {path.name}")
        mapped = expected.reindex(df.img_id)
        correct_split = np.array_equal(mapped.split.to_numpy(), df.split.to_numpy())
        labels = mapped.diagnostic.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
        correct_labels = np.array_equal(labels, df.y_true.to_numpy())
        complete_test = set(clean.img_id) == set(split.loc[split.split == "test", "img_id"])
        if not (correct_split and correct_labels and complete_test):
            mismatch.append(path.stem)
        metrics = evaluate_frame(clean)
        reports[path.stem] = {"origin": "Predicciones CSV del repositorio; no inferencia nueva ni entrenamiento reejecutado.",
                              "split_and_label_match": correct_split and correct_labels and complete_test, **metrics}
        summary.append(dict(run_id=path.stem, **{k:metrics[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1_six_classes", "auc_cancer")}))
    (destination / "historical_metrics.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    pd.DataFrame(summary).to_csv(destination / "historical_comparison.csv", index=False)
    audit = audit_split(root / "data/splits/pad_split.csv")
    audit["prediction_runs_with_mismatches"] = mismatch
    (destination / "split_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit, pd.DataFrame(summary)
