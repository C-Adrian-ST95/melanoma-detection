"""Evaluación propia sin entrenamiento; manifiesto CSV: filename,diagnostic."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from dermatoai.classifier import Classifier, image_from_bytes
from dermatoai.evaluation import evaluate_frame
from dermatoai.settings import ROOT, CLASSES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results_dermatoai/new_test")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    labels = pd.read_csv(args.manifest)
    if not {"filename", "diagnostic"}.issubset(labels.columns) or labels.empty or labels.filename.duplicated().any():
        raise ValueError("Manifiesto vacío, inválido o con nombres repetidos.")
    if labels[["filename", "diagnostic"]].isna().any().any() or not set(labels.diagnostic) <= set(CLASSES):
        raise ValueError("Clase inválida o dato vacío.")
    base = args.images.resolve(); paths = [(base / f).resolve() for f in labels.filename]
    if any(not path.is_relative_to(base) or not path.is_file() for path in paths):
        raise ValueError("Imagen ausente o fuera de --images.")
    args.out.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoints:
        model = Classifier(checkpoint, args.device)
        rows = []; start = time.perf_counter()
        for begin in range(0, len(paths), 8):
            images = [image_from_bytes(path.read_bytes()) for path in paths[begin:begin+8]]
            for i, result in enumerate(model.predict_many(images)):
                item = labels.iloc[begin+i]
                rows.append(dict(img_id=item.filename, y_true=CLASSES.index(item.diagnostic),
                                 **{f"p_{c}":result["scores"][c] for c in CLASSES}, forward_ms=result["forward_ms"]))
        frame = pd.DataFrame(rows)
        metrics = evaluate_frame(frame)
        metrics.update(checkpoint=model.info, wall_seconds=time.perf_counter()-start,
                       note="Evaluación propia sobre el manifiesto; verificar independencia respecto al entrenamiento.")
        frame.to_csv(args.out / f"{checkpoint.stem}_predictions.csv", index=False)
        (args.out / f"{checkpoint.stem}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        del model
        print(checkpoint.stem, metrics["accuracy"], metrics["macro_f1_six_classes"])
if __name__ == "__main__":
    main()
