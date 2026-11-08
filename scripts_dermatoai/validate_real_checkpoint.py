"""Contrasta pesos e imágenes REALES contra las predicciones archivadas, sin entrenar."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dermatoai.classifier import Classifier,image_from_bytes
from dermatoai.settings import ROOT,CLASSES
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",default="vits16_robust_s1")
    p.add_argument("--images",required=True,type=Path)
    p.add_argument("--per-class",type=int,default=2)
    p.add_argument("--atol",type=float,default=1e-4)
    args=p.parse_args()
    if args.per_class<1:raise ValueError("--per-class debe ser positivo")
    old=pd.read_csv(ROOT/f"results/preds/{args.run}.csv")
    clean=old[(old.split=="test")&(old.corruption=="clean")].groupby("y_true",group_keys=False).head(args.per_class)
    files={f.name:f for f in args.images.rglob("*.png")}
    missing=[i for i in clean.img_id if i not in files]
    if missing:raise FileNotFoundError("Faltan imágenes: "+", ".join(missing))
    model=Classifier(ROOT/f"results/checkpoints/{args.run}.pt")
    pred=model.predict_many([image_from_bytes(files[i].read_bytes()) for i in clean.img_id])
    pnew=np.array([[r["scores"][c] for c in CLASSES] for r in pred])
    pold=clean[[f"p_{c}" for c in CLASSES]].to_numpy()
    error=np.abs(pnew-pold)
    report=dict(run_id=args.run,sha256=model.info["sha256"],n=len(clean),max_abs_error=float(error.max()),
                mean_abs_error=float(error.mean()),tolerance=args.atol,match=bool(np.allclose(pnew,pold,atol=args.atol,rtol=0)))
    out=ROOT/"results_dermatoai/real_checkpoint_validation.json"
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report["match"]:raise SystemExit("No coincide: revisar pesos, versión y datos antes de usar.")
if __name__=="__main__":main()
