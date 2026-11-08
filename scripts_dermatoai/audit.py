import argparse
from pathlib import Path
from dermatoai.evaluation import historical_reports
from dermatoai.settings import ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "results_dermatoai/audit")
    args = p.parse_args()
    audit, metrics = historical_reports(ROOT, args.out)
    print(audit)
    print(metrics.to_string(index=False))
if __name__ == "__main__":
    main()
