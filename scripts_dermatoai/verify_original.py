import argparse
import hashlib
import json
import zipfile
from dermatoai.settings import ROOT
def main():
    p=argparse.ArgumentParser();p.add_argument("--zip");args=p.parse_args()
    manifest=json.loads((ROOT/"docs_dermatoai/ORIGINALES_SHA256.json").read_text())
    errors=[]
    for name,digest in manifest.items():
        file=ROOT/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            errors.append(name)
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            for name in z.namelist():
                if name.endswith("/"):continue
                relative="/".join(name.split("/")[1:])
                if relative not in manifest or hashlib.sha256(z.read(name)).hexdigest()!=manifest[relative]:
                    errors.append("ZIP:"+relative)
    if errors:raise SystemExit("Originales diferentes: "+", ".join(errors))
    print(f"OK: {len(manifest)} archivos originales conservados byte a byte.")
if __name__=="__main__":main()
