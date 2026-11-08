"""Diagnóstico de instalación sin mostrar secretos ni iniciar conexiones."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from dermatoai.settings import ROOT

def main():
    modules=['streamlit','pypdf','filelock','PIL','numpy','pandas','sklearn','requests','jwt','cryptography','torch','torchvision','timm']
    manifest=json.loads((ROOT/'docs_dermatoai/ORIGINALES_SHA256.json').read_text())
    changed=[name for name,digest in manifest.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    report={'python':sys.version,'executable':sys.executable,'system':platform.platform(),
       'modules':{name:importlib.util.find_spec(name) is not None for name in modules},
       'original_files_checked':len(manifest),'original_files_missing_or_modified':changed,
       'checkpoints':[p.name for p in (ROOT/'results/checkpoints').glob('*.pt')],
       'documents_directory':str(ROOT/'knowledge_dermatoai/documents'),
       'database_exists':(ROOT/'knowledge_dermatoai/rag.sqlite3').exists()}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if changed:raise SystemExit(1)
if __name__=='__main__':main()
