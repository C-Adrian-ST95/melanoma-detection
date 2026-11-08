"""Administración explícita del corpus; no toca src ni resultados originales."""
import argparse
import json
from dermatoai.settings import ROOT
from dermatoai.knowledge import KnowledgeBase

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['sync','list','vectors'])
    args=p.parse_args()
    kb=KnowledgeBase(ROOT/'knowledge_dermatoai/rag.sqlite3',ROOT/'knowledge_dermatoai/documents')
    if args.action=='sync':
        report=kb.sync();print(json.dumps(report,ensure_ascii=False,indent=2))
        if report['errors']:raise SystemExit(1)
    elif args.action=='list':print(json.dumps(kb.list_sources(),ensure_ascii=False,indent=2))
    else:
        from dermatoai.rag import LocalEncoder
        print('Embeddings guardados:',kb.build_vectors(LocalEncoder(ROOT/'knowledge_local/embedding_model')))
if __name__=='__main__':main()
