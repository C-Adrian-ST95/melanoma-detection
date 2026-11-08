import argparse
import json
from pathlib import Path
from dermatoai.settings import ROOT
from dermatoai.knowledge import KnowledgeBase
from dermatoai.rag_metrics import evaluate_cases

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results_dermatoai/rag_evaluation_v2.json')
    p.add_argument('--method',choices=['none','lexical','semantic','hybrid'],default='lexical')
    p.add_argument('--k',type=int,default=6)
    p.add_argument('--include-review',action='store_true')
    p.add_argument('--categories',nargs='*',choices=['ACK','BCC','MEL','NEV','SCC','SEK'])
    p.add_argument('--languages',nargs='*',choices=['es','en','und'])
    args=p.parse_args()
    kb=KnowledgeBase(ROOT/'knowledge_dermatoai/rag.sqlite3',ROOT/'knowledge_dermatoai/documents')
    encoder=None
    if args.method in {'semantic','hybrid'}:
        from dermatoai.rag import LocalEncoder
        encoder=LocalEncoder(ROOT/'knowledge_local/embedding_model')
    cases=[json.loads(line) for line in args.cases.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    report=evaluate_cases(kb,cases,args.k,args.method,encoder,include_review=args.include_review,categories=args.categories,languages=args.languages)
    report['ablation_without_retrieval']=evaluate_cases(kb,cases,args.k,'none')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output)
if __name__=='__main__':main()
