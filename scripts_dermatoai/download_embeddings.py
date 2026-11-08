"""Descarga explícita de pesos públicos; no envía documentos ni fotografías."""
import argparse
from pathlib import Path
from dermatoai.settings import MODEL_DIR

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destination", type=Path, default=MODEL_DIR)
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = p.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, trust_remote_code=False, device="cpu", tokenizer_kwargs={"fix_mistral_regex": True})
    model.save(str(args.destination))
    print("Embeddings guardados en", args.destination)
if __name__ == "__main__":
    main()
