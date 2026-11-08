from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
NAMES = {"ACK": "Queratosis actínica", "BCC": "Carcinoma basocelular", "MEL": "Melanoma",
         "NEV": "Nevo", "SCC": "Carcinoma espinocelular", "SEK": "Queratosis seborreica"}
MODES = {"PDF + organizaciones": ("pdf", "organization"), "Solo PDF": ("pdf",),
         "Solo organizaciones": ("organization",), "Sin RAG": ()}
ALLOWED_DOMAINS = ("aad.org", "cancer.gov", "cancer.org", "who.int", "iarc.who.int")
MODEL_DIR = ROOT / "knowledge_local" / "embedding_model"
INDEX_FILE = ROOT / "knowledge_local" / "index.sqlite3"
NOTICE = ("Uso académico y de investigación. Las puntuaciones softmax no son probabilidades clínicas "
          "calibradas. Este sistema no confirma ni descarta enfermedades. Se requiere evaluación dermatológica.")
SCOPE = ("Solo distingue ACK, BCC, MEL, NEV, SCC y SEK. No detecta automáticamente imágenes ajenas a "
         "estas categorías y no dispone de una clase «ninguna». No presupone rasgos clínicos observados.")
