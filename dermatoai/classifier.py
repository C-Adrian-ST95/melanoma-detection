"""Carga estricta del contrato original; ninguna llamada entrena ni descarga pesos."""
import hashlib
import io
import threading
import time
from pathlib import Path
from .settings import ROOT, CLASSES

def image_from_bytes(data):
    from PIL import Image
    if len(data) > 15 * 1024 * 1024:
        raise ValueError("Imagen demasiado grande: máximo 15 MiB.")
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Use PNG, JPEG o WEBP.")
        if source.width * source.height > 25_000_000:
            raise ValueError("Máximo 25 millones de píxeles.")
        # No EXIF transpose: se respeta la lectura original PIL -> RGB.
        return source.convert("RGB")

def prepare_image(image):
    from PIL import Image
    image = image.convert("RGB")
    scale = 256 / min(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image

def checkpoint_files(directory=None):
    folder = Path(directory or ROOT / "results/checkpoints")
    return sorted(p for p in folder.glob("*.pt") if not p.stem.endswith("_last"))

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

class Classifier:
    def __init__(self, checkpoint, device="cpu"):
        import torch
        from src.train import Net
        from src.data import CLASSES as original_classes
        from src.corruptions import build_transform
        if tuple(original_classes) != CLASSES:
            raise ValueError("El orden de clases original cambió; revisar src/data.py.")
        self.path = Path(checkpoint).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Falta el checkpoint: {self.path}")
        ck = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or not {"state_dict", "config", "run_id", "epoch"}.issubset(ck):
            raise ValueError("Se espera checkpoint BEST con state_dict, config, run_id y epoch; no _last ni state_dict suelto.")
        cfg = ck["config"]
        if cfg.get("model") not in {"vit_small_patch16_224", "efficientnet_b0"} or cfg.get("img_size") != 224:
            raise ValueError("Solo se admiten las arquitecturas originales a 224 px.")
        if ck["run_id"] != self.path.stem:
            raise ValueError("El run_id del checkpoint no coincide con el nombre del archivo.")
        state = ck["state_dict"]
        if not {"mean", "std"}.issubset(state):
            raise ValueError("Faltan buffers de normalización; no se puede garantizar compatibilidad.")
        if not all(torch.isfinite(state[k]).all() for k in state):
            raise ValueError("Pesos con valores no finitos.")
        if torch.any(state["std"] <= 0):
            raise ValueError("Desviación de normalización inválida.")
        self.device = device
        self.model = Net(cfg["model"], pretrained=False)
        self.model.load_state_dict(state, strict=True)  # incluye mean y std originales
        self.model = self.model.to(device).eval()
        self.transform = build_transform("none", False, 224)
        self.info = dict(run_id=ck["run_id"], architecture=cfg["model"], epoch=int(ck["epoch"]),
                         checkpoint=self.path.name, sha256=file_sha256(self.path), device=device,
                         img_size=224, classes=list(CLASSES), mean=state["mean"].flatten().tolist(),
                         std=state["std"].flatten().tolist())
        self.lock = threading.RLock()

    def tensor(self, images):
        import torch
        return torch.stack([self.transform(prepare_image(im)) for im in images]).to(self.device)

    def predict_many(self, images, batch_size=8):
        import torch
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        out = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = self.tensor(images[start:start + batch_size])
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                scores = torch.softmax(self.model(batch).float(), 1)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - t0) / len(batch)
                for row in scores.cpu().tolist():
                    if len(row) != 6 or not all(0 <= s <= 1 for s in row):
                        raise ValueError("Salidas inválidas del modelo.")
                    top = max(range(6), key=row.__getitem__)
                    out.append(dict(top_class=CLASSES[top], top_score=row[top],
                                    scores=dict(zip(CLASSES, row)), forward_ms=elapsed, **self.info))
        return out

    def predict(self, image):
        return self.predict_many([image])[0]

    def explain(self, image):
        import torch
        import numpy as np
        import torch.nn.functional as F
        from torchvision import transforms as T
        from src.explain import gradcam, rollout
        prepared = prepare_image(image)
        x = self.tensor([prepared])
        with self.lock:
            # rollout modifica fused_attn: restaurarlo siempre, también si falla.
            blocks = list(getattr(self.model.backbone, "blocks", [])) if self.info["architecture"].startswith("vit") else []
            flags = [block.attn.fused_attn for block in blocks]
            try:
                if blocks:
                    with torch.no_grad():
                        heat, _ = rollout(self.model, x)
                    method = "Attention Rollout (atención CLS; no específica de clase)"
                else:
                    with torch.enable_grad():
                        heat, _ = gradcam(self.model, x.requires_grad_(True))
                    method = "Grad-CAM del agregado BCC + MEL + SCC; no de la clase ganadora"
            finally:
                for block, flag in zip(blocks, flags):
                    block.attn.fused_attn = flag
            mask = F.interpolate(heat.detach().cpu()[:, None], (224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
        if not np.isfinite(mask).all():
            raise ValueError("El método original generó un mapa no finito para esta imagen.")
        base = np.asarray(T.CenterCrop(224)(T.Resize(224)(prepared)), dtype=np.float32) / 255
        color = np.stack([mask, 0.3 * (1 - mask), 1 - mask], axis=-1)
        return np.uint8(np.clip(0.65 * base + 0.35 * color, 0, 1) * 255), method
