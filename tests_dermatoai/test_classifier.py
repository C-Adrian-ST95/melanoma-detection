import io
import numpy as np
import pytest
import torch
from PIL import Image
from dermatoai.classifier import Classifier, image_from_bytes, prepare_image
from src.train import Net
from src.corruptions import build_transform

@pytest.fixture(scope="module",params=["efficientnet_b0","vit_small_patch16_224"])
def classifier(request,tmp_path_factory):
    torch.set_num_threads(1);torch.manual_seed(12)
    directory=tmp_path_factory.mktemp("synthetic_weights")
    path=directory/f"synthetic_{request.param}.pt"
    model=Net(request.param,pretrained=False)
    torch.save({"state_dict":model.state_dict(),"config":{"model":request.param,"img_size":224},
                "run_id":path.stem,"epoch":0},path)
    # Pesos aleatorios SOLO para probar software; no son modelos entrenados ni resultados clínicos.
    return Classifier(path)

def test_original_preprocessing_and_single_batch(classifier):
    img=Image.fromarray(np.full((300,480,3),180,dtype=np.uint8))
    x=classifier.tensor([img])
    expected=build_transform("none",False)(prepare_image(img))[None]
    assert torch.equal(x,expected) and x.shape==(1,3,224,224)
    assert 0<=x.min()<=x.max()<=1
    a=classifier.predict(img);b=classifier.predict_many([img,img])
    assert list(a["scores"])==["ACK","BCC","MEL","NEV","SCC","SEK"]
    assert sum(a["scores"].values())==pytest.approx(1,abs=1e-6)
    assert list(a["scores"].values())==pytest.approx(list(b[0]["scores"].values()),abs=1e-5)
    assert a["sha256"] and a["epoch"]==0

def test_original_explanations(classifier):
    img=Image.new("RGB",(256,256),(170,130,100))
    before=[b.attn.fused_attn for b in classifier.model.backbone.blocks] if classifier.info["architecture"].startswith("vit") else []
    overlay,method=classifier.explain(img)
    assert overlay.shape==(224,224,3) and overlay.dtype==np.uint8
    assert method
    if before:
        assert before==[b.attn.fused_attn for b in classifier.model.backbone.blocks]

def test_missing_and_bad_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        Classifier(tmp_path/"missing.pt")
    file=tmp_path/"invalid.pt";torch.save({"state_dict":{}},file)
    with pytest.raises(ValueError,match="BEST"):
        Classifier(file)

def test_invalid_image_and_rgb():
    with pytest.raises(Exception):
        image_from_bytes(b"not an image")
    b=io.BytesIO();Image.new("L",(10,10)).save(b,format="PNG")
    assert image_from_bytes(b.getvalue()).mode=="RGB"
