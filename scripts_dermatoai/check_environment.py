import importlib.metadata
import json
import platform
import torch
def main():
    report={"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),
            "cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "versions":{name:importlib.metadata.version(name) for name in ["torch","torchvision","timm","streamlit","sentence-transformers","numpy","pandas","scikit-learn","pypdf"]}}
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
