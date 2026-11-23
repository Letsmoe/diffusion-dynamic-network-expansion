import sys
sys.path.insert(0, "/home/moritz/Documents/personal/third_party/HobbyLM")
import torch
from safetensors.torch import load_file
from hobbylm.config import ModelConfig
from hobbylm.model import MoETransformer
from hobbylm.diffusion import forward_mask

cfg = ModelConfig(**{k: v for k, v in __import__("json").load(open("/home/moritz/Documents/personal/models/HobbyLM-Diffusion/config.json")).items() if k != "preset"})
model = MoETransformer(cfg)
sd = load_file("/home/moritz/Documents/personal/models/HobbyLM-Diffusion/model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing[:5], len(missing))
print("unexpected:", unexpected[:5], len(unexpected))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
