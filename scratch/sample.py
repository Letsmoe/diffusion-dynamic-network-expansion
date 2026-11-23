import sys, json, torch
sys.path.insert(0, "/home/moritz/Documents/personal/third_party/HobbyLM")
sys.path.insert(0, "/home/moritz/Documents/personal/diffusion-dyanmic-network-expansion/experiments")
from transformers import AutoTokenizer
from hobbylm.diffusion import generate
from q1_add_expert import load_model

tok = AutoTokenizer.from_pretrained("gpt2")
model = load_model("cpu")
prompts = [
    "USER: Write a Python function that reverses a string.\nASSISTANT:",
    "USER: What is the capital of France?\nASSISTANT:",
    "def fibonacci(n):\n",
]
torch.manual_seed(0)
for p in prompts:
    ids = torch.tensor([tok(p)["input_ids"]], device="cpu")
    out = generate(model, ids, gen_len=96, steps=128, temperature=0.2,
                   rep_penalty=1.5, remask_steps=2, valid_vocab=50257)
    print("PROMPT:", repr(p))
    print("OUTPUT:", repr(tok.decode(out[0].tolist(), skip_special_tokens=True)))
    print("---", flush=True)
import os; os._exit(0)
