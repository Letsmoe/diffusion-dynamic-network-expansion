import sys, torch
sys.path.insert(0, ".")
import qwen_refine as qr
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(qr.PATH)
model = qr.load_model()
user = "Complete the following Python function. Return the full function in a ```python block.\n\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    \"\"\"\n"
text = tok.apply_chat_template([{"role":"user","content":user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(qr.DEVICE)
torch.manual_seed(0)
snaps = qr.generate(model, ids, gen_len=256, steps=128, rounds=0, remask_frac=0.3)
print("RAW_TOKENS", (snaps[0]==qr.MASK_ID).sum().item(), "masked of 256")
print(repr(tok.decode(snaps[0], skip_special_tokens=False)))
import os, sys; sys.stdout.flush(); os._exit(0)
