"""Sample from the Qwen3-0.6B masked-diffusion checkpoint and track refinement rounds.

Fill: start from an all-mask answer, commit the most confident masked positions over `steps` passes.
Refine: re-mask the least confident fraction of the answer, re-predict them, repeat `rounds` times.
Prints the answer after the fill and after each refinement round.
"""
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

PATH = "/home/moritz/Documents/personal/models/qwen3-0.6b-diffusion-mdlm"
MASK_ID = 151669
DEVICE = "cuda"


@torch.no_grad()
def predict(model, x, gen_start):
    """Return (argmax tokens, confidence) for the answer region only."""
    logits = model(input_ids=x).logits[:, gen_start:].float()
    logits[..., MASK_ID] = -float("inf")  # never emit the mask token
    probs = logits.softmax(-1)
    conf, pred = probs.max(-1)
    return pred, conf


@torch.no_grad()
def generate(model, prompt_ids, gen_len, steps, rounds, remask_frac):
    P = prompt_ids.shape[1]
    x = torch.cat([prompt_ids, torch.full((1, gen_len), MASK_ID, device=DEVICE)], dim=1)
    snapshots = []

    # fill: commit the most confident masked positions over `steps` passes
    for s in range(steps):
        pred, conf = predict(model, x, P)
        still = x[0, P:] == MASK_ID
        if not still.any():
            break
        remaining_steps = steps - s
        k = max(1, int(still.sum()) // remaining_steps)
        conf = torch.where(still, conf[0], torch.full_like(conf[0], -1.0))
        idx = conf.topk(k).indices
        x[0, P + idx] = pred[0, idx]
    snapshots.append(x[0, P:].clone())

    # refine: re-mask the least confident tokens, then re-predict them with full context
    for _ in range(rounds):
        pred, conf = predict(model, x, P)
        r = max(1, int(gen_len * remask_frac))
        low = conf[0].topk(r, largest=False).indices
        x[0, P + low] = MASK_ID
        pred, _ = predict(model, x, P)
        still = (x[0, P:] == MASK_ID).nonzero(as_tuple=True)[0]
        x[0, P + still] = pred[0, still]
        snapshots.append(x[0, P:].clone())
    return snapshots


def load_model():
    # Import the checkpoint's modeling file directly. Its `import dllm` sits under
    # `if __name__ == "__main__"`, so importing it as a module doesn't need dllm installed.
    import importlib.util
    from safetensors.torch import load_file
    spec = importlib.util.spec_from_file_location("a2d_modeling", f"{PATH}/modeling_qwen3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = mod.A2DQwen3Config.from_pretrained(PATH)
    model = mod.A2DQwen3LMHeadModel(config)
    # transformers>=5 decoder layers don't set attention_type; the checkpoint's modeling code reads it
    for layer, kind in zip(model.model.layers, config.layer_types):
        layer.attention_type = kind
    missing, unexpected = model.load_state_dict(load_file(f"{PATH}/model.safetensors"), strict=False)
    print("missing:", missing, "unexpected:", unexpected[:5], flush=True)
    return model.to(torch.bfloat16).to(DEVICE).eval()


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = load_model()

    prompts = [
        "Write a Python function that reverses a string.",
        "Write a Python function that returns the nth Fibonacci number.",
    ]
    torch.manual_seed(0)
    for user in prompts:
        msgs = [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        snaps = generate(model, ids, gen_len=128, steps=64, rounds=3, remask_frac=0.3)
        print("PROMPT:", user)
        for i, snap in enumerate(snaps):
            label = "fill" if i == 0 else f"refine {i}"
            print(f"--- {label} ---")
            print(tok.decode(snap, skip_special_tokens=True))
        print("=" * 60, flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
