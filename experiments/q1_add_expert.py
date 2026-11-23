"""Q1: add one routed expert to every MoE block and retrain the routers.

Arms (--arm):
  baseline   : original checkpoint, no training (reference numbers)
  router     : train the routers of all MoE blocks only (no new expert)
  add_expert : add one expert to every MoE block, train routers + new experts

Only the gates and expert weights of the MoE blocks are trainable; everything else is frozen.
Retention = masked CE on general text (fineweb-edu). Gain = masked CE on code (codeparrot-clean, python).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
HOBBYLM = Path(os.environ.get("HOBBYLM_REPO", "/home/moritz/Documents/personal/third_party/HobbyLM"))
MODEL_DIR = Path(os.environ.get("HOBBYLM_WEIGHTS", "/home/moritz/Documents/personal/models/HobbyLM-Diffusion"))
sys.path.insert(0, str(HOBBYLM))

from hobbylm.config import ModelConfig  # noqa: E402
from hobbylm.diffusion import forward_mask  # noqa: E402
from hobbylm.model import MoETransformer  # noqa: E402
from hobbylm.moe import SwiGLUWeights  # noqa: E402

MASK_ID = 50257


def load_model(device):
    raw = json.loads((MODEL_DIR / "config.json").read_text())
    raw.pop("preset", None)
    cfg = ModelConfig(**raw)
    model = MoETransformer(cfg)
    model.load_state_dict(load_file(MODEL_DIR / "model.safetensors"), strict=True)
    return model.to(device)


class GrownExperts(SwiGLUWeights):
    """Expert stack with the original experts frozen and one trainable new expert.

    The original w13/w2 are kept as buffers, so the optimizer never allocates state for them.
    w13/w2 are presented as the full (E+1, ...) stacks, so the router and forward pass are unchanged.
    """
    def __init__(self, base: SwiGLUWeights, init_std: float):
        nn.Module.__init__(self)
        self.n_experts = base.n_experts + 1
        self.d_model = base.d_model
        self.ffn = base.ffn
        self.register_buffer("base_w13", base.w13.detach().clone())
        self.register_buffer("base_w2", base.w2.detach().clone())
        # w2 starts at zero, so the new expert adds nothing to the output at step 0
        self.new_w13 = nn.Parameter(torch.randn(1, self.d_model, 2 * self.ffn, device=base.w13.device) * init_std)
        self.new_w2 = nn.Parameter(torch.zeros(1, self.ffn, self.d_model, device=base.w13.device))

    @property
    def w13(self):
        return torch.cat([self.base_w13, self.new_w13])

    @property
    def w2(self):
        return torch.cat([self.base_w2, self.new_w2])


def add_expert(moe, init_std=0.02):
    """Grow one MoE block by one routed expert, in place.

    The gate gets one extra row and the aux-free bias one extra zero. The new expert's
    router score can still displace a top-k expert, so the step-0 loss is measured by the caller.
    """
    E, d = moe.gate.weight.shape
    gate = nn.Linear(d, E + 1, bias=False, device=moe.gate.weight.device, dtype=moe.gate.weight.dtype)
    with torch.no_grad():
        gate.weight[:E] = moe.gate.weight
        gate.weight[E:].normal_(0.0, init_std)
    moe.gate = gate
    moe.experts = GrownExperts(moe.experts, init_std)
    moe.expert_bias = torch.cat([moe.expert_bias, moe.expert_bias.new_zeros(1)])
    moe.n_experts = E + 1


def moe_blocks(model):
    """(block_idx, MoE layer) for every MoE block; block 0 is dense in HobbyLM-Diffusion."""
    return [(i, blk.ffn) for i, blk in enumerate(model.blocks) if blk.is_moe]


def freeze_except_routers(model, grown=False):
    """Gates always train. With grown=True the new expert in each block trains too; the rest of the experts stay frozen."""
    model.requires_grad_(False)
    for _, moe in moe_blocks(model):
        moe.gate.weight.requires_grad_(True)
        if grown:
            moe.experts.new_w13.requires_grad_(True)
            moe.experts.new_w2.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def batches(stream_iter, tokenizer, seq_len, batch_size, device):
    buf = []
    batch = []
    for ex in stream_iter:
        text = ex.get("content") or ex.get("text")
        if not text:
            continue
        buf.extend(tokenizer(text)["input_ids"] + [tokenizer.eos_token_id])
        while len(buf) >= seq_len:
            batch.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(batch) == batch_size:
                yield torch.tensor(batch, device=device)
                batch = []


@torch.no_grad()
def evaluate(model, data_iter, n_batches, device, seed=0):
    """Masked CE averaged over n_batches, with a fixed mask pattern (generator seed)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for i, ids in enumerate(data_iter):
        if i >= n_batches:
            break
        noisy, labels, p = forward_mask(ids, MASK_ID, generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, info = model(noisy, labels, p_mask=p)
        total += info["ce"].item()
        print(f"  eval batch {i + 1}/{n_batches}", flush=True)
    model.train()
    return total / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "router", "add_expert"], required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8, help="effective sequences per optimizer step")
    ap.add_argument("--micro", type=int, default=2, help="sequences per forward/backward pass")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "q1")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained("gpt2")

    model = load_model(device)
    model.set_bias_update_rate(0.0)  # keep the aux-free bias fixed; only gate + expert weights learn

    grown = args.arm == "add_expert"
    if grown:
        for _, moe in moe_blocks(model):
            add_expert(moe)

    def code_eval():
        ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                          streaming=True)
        return batches(ds, tok, args.seq, args.micro, device)

    def general_eval():
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                          streaming=True)
        return batches(ds, tok, args.seq, args.micro, device)

    results = {"arm": args.arm, "moe_blocks": [i for i, _ in moe_blocks(model)]}
    print("eval before training: code", flush=True)
    results["code_before"] = evaluate(model, code_eval(), args.eval_batches, device)
    print("eval before training: general", flush=True)
    results["general_before"] = evaluate(model, general_eval(), args.eval_batches, device)
    print(json.dumps(results), flush=True)

    if args.arm != "baseline":
        params = freeze_except_routers(model, grown)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        model.train()
        train_iter = batches(
            load_dataset("codeparrot/codeparrot-clean", split="train",
                         streaming=True).shuffle(seed=args.seed, buffer_size=10_000),
            tok, args.seq, args.micro, device)
        for step in range(args.steps):
            # gradient accumulation: each micro-batch is --micro sequences, --batch is the effective size
            opt.zero_grad(set_to_none=True)
            for _ in range(args.batch // args.micro):
                ids = next(train_iter)
                noisy, labels, p = forward_mask(ids, MASK_ID)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = model(noisy, labels, p_mask=p)
                (loss / (args.batch // args.micro)).backward()
            opt.step()
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f}", flush=True)

        print("eval after training: code", flush=True)
        results["code_after"] = evaluate(model, code_eval(), args.eval_batches, device)
        print("eval after training: general", flush=True)
        results["general_after"] = evaluate(model, general_eval(), args.eval_batches, device)
        print(json.dumps(results), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.arm}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    # Streams from the eval loop can be left open (evaluate stops early), and their
    # background HTTP threads keep the interpreter alive after main() returns.
    sys.stdout.flush()
    os._exit(0)
