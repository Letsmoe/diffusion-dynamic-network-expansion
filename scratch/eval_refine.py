"""Verifier-based refinement eval for the Qwen3-0.6B diffusion checkpoint on HumanEval.

For each problem: fill once, then refine R times. After every snapshot, the extracted code runs
against the problem's tests in a subprocess. Reports pass rate per snapshot, the oracle
(any snapshot passes), and how often the code changed from the fill.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_refine as qr  # noqa: E402

import os

OUT = Path(os.environ.get("EVAL_OUT", Path(__file__).resolve().parent / "eval_refine.json"))
GEN_LEN = int(os.environ.get("GEN_LEN", 256))
STEPS = int(os.environ.get("STEPS", 128))
ROUNDS = int(os.environ.get("ROUNDS", 3))
REMASK = float(os.environ.get("REMASK", 0.3))
N_PROBLEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 164


def extract_code(text):
    # closed fence, else an unclosed fence running to the end, else the raw text
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S) or re.search(r"```(?:python)?\n(.*)", text, re.S)
    return m.group(1) if m else text


def passes(problem, completion):
    """Run the completion plus the problem's tests in a fresh subprocess. Returns True on pass."""
    program = completion + "\n\n" + problem["test"] + f"\n\ncheck({problem['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, timeout=10, cwd=d)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def main():
    tok = AutoTokenizer.from_pretrained(qr.PATH)
    model = qr.load_model()
    problems = list(load_dataset("openai/openai_humaneval", split="test"))[:N_PROBLEMS]

    records = []
    torch.manual_seed(0)
    for i, p in enumerate(problems):
        user = "Complete the following Python function. Return the full function in a ```python block.\n\n" + p["prompt"]
        text = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(qr.DEVICE)
        snaps = qr.generate(model, ids, gen_len=GEN_LEN, steps=STEPS, rounds=ROUNDS, remask_frac=REMASK)
        texts = [tok.decode(s, skip_special_tokens=True) for s in snaps]
        codes = [extract_code(t) for t in texts]
        results = [passes(p, c) for c in codes]
        changed = [c != codes[0] for c in codes]
        records.append({"task": p["task_id"], "passes": results, "changed_from_fill": changed, "codes": codes})
        print(f"{i + 1}/{len(problems)} {p['task_id']} passes={results}", flush=True)

    n = len(records)
    per_snap = [sum(r["passes"][k] for r in records) / n for k in range(ROUNDS + 1)]
    oracle = sum(any(r["passes"]) for r in records) / n
    summary = {
        "n": n,
        "pass_rate_per_snapshot": per_snap,  # index 0 = fill, 1..R = refinement rounds
        "oracle_any_snapshot": oracle,
        "changed_from_fill_rate": [sum(r["changed_from_fill"][k] for r in records) / n for k in range(ROUNDS + 1)],
    }
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
