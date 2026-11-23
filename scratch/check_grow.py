import sys, torch
sys.path.insert(0, "/home/moritz/Documents/personal/diffusion-dyanmic-network-expansion/experiments")
from q1_add_expert import load_model, add_expert, freeze_except_routers, moe_blocks
from hobbylm.diffusion import forward_mask

torch.manual_seed(0)
model = load_model("cpu")
ids = torch.randint(0, 50000, (1, 64))
noisy, labels, p = forward_mask(ids, 50257, generator=torch.Generator().manual_seed(0))
model.eval()
with torch.no_grad():
    _, base = model(noisy, labels, p_mask=p)
for _, moe in moe_blocks(model):
    add_expert(moe)
with torch.no_grad():
    _, grown = model(noisy, labels, p_mask=p)
print("base ce", base["ce"].item(), "grown ce (step 0)", grown["ce"].item())
params = freeze_except_routers(model, grown=True)
n = sum(p.numel() for p in params)
print("trainable tensors", len(params), "trainable params", n)
print("expert shapes", moe_blocks(model)[0][1].experts.w13.shape, moe_blocks(model)[0][1].experts.w2.shape)
import os; sys.stdout.flush(); os._exit(0)
