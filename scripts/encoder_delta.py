import torch
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
base = torch.load(REPO / 'checkpoints/medsam_vit_b.pth', map_location='cpu', weights_only=False)
base_state = base.get('model', base)

results = {}
for method in ['decoder_only', 'vpt_shallow', 'vpt_deep', 'lora', 'full_ft']:
    ckpt_path = REPO / f'checkpoints/runs/{method}_seed0/best.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    trained_state = ckpt['trainable_state']

    # Only compare keys that exist in both base and trained
    encoder_deltas = []
    for k, v in trained_state.items():
        if 'image_encoder' not in k:
            continue
        if k not in base_state:
            # LoRA/VPT add new params not in base — skip those (separate analysis)
            continue
        delta = (v.float() - base_state[k].float()).norm() / (base_state[k].float().norm() + 1e-9)
        encoder_deltas.append(delta.item())

    if encoder_deltas:
        results[method] = sum(encoder_deltas) / len(encoder_deltas)
    else:
        results[method] = 0.0  # decoder_only doesn't touch encoder; VPT/LoRA add NEW params (handled separately below)

    print(f'{method:15s}  mean relative encoder weight change: {results[method]:.4f}')

# Bar chart
fig, ax = plt.subplots(figsize=(7, 4))
methods = list(results.keys())
ax.bar(methods, [results[m] for m in methods])
ax.set_ylabel('Mean relative encoder weight change\n||W_finetuned - W_base|| / ||W_base||')
ax.set_title('How much does each PEFT method actually move the encoder?')
plt.xticks(rotation=20)
plt.tight_layout()
out = REPO / 'results/figures/encoder_delta.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(f'figure saved to {out}')
