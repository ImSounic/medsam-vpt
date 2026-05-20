import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.medsam import load_medsam
from src.models.methods import setup_method
from src.data.isic import ISIC2018, isic_collate
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

# Build a small fixed batch from ISIC test (32 images is plenty for stable feature distances)
ds = ISIC2018(root=REPO / 'data', split='test', image_size=1024)
ds.items = ds.items[:32]
loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=isic_collate, shuffle=False)
images = torch.cat([b['image'] for b in loader], dim=0).to(device)
print(f'feature-shift probe set: {images.shape[0]} images')

# Get baseline encoder features from frozen MedSAM
base_sam = load_medsam(REPO / 'checkpoints/medsam_vit_b.pth', arch='vit_b', device=device)
base_sam.eval()
with torch.no_grad():
    base_feats = base_sam.image_encoder(images).flatten(1)

base_norm = base_feats.norm(dim=1).mean()
print(f'baseline mean feature norm: {base_norm.item():.4f}')
print()

results = {}
methods_to_test = [
    ('decoder_only', {}),
    ('vpt_shallow',  {}),
    ('vpt_deep',     {}),
    ('lora',         {}),
    ('full_ft',      {}),
]

for method, kwargs in methods_to_test:
    # Rebuild SAM fresh, apply method wrapper, load checkpoint
    sam = load_medsam(REPO / 'checkpoints/medsam_vit_b.pth', arch='vit_b', device=device)
    ckpt_path = REPO / f'checkpoints/runs/{method}_seed0/best.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    method_kwargs = ckpt.get('config', {}).get('method_kwargs', {}) or {}
    setup_method(sam, method, **method_kwargs)
    state = {k: v.to(device) for k, v in ckpt['trainable_state'].items()}
    sam.load_state_dict(state, strict=False)
    sam.eval()

    with torch.no_grad():
        ft_feats = sam.image_encoder(images).flatten(1)

    # Per-sample L2 distance, normalized by baseline feature norm
    rel_shift = ((ft_feats - base_feats).norm(dim=1) / base_feats.norm(dim=1)).mean()
    results[method] = rel_shift.item()
    print(f'{method:15s}  mean relative feature shift: {rel_shift.item():.4f}')
    del sam
    if device == 'cuda':
        torch.cuda.empty_cache()

# Bar chart
labels = {'decoder_only':'Decoder-only','vpt_shallow':'VPT-shallow','vpt_deep':'VPT-deep','lora':'LoRA','full_ft':'Full FT'}
fig, ax = plt.subplots(figsize=(8, 4.5))
methods = list(results.keys())
bars = ax.bar([labels[m] for m in methods], [results[m] for m in methods],
              color=['#1f77b4','#ff7f0e','#d62728','#2ca02c','#9467bd'],
              edgecolor='black', linewidth=0.6)
for bar, m in zip(bars, methods):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{results[m]:.4f}', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Mean relative feature shift\n||f_finetuned - f_base|| / ||f_base||')
ax.set_title('How much does each PEFT method shift the encoder OUTPUT?\n(measures additive perturbations correctly, unlike weight delta)')
plt.xticks(rotation=15)
plt.tight_layout()
out = REPO / 'results/figures/encoder_feature_shift.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'\nfigure saved to {out}')
