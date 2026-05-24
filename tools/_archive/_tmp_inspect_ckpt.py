"""Inspect local implicit_head checkpoint."""
import torch
from pathlib import Path

ckpts = [
    "checkpoints/lut_v11a_pathX_implicit_head/best.pt",
    "checkpoints/lut_v11a_action_gated_context/best.pt",
]

for p in ckpts:
    fp = Path(p)
    if not fp.exists():
        print(f"MISSING: {p}")
        continue
    ckpt = torch.load(fp, map_location="cpu", weights_only=False)
    ca = ckpt["args"]
    print(f"\n=== {p} ===")
    print(f"  val_psnr={ckpt.get('val_psnr', 0):.2f} @ Ep{ckpt.get('epoch', '?')}")
    print(f"  use_implicit_head={ca.get('use_implicit_head', False)}")
    print(f"  jsonl_path={ca.get('jsonl_path', '?')}")
    impl_keys = [k for k in ckpt["model_state_dict"] if "implicit" in k]
    print(f"  implicit_head keys: {len(impl_keys)}")
    if impl_keys:
        print(f"    {impl_keys[:5]}")
