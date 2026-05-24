"""Re-upload fixed train_lut.py and verify --help works on AutoDL."""
import os
import paramiko
from pathlib import Path

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
PY = "/root/miniconda3/bin/python"
REPO = "/root/autodl-tmp/IntelligenceCamera"
LOCAL_ROOT = Path(__file__).resolve().parents[1]


def run(c, cmd, t=60, label=None):
    if label:
        print(f"--- {label} ---")
    _, o, e = c.exec_command(cmd, timeout=t)
    out = o.read().decode(errors="ignore").rstrip()
    err = e.read().decode(errors="ignore").rstrip()
    if out:
        print(out)
    if err:
        print("[stderr]", err[:500])
    print()


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=20)
    sftp = c.open_sftp()

    print("=== [1] Re-upload fixed train_lut.py ===")
    local = LOCAL_ROOT / "training" / "firered_baseline" / "train_lut.py"
    remote = f"{REPO}/training/firered_baseline/train_lut.py"
    sftp.put(str(local), remote)
    print(f"  uploaded {local.stat().st_size} bytes")
    print()

    print("=== [2] train_lut.py --help (verify imports OK) ===")
    run(c, f"cd {REPO} && {PY} training/firered_baseline/train_lut.py --help 2>&1 | head -40")

    print("=== [3] Quick smoke: instantiate model + load backbone (CUDA) ===")
    smoke_path = "/tmp/preheat_smoke.py"
    smoke_code = f"""
import sys, os
sys.path.insert(0, '{REPO}')
os.chdir('{REPO}')
import torch
from training.firered_baseline.implicit_head import ImplicitResidualHead
from models.vision_encoder import MobileViTSmall
from models.lut3d import Basis3DLUT
from models.nilut import NILUT
from models.vera_renderer import VeraRenderer
print('all local imports OK')

ck = torch.load('{REPO}/checkpoints/lut_v11a_action_gated_context/best.pt',
                map_location='cpu', weights_only=False)
sd = ck['model_state_dict']
print(f'backbone state_dict has {{len(sd)}} keys')
print(f'sample keys: {{sorted(sd.keys())[:3]}}')
print(f'epoch: {{ck.get("epoch")}}, val_l1: {{ck.get("val_l1"):.4f}}, val_psnr: {{ck.get("val_psnr"):.2f}}')

# also test ImplicitResidualHead can be instantiated
head = ImplicitResidualHead(base_ch=32, gate_init=1.0)
n_params = sum(p.numel() for p in head.parameters())
print(f'ImplicitResidualHead OK, {{n_params/1e6:.2f}}M params')

# CUDA sanity (forward needs orig, refined, action_onehot)
if torch.cuda.is_available():
    head = head.cuda()
    orig = torch.rand(2, 3, 256, 256, device='cuda')
    refined = torch.rand(2, 3, 256, 256, device='cuda')
    action_onehot = torch.zeros(2, 5, device='cuda')
    action_onehot[:, 4] = 1.0  # wb
    y = head(orig, refined, action_onehot)
    print(f'CUDA forward OK: out {{y.shape}}, range [{{y.min():.3f}}, {{y.max():.3f}}]')
print('SMOKE_OK')
"""
    with sftp.open(smoke_path, "w") as f:
        f.write(smoke_code)
    run(c, f"{PY} {smoke_path}", t=120)
    sftp.remove(smoke_path)

    print("=== [4] Final readiness ===")
    run(c, f"ls -lah {REPO}/training/firered_baseline/*.py | head -10", label="firered_baseline files")
    run(c, f"ls -lah {REPO}/scripts/autodl/train_pathX_v2.sh", label="train_pathX_v2.sh")
    run(c, "nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader", label="GPU")

    sftp.close()
    c.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
