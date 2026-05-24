"""One-shot preheat readiness check for Path X v2 training on AutoDL."""
import os
import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=20)

    def run(cmd, t=60, label=None):
        if label:
            print(f"--- {label} ---")
        _, o, e = c.exec_command(cmd, timeout=t)
        out = o.read().decode(errors="ignore").rstrip()
        err = e.read().decode(errors="ignore").rstrip()
        if out:
            print(out)
        if err:
            print("[stderr]", err[:400])
        print()

    PY = "/root/miniconda3/bin/python"
    REPO = "/root/autodl-tmp/IntelligenceCamera"

    print("=== [1] IntelligenceCamera repo ===")
    run(f"ls {REPO}/ 2>/dev/null | head -40", label="repo top")
    run(f"ls {REPO}/training/ 2>/dev/null | head -30", label="training/")
    run(f"test -d {REPO}/training/firered_baseline && echo HAS_DIR || echo NO_DIR", label="firered_baseline dir?")
    run(f"ls {REPO}/training/firered_baseline/ 2>/dev/null | head -30", label="firered_baseline/")
    run(f"test -f {REPO}/training/firered_baseline/train_lut.py && wc -l {REPO}/training/firered_baseline/train_lut.py || echo NO_train_lut", label="train_lut.py?")
    run(f"test -f {REPO}/training/firered_baseline/implicit_head.py && wc -l {REPO}/training/firered_baseline/implicit_head.py || echo NO_implicit_head", label="implicit_head.py?")
    run(f"ls {REPO}/scripts/ 2>/dev/null | head -20", label="scripts/")
    run(f"ls {REPO}/scripts/local/ 2>/dev/null | head -10", label="scripts/local/")
    run(f"ls {REPO}/tools/data/data_prep/ 2>/dev/null | head -10", label="tools/data/data_prep/")

    print("=== [2] backbone / pretrained ckpts ===")
    run(f"find /root/autodl-tmp/checkpoints {REPO}/checkpoints {REPO}/outputs -maxdepth 5 -name '*v11*' 2>/dev/null | head -20", label="*v11* anywhere")
    run("find /root/autodl-tmp -maxdepth 6 -name 'pathX_*' -o -name 'venus_lut*' 2>/dev/null | head -10", label="pathX/venus_lut")
    run(f"find {REPO}/checkpoints -maxdepth 4 -name 'best.pt' -o -name 'final.pt' -o -name 'last.pt' 2>/dev/null | head -10", label="repo ckpts")

    print("=== [3] master jsonl on AutoDL ===")
    run("find /root/autodl-tmp -maxdepth 7 -name 'master*.jsonl' -o -name '*pathY*.jsonl' -o -name '*pseudo*.jsonl' 2>/dev/null | head -20", label="jsonls anywhere")
    run(f"ls {REPO}/data/ 2>/dev/null | head -20", label="repo/data/")

    print("=== [4] FiveK / PPR10K ===")
    run("ls /root/autodl-tmp/fivek_jpeg/ 2>/dev/null | wc -l", label="fivek_jpeg count")
    run("ls /root/autodl-tmp/fivek_expert_c/ 2>/dev/null | wc -l", label="fivek_expert_c count")

    print("=== [5] Python env ===")
    run("ls /root/miniconda3/envs/ 2>/dev/null", label="conda envs")
    run(f"{PY} --version", label="base python")
    run(f"{PY} -c 'import sys; print(sys.executable)'", label="executable")
    run(f"{PY} -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO_GPU\")'", label="torch")
    run(f"{PY} -c 'import lpips, torchvision, kornia, einops, transformers, accelerate, peft; print(\"lpips:\", lpips.__version__); print(\"tv:\", torchvision.__version__); print(\"kornia:\", kornia.__version__); print(\"transformers:\", transformers.__version__)'", label="key deps")
    run(f"{PY} -c 'import timm, safetensors, omegaconf; print(\"timm:\", timm.__version__); print(\"safetensors:\", safetensors.__version__)'", label="timm/safetensors")

    print("=== [6] GPU ===")
    run("nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader")

    print("=== [7] Path Y staging dirs ===")
    run("ls /root/autodl-tmp/pathY_outputs/ /root/autodl-tmp/pathY_captions/ 2>/dev/null")
    run("ls /root/autodl-tmp/pathY_outputs/wb /root/autodl-tmp/pathY_outputs/wb_cooler 2>/dev/null | wc -l", label="png counts")

    print("=== [8] Disk ===")
    run("df -h /root/autodl-tmp | tail -1")

    c.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
