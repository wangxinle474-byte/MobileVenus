"""Pull final training summary from AutoDL log + ckpt metadata."""
import os, sys, paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"
PY = "/root/miniconda3/bin/python"


def run(c, cmd, t=60):
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip()


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)

    # Latest log
    log = run(c, f"ls -t {REPO}/logs/pathX_v2_*.log 2>/dev/null | head -1")
    print(f"[log] {log}")

    print()
    print("=== Tail (last 60 lines) ===")
    print(run(c, f"tail -n 60 {log}", t=30))

    print()
    print("=== Best ckpt summary ===")
    ckpt = f"{REPO}/checkpoints/lut_v11a_pathX_v2_implicit_head/best.pt"
    print(run(c, f"ls -la {ckpt}"))
    cmd = (
        f"{PY} -c \""
        "import torch, json; "
        f"ck = torch.load('{ckpt}', map_location='cpu', weights_only=False); "
        "info = {"
        " 'epoch': ck.get('epoch'),"
        " 'val_psnr': float(ck.get('val_psnr', 0)),"
        " 'val_l1': float(ck.get('val_l1', 0)) if 'val_l1' in ck else None,"
        " 'state_keys': len(ck.get('model_state', ck.get('state_dict', {})))"
        "}; "
        "print(json.dumps(info, indent=2))"
        "\""
    )
    print(run(c, cmd, t=60))

    print()
    print("=== Per-epoch val_psnr trajectory ===")
    print(run(c, f"grep -E 'Ep\\s+[0-9]+/40' {log} | grep 'val:' "
              f"| awk '{{for(i=1;i<=NF;i++) if($i~/PSNR=.*dB/) {{print $2,$i; break}}}}' "
              f"| head -42"))

    print()
    print("=== Best epoch lines ===")
    print(run(c, f"grep 'best val_psnr' {log} | tail -10"))

    print()
    print("=== Training duration ===")
    print(run(c, f"head -3 {log} | grep -E '20[0-9]+'"))
    print(run(c, f"tail -5 {log} | grep -E '20[0-9]+'"))

    c.close()


if __name__ == "__main__":
    main()
