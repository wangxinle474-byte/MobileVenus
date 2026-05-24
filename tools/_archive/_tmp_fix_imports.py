"""Fix missing modules on AutoDL for eval_track2.py."""
import os, sys
import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"
PY = "/root/miniconda3/bin/python"
LOCAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(c, cmd, t=30):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.channel.recv_exit_status(), o.read().decode(errors="ignore").strip(), e.read().decode(errors="ignore").strip()


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)
    sftp = c.open_sftp()

    # 1. Check what's in tools/ on AutoDL
    rc, out, _ = run(c, f"ls {REPO}/tools/*.py | head -20")
    print(f"AutoDL tools/ files:\n{out}\n")

    # 2. Check for __init__.py
    rc, out, _ = run(c, f"test -f {REPO}/tools/__init__.py && echo EXISTS || echo MISSING")
    print(f"tools/__init__.py: {out}")

    # 3. Upload __init__.py if missing
    init_local = os.path.join(LOCAL_ROOT, "tools", "__init__.py")
    if os.path.exists(init_local):
        sftp.put(init_local, f"{REPO}/tools/__init__.py")
        print(f"uploaded tools/__init__.py (from local)")
    else:
        # Create empty __init__.py
        run(c, f"touch {REPO}/tools/__init__.py")
        print("created empty tools/__init__.py on AutoDL")

    # 4. Upload merge_jsonl_for_joint_training.py
    src = os.path.join(LOCAL_ROOT, "tools", "merge_jsonl_for_joint_training.py")
    dst = f"{REPO}/tools/merge_jsonl_for_joint_training.py"
    if os.path.exists(src):
        sftp.put(src, dst)
        rc, out, _ = run(c, f"wc -l {dst}")
        print(f"uploaded merge_jsonl_for_joint_training.py ({out})")
    else:
        print(f"LOCAL MISSING: {src}")

    # 5. Also check training/__init__.py and training/firered_baseline/__init__.py
    for sub in ["training", "training/firered_baseline", "models"]:
        init_remote = f"{REPO}/{sub}/__init__.py"
        rc, out, _ = run(c, f"test -f {init_remote} && echo EXISTS || echo MISSING")
        print(f"{sub}/__init__.py: {out}")
        if out == "MISSING":
            init_local2 = os.path.join(LOCAL_ROOT, sub, "__init__.py")
            if os.path.exists(init_local2):
                sftp.put(init_local2, init_remote)
                print(f"  uploaded from local")
            else:
                run(c, f"touch {init_remote}")
                print(f"  created empty")

    # 6. Re-test import
    check = (
        f'{PY} -c "import sys; sys.path.insert(0, \'{REPO}\'); '
        f'from tools._lib.merge_jsonl_for_joint_training import extract_fivek_stem; '
        f'print(\'OK: extract_fivek_stem imported\')"'
    )
    rc, out, err = run(c, check, t=20)
    print(f"\nimport test: rc={rc}")
    print(f"  {out}")
    if err:
        print(f"  err: {err[:300]}")

    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
