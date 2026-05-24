"""Check AutoDL data layout to plan v2 jsonl path rewrite + sync."""
import os
import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"


def run(c, cmd, t=30):
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip()


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)

    print("=== Search for fivek dataset on AutoDL ===")
    print(run(c, "find /root/autodl-tmp -maxdepth 4 -type d -name 'fivek*' 2>/dev/null"))
    print()

    print("=== Search for sample fivek jpgs (a0005, a0939) ===")
    print("a0005:")
    print(run(c, "find /root/autodl-tmp -maxdepth 5 -name 'a0005*' 2>/dev/null | head -3"))
    print("a0939:")
    print(run(c, "find /root/autodl-tmp -maxdepth 5 -name 'a0939*' 2>/dev/null | head -3"))
    print()

    print(f"=== {REPO}/outputs/teacher_edits/ subdirs ===")
    print(run(c, f"ls -la {REPO}/outputs/teacher_edits/ 2>/dev/null"))
    print()

    print("=== fivek_per_action PNG counts on AutoDL ===")
    for action in ["contrast", "saturation", "shadows", "highlights", "wb"]:
        out = run(c, f"ls {REPO}/outputs/teacher_edits/fivek_per_action/{action}/*.png 2>/dev/null | wc -l")
        print(f"  {action:12s}: {out}")
    print()

    print("=== existing master jsonls on AutoDL ===")
    print(run(c, f"find {REPO}/outputs/inverse_fit_pilot -maxdepth 2 -name pseudo_labels.jsonl 2>/dev/null"))
    print()

    print("=== fivek_500_master 1st record (path format used in previous training) ===")
    out = run(c, f"head -1 {REPO}/outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl 2>/dev/null")
    print(out[:600])
    print()

    print("=== disk usage breakdown ===")
    print(run(c, "df -h /root/autodl-tmp | tail -1"))
    for d in ["/root/autodl-tmp/data", f"{REPO}/data",
              f"{REPO}/outputs/teacher_edits", f"{REPO}/outputs/inverse_fit_pilot"]:
        out = run(c, f"du -sh {d} 2>/dev/null")
        if out:
            print(f"  {out}")

    print()
    print("=== specific fivek_jpeg path test ===")
    # Test if there is a fivek_jpeg dir with a0005-jn_2007_05_10__564.jpg
    for path in [
        "/root/autodl-tmp/data/fivek_jpeg",
        f"{REPO}/data/fivek_jpeg",
        "/root/autodl-tmp/fivek_jpeg",
    ]:
        out = run(c, f"ls {path}/a0005* 2>/dev/null | head -1")
        if out:
            print(f"  FOUND at {path}: {out}")

    c.close()


if __name__ == "__main__":
    main()
