"""One-shot: verify imports on AutoDL then re-launch both evals."""
import os, sys, time
import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"
PY = "/root/miniconda3/bin/python"


def run(c, cmd, t=30):
    _, o, e = c.exec_command(cmd, timeout=t)
    out = o.read().decode(errors="ignore").strip()
    err = e.read().decode(errors="ignore").strip()
    rc = o.channel.recv_exit_status()
    return rc, out, err


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)

    # Quick import check
    check_cmd = (
        f'{PY} -c "import sys; sys.path.insert(0, \'{REPO}\'); '
        f'from tools._lib.merge_jsonl_for_joint_training import extract_fivek_stem; '
        f'from training.firered_baseline.train_lut import ACTIONS, LUTDataset, '
        f'NamedCurvesPredictor, build_data; print(\'ALL IMPORTS OK\')"'
    )
    rc, out, err = run(c, check_cmd, t=30)
    print(f"import check: rc={rc}")
    print(f"  stdout: {out}")
    if err:
        print(f"  stderr: {err[:500]}")
    if rc != 0:
        print("IMPORT FAILED - cannot launch eval. Fix missing deps first.")
        c.close()
        sys.exit(1)

    # Clear old logs
    run(c, f"rm -f {REPO}/outputs/eval_track2/v2/eval.log "
           f"{REPO}/outputs/eval_track2/baseline/eval.log")

    # Launch v2 eval (fire-and-forget, don't read stdout)
    cmd_v2 = (
        f"cd {REPO} && mkdir -p outputs/eval_track2/v2 && "
        f"nohup {PY} -u tools/eval_track2.py "
        f"--ckpt checkpoints/lut_v11a_pathX_v2_implicit_head/best.pt "
        f"--out_dir outputs/eval_track2/v2 "
        f"--batch_size 4 "
        f"--skip clean_expertC mmart_real_lr "
        f"> outputs/eval_track2/v2/eval.log 2>&1 &"
    )
    c.exec_command(cmd_v2, timeout=10)
    print("v2 eval launched (fire-and-forget)")
    time.sleep(3)

    # Launch baseline eval (fire-and-forget)
    cmd_b = (
        f"cd {REPO} && mkdir -p outputs/eval_track2/baseline && "
        f"nohup {PY} -u tools/eval_track2.py "
        f"--ckpt checkpoints/lut_v11a_action_gated_context/best.pt "
        f"--out_dir outputs/eval_track2/baseline "
        f"--batch_size 4 "
        f"--skip clean_expertC mmart_real_lr "
        f"> outputs/eval_track2/baseline/eval.log 2>&1 &"
    )
    c.exec_command(cmd_b, timeout=10)
    print("baseline eval launched (fire-and-forget)")
    time.sleep(5)

    # Verify
    rc, procs, _ = run(c, "pgrep -af eval_track2")
    print(f"\nrunning processes:\n{procs or '(none)'}")

    if not procs:
        print("\nWARN: no eval processes found. Checking logs...")
        for label in ("v2", "baseline"):
            rc, log, _ = run(c, f"tail -10 {REPO}/outputs/eval_track2/{label}/eval.log")
            print(f"--- {label} ---")
            print(log)
    else:
        print("\nBoth evals running. Use autodl_tail_pathX.py or check later.")

    c.close()


if __name__ == "__main__":
    main()
