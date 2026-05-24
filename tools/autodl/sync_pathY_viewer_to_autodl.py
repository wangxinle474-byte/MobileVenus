"""Sync local Path Y wb PNGs to AutoDL + generate Jupyter Lab viewer.

Steps:
  1. SFTP upload outputs/teacher_edits/pathY_outputs/<action>/*.png
     → /root/autodl-tmp/pathY_outputs/<action>/
  2. Generate HTML viewer at /root/autodl-tmp/pathY_viewer.html with
     relative paths to fivek_jpeg/ and pathY_outputs/
  3. Print URL hint for Jupyter Lab access

Usage:
    $env:AUTODL_PASS = "..."
    python tools/sync_pathY_viewer_to_autodl.py [--actions wb] [--max 100]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"

REMOTE_BASE = "/root/autodl-tmp"
REMOTE_OUT = f"{REMOTE_BASE}/pathY_outputs"
REMOTE_VIEWER = f"{REMOTE_BASE}/pathY_viewer.html"

PROJECT = Path(__file__).resolve().parents[1]
LOCAL_OUT = PROJECT / "outputs" / "teacher_edits" / "pathY_outputs"
CAPTIONS_DIR = PROJECT / "data" / "pathY_captions"

ACTIONS = ["wb", "highlights", "saturation", "shadows", "contrast", "wb_cooler"]


def load_samples(action: str) -> dict[int, dict]:
    captions = CAPTIONS_DIR / f"pathY_{action}.json"
    if not captions.exists():
        return {}
    with open(captions, encoding="utf-8") as f:
        cfg = json.load(f)
    return {s["idx"]: s for s in cfg["samples"]}


def remote_exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except IOError:
        return False


def remote_listdir(sftp: paramiko.SFTPClient, path: str) -> set[str]:
    try:
        return set(sftp.listdir(path))
    except IOError:
        return set()


def upload_pngs(sftp: paramiko.SFTPClient, cli: paramiko.SSHClient,
                action: str, max_n: int) -> list[int]:
    """Returns list of idx values present on remote after upload."""
    local_dir = LOCAL_OUT / action
    if not local_dir.exists():
        return []

    remote_dir = f"{REMOTE_OUT}/{action}"
    cli.exec_command(f"mkdir -p {remote_dir}")
    remote_files = remote_listdir(sftp, remote_dir)

    local_pngs = sorted(local_dir.glob("*.png"))[:max_n]
    if not local_pngs:
        print(f"  [{action}] no local PNGs")
        return []

    to_upload = [p for p in local_pngs if p.name not in remote_files]
    print(f"  [{action}] local={len(local_pngs)} remote={len(remote_files & {p.name for p in local_pngs})} to_upload={len(to_upload)}")

    if to_upload:
        t0 = time.time()
        for i, p in enumerate(to_upload, 1):
            sftp.put(str(p), f"{remote_dir}/{p.name}")
            if i % 10 == 0 or i == len(to_upload):
                print(f"    [{i}/{len(to_upload)}] in {time.time()-t0:.1f}s")
    return [int(p.stem) for p in local_pngs]


def render_html(pairs: list[dict]) -> str:
    """Generate viewer HTML with paths relative to /root/autodl-tmp/."""
    style = """
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px; background: #1e1e1e; color: #ddd; }
    h1 { color: #fff; }
    .meta { color: #aaa; margin-bottom: 20px; }
    .row { display: flex; gap: 12px; margin-bottom: 24px; padding: 12px; background: #2a2a2a; border-radius: 8px; }
    .col { flex: 1; min-width: 0; }
    .col img { width: 100%; max-height: 600px; object-fit: contain; background: #000; border-radius: 4px; }
    .col .label { color: #888; font-size: 12px; margin-top: 4px; font-family: monospace; }
    .info { width: 320px; flex-shrink: 0; color: #ccc; font-size: 13px; line-height: 1.5; }
    .info .idx { color: #4fc3f7; font-weight: bold; font-size: 16px; }
    .info .action { color: #ff9800; font-family: monospace; }
    .info .caption { color: #b0bec5; margin-top: 8px; font-style: italic; }
    .stats { background: #333; padding: 12px; border-radius: 8px; margin-bottom: 16px; font-family: monospace; }
    """
    rows = []
    stats = {}
    for p in pairs:
        stats[p["action"]] = stats.get(p["action"], 0) + 1
        rows.append(f"""
        <div class="row">
          <div class="col">
            <img src="fivek_jpeg/{p['source_image']}" loading="lazy">
            <div class="label">orig: {p['source_image']}</div>
          </div>
          <div class="col">
            <img src="pathY_outputs/{p['action']}/{p['idx']:04d}.png" loading="lazy">
            <div class="label">edit: {p['action']}/{p['idx']:04d}.png</div>
          </div>
          <div class="info">
            <div class="idx">#{p['idx']}</div>
            <div class="action">action={p['action']}</div>
            <div class="caption">"{p['caption']}"</div>
          </div>
        </div>
        """)

    stats_str = "  |  ".join(f"<b>{a}</b>: {c}" for a, c in sorted(stats.items()))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Path Y Viewer (AutoDL)</title>
<style>{style}</style>
</head>
<body>
<h1>Path Y 生成结果对比 (FireRed-Image-Edit-1.1 Lightning)</h1>
<div class="stats">{stats_str} | total pairs: {len(pairs)}</div>
<div class="meta">左：FiveK 原图 | 右：FireRed 编辑后 | 路径相对于 /root/autodl-tmp/</div>
{''.join(rows)}
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default="wb,wb_cooler",
                    help="comma-separated, e.g. 'wb' or 'wb,wb_cooler'")
    ap.add_argument("--max", type=int, default=300,
                    help="max PNGs per action to upload")
    args = ap.parse_args()
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    pwd = os.environ.get("AUTODL_PASS")
    if not pwd:
        print("ERROR: set AUTODL_PASS env var first", file=sys.stderr)
        sys.exit(2)

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[CONNECT] {USER}@{HOST}:{PORT}")
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=20)
    sftp = cli.open_sftp()
    print("[CONNECTED]\n")

    # 1. Upload PNGs per action
    print("[1/2] Uploading PNGs ...")
    pairs = []
    for action in actions:
        idx_list = upload_pngs(sftp, cli, action, args.max)
        samples = load_samples(action)
        for idx in idx_list:
            s = samples.get(idx)
            if not s:
                continue
            pairs.append({
                "idx": idx,
                "action": action,
                "source_image": s["source_image"],
                "caption": s.get("new_caption", ""),
            })
    print(f"  total pairs: {len(pairs)}\n")

    # 2. Generate + upload viewer HTML
    print("[2/2] Generating + uploading viewer HTML ...")
    html = render_html(pairs)
    html_bytes = html.encode("utf-8")
    with sftp.open(REMOTE_VIEWER, "w") as f:
        f.write(html)
    # SFTP write doesn't always set size; verify
    stat = sftp.stat(REMOTE_VIEWER)
    print(f"  uploaded {REMOTE_VIEWER} ({stat.st_size} bytes)")

    sftp.close()
    cli.close()

    print(f"""
[DONE] Open in AutoDL Jupyter Lab:
  1. 访问 AutoDL 实例的 Jupyter Lab (通常通过 AutoDL 控制台的 'JupyterLab' 按钮)
  2. 在文件树中导航到: /root/autodl-tmp/
  3. 双击 pathY_viewer.html
     - Jupyter Lab 会在新标签页中以 HTML preview 模式打开
     - 如果只看到代码, 右键 → 'Open With' → 'HTML File'
  4. 或直接通过 Jupyter Lab 的 URL 访问:
     http://<jupyter-host>/files/pathY_viewer.html
""")


if __name__ == "__main__":
    main()
