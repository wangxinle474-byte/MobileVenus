"""
实时监控 AesExpert 重打分进度。

用法:
    python tools/data/monitor_rescore.py            # 实时刷新, Ctrl+C 退出
    python tools/data/monitor_rescore.py --once     # 只打印当前快照
    python tools/data/monitor_rescore.py --interval 5  # 每 5 秒刷新一次
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

# Windows UTF-8 输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'outputs' / 'data' / 'aesexpert_full.log'
JSON = ROOT / 'outputs' / 'data' / 'aug_aesexpert.json'
TOTAL = 22394

PROG_RE = re.compile(r'\[(\d+)/(\d+)\].*?speed=([\d.]+)/s\s+ETA=([\d.]+)min\s+last=([\d.]+)')


def _read_log(log_path: Path) -> str:
    """自动判定日志编码 (PowerShell 重定向默认 UTF-16 LE)."""
    raw = log_path.read_bytes()
    # BOM 嗅探
    if raw.startswith(b'\xff\xfe'):
        return raw.decode('utf-16-le', errors='replace')
    if raw.startswith(b'\xfe\xff'):
        return raw.decode('utf-16-be', errors='replace')
    if raw.startswith(b'\xef\xbb\xbf'):
        return raw[3:].decode('utf-8', errors='replace')
    # 启发式: 大量 \x00 → UTF-16 无 BOM
    if raw[:200].count(b'\x00') > 50:
        return raw.decode('utf-16-le', errors='replace')
    return raw.decode('utf-8', errors='replace')


def parse_progress(log_path: Path):
    """解析最近 N 行, 提取最后一条进度."""
    if not log_path.exists():
        return None
    text = _read_log(log_path)
    lines = text.splitlines()
    last = None
    for line in reversed(lines):
        m = PROG_RE.search(line)
        if m:
            last = {
                'cur': int(m.group(1)),
                'total': int(m.group(2)),
                'speed': float(m.group(3)),
                'eta_min': float(m.group(4)),
                'last_score': float(m.group(5)),
            }
            break
    # 统计最近 50 行的分数分布
    recent_scores = []
    for line in lines[-100:]:
        m = PROG_RE.search(line)
        if m:
            recent_scores.append(float(m.group(5)))
    last_ckpt = None
    for line in reversed(lines):
        if '检查点' in line or 'checkpoint' in line.lower():
            last_ckpt = line.strip()
            break
    return {
        'last': last,
        'recent_scores': recent_scores,
        'last_ckpt': last_ckpt,
        'log_size_kb': log_path.stat().st_size / 1024,
        'log_mtime': log_path.stat().st_mtime,
    }


def get_gpu_info():
    """通过 nvidia-smi 取一次 GPU 状态."""
    import subprocess
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(',')
            return {
                'mem_used': int(parts[0]),
                'mem_total': int(parts[1]),
                'util': int(parts[2]),
            }
    except Exception:
        pass
    return None


def render(snap, last_cur=None, last_t=None, clear=True):
    if not snap or not snap['last']:
        print('日志暂无进度行', flush=True)
        return None, None
    last = snap['last']
    cur, total = last['cur'], last['total']
    pct = cur / total * 100
    eta_h = last['eta_min'] / 60
    bar_w = 40
    filled = int(bar_w * cur / total)
    bar = '█' * filled + '░' * (bar_w - filled)

    now = time.time()
    inst_speed = None
    if last_cur is not None and last_t is not None:
        dt = now - last_t
        dn = cur - last_cur
        if dt > 0 and dn >= 0:
            inst_speed = dn / dt

    rs = snap['recent_scores']
    if rs:
        s_min, s_max = min(rs), max(rs)
        s_avg = sum(rs) / len(rs)
        n7 = sum(1 for s in rs if s >= 7.0)
        score_line = (f'近 {len(rs)} 张分数: avg={s_avg:.2f} '
                      f'min={s_min:.1f} max={s_max:.1f} ≥7.0={n7}({n7/len(rs)*100:.0f}%)')
    else:
        score_line = '近期分数: (无数据)'

    gpu = get_gpu_info()
    gpu_line = (f'GPU: {gpu["mem_used"]}/{gpu["mem_total"]} MiB  util={gpu["util"]}%'
                if gpu else 'GPU: (nvidia-smi 不可用)')

    log_age = time.time() - snap['log_mtime']

    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    print('=' * 60, flush=True)
    print(f'  AesExpert 全量重打分实时监控   ({time.strftime("%H:%M:%S")})', flush=True)
    print('=' * 60, flush=True)
    print(f'  进度: [{bar}] {pct:5.2f}%', flush=True)
    print(f'        {cur:>6} / {total}   ({total - cur} 待打分)', flush=True)
    print(flush=True)
    print(f'  整体速度: {last["speed"]:.2f} img/s', end='', flush=True)
    if inst_speed is not None:
        print(f'   实时: {inst_speed:.2f} img/s', end='', flush=True)
    print(flush=True)
    print(f'  ETA:     {last["eta_min"]:.1f} min  (~{eta_h:.2f} h)', flush=True)
    print(f'  上一张:  score={last["last_score"]}', flush=True)
    print(flush=True)
    print(f'  {score_line}', flush=True)
    print(f'  {gpu_line}', flush=True)
    if snap['last_ckpt']:
        print(f'  最近 checkpoint: {snap["last_ckpt"][-60:]}', flush=True)
    print(f'  日志: {snap["log_size_kb"]:.0f} KB, {log_age:.0f}s 前更新', flush=True)
    print('=' * 60, flush=True)
    return cur, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true', help='只打印一次快照')
    ap.add_argument('--interval', type=float, default=10.0, help='刷新间隔秒')
    args = ap.parse_args()

    if not LOG.exists():
        print(f'日志不存在: {LOG}')
        sys.exit(1)

    if args.once:
        snap = parse_progress(LOG)
        render(snap, clear=False)
        return

    last_cur, last_t = None, None
    try:
        while True:
            snap = parse_progress(LOG)
            last_cur, last_t = render(snap, last_cur, last_t)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n监控已退出')


if __name__ == '__main__':
    main()
