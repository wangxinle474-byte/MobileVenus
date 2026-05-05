"""把 100 对 IP2P 图片 + metadata 打包, 供 scp 到 AutoDL."""
import json
import re
import tarfile
from pathlib import Path


SRC_DIR = Path('outputs/ip2p_pilot_100')
OUT_TAR = Path('outputs/ip2p_pilot_100.tar.gz')
PROMPTS_JSON = Path('data/venus_edit_prompts.json')


def main():
    # 1. 构建 manifest: idx → {image, edit_prompt, suggestion_full}
    prompts_data = json.load(open(PROMPTS_JSON, 'r', encoding='utf-8'))
    prompts_by_idx = {i: p for i, p in enumerate(prompts_data['prompts'])}

    manifest = []
    orig_files = sorted(SRC_DIR.glob('*_orig.png'))
    for f in orig_files:
        m = re.match(r'(\d+)_orig\.png', f.name)
        if not m:
            continue
        idx = int(m.group(1))
        edit_f = SRC_DIR / f'{idx:04d}_edit.png'
        if not edit_f.exists():
            continue
        p = prompts_by_idx.get(idx, {})
        manifest.append({
            'idx': idx,
            'image': p.get('image', f'{idx:04d}.png'),
            'orig_png': f'{idx:04d}_orig.png',
            'edit_png': f'{idx:04d}_edit.png',
            'edit_prompt': p.get('edit_prompt', ''),
            'suggestion_full': p.get('suggestion_full', ''),
            'description_full': p.get('description_full', ''),
        })

    # 2. 保存 manifest
    manifest_path = SRC_DIR / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_pairs': len(manifest),
            'source': 'Benchmark_AesGuide',
            'editor': 'timbrooks/instruct-pix2pix (20 steps, img_g=1.5, prompt_g=7.5)',
            'pairs': manifest,
        }, f, ensure_ascii=False, indent=2)
    print(f'Manifest saved: {manifest_path} ({len(manifest)} pairs)')

    # 3. 打包
    print(f'Packing to {OUT_TAR}...')
    with tarfile.open(OUT_TAR, 'w:gz') as tar:
        tar.add(manifest_path, arcname='manifest.json')
        for item in manifest:
            orig = SRC_DIR / item['orig_png']
            edit = SRC_DIR / item['edit_png']
            tar.add(orig, arcname=item['orig_png'])
            tar.add(edit, arcname=item['edit_png'])

    size_mb = OUT_TAR.stat().st_size / 1024 / 1024
    print(f'Done: {OUT_TAR}  ({size_mb:.1f} MB)')
    print(f'\nNext: scp -P 35345 {OUT_TAR} root@connect.bjb2.seetacloud.com:/root/autodl-tmp/')


if __name__ == '__main__':
    main()
