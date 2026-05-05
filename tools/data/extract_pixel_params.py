"""像素层面从 (orig, edit) 图对推出 Lightroom-style 参数。

比 VLM 反推更可靠 - 直接基于 RGB 统计差异推算:
  ev_compensation   ← log2(mean_luma_ratio), [-3, 3]
  white_balance_K   ← R/B 比值的 color-temp 估计, [2000, 10000]
  contrast          ← std(luma) 变化百分比, [-100, 100]
  shadows           ← 暗部 (luma<0.25) 均值变化, [-100, 100]
  highlights        ← 亮部 (luma>0.75) 均值变化, [-100, 100]
  saturation        ← mean chroma (HSV 的 S) 变化百分比, [-100, 100]

输出: outputs/aug_ip2p_pilot_with_params.json
每对增加 pixel_params 字段 + delta_score 已有。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_img(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def rgb_to_luma(arr):
    # Rec. 709 luma
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def rgb_to_hsv_s(arr):
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    s = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return s


def estimate_color_temp(arr):
    """粗略估计色温 (K). McCamy 近似, 仅用 mean RGB."""
    r = arr[..., 0].mean()
    g = arr[..., 1].mean() + 1e-6
    b = arr[..., 2].mean() + 1e-6
    # R/B 比 > 1 偏暖 (低色温); 比 < 1 偏冷 (高色温)
    rb = r / b
    # 经验映射: rb ≈ 1.0 -> 6500K; rb > 1 -> 更低 (暖); rb < 1 -> 更高 (冷)
    # 使用对数映射
    temp = 6500 * (1.0 / rb) ** 0.8
    return float(np.clip(temp, 2000, 10000))


def extract_params(orig_path, edit_path):
    o = load_img(orig_path)
    e = load_img(edit_path)

    lum_o = rgb_to_luma(o)
    lum_e = rgb_to_luma(e)

    # 1. ev_compensation: log2(mean_e / mean_o), clamp [-3, 3]
    eps = 1e-3
    ratio = (lum_e.mean() + eps) / (lum_o.mean() + eps)
    ev = math.log2(max(ratio, 1e-3))
    ev = float(np.clip(ev, -3, 3))

    # 2. white_balance: 估算目标色温差
    temp_o = estimate_color_temp(o)
    temp_e = estimate_color_temp(e)
    wb_k = temp_e  # 目标色温

    # 3. contrast: std(luma) 变化百分比
    std_o = float(lum_o.std() + eps)
    std_e = float(lum_e.std() + eps)
    contrast = (std_e / std_o - 1.0) * 100.0
    contrast = float(np.clip(contrast, -100, 100))

    # 4. shadows: 暗部均值变化 (lum < 0.25)
    shadow_mask_o = lum_o < 0.25
    shadow_mask_e = lum_e < 0.25
    # 用 orig 的 mask 测在 orig 和 edit 中的对应区域
    if shadow_mask_o.sum() > 100:
        shadow_o = lum_o[shadow_mask_o].mean()
        shadow_e = lum_e[shadow_mask_o].mean()  # 同一块区域在 edit 里的亮度
        shadows = (shadow_e - shadow_o) * 400.0  # 0.25 * 400 = 100
        shadows = float(np.clip(shadows, -100, 100))
    else:
        shadows = 0.0

    # 5. highlights: 亮部均值变化 (lum > 0.75)
    hi_mask_o = lum_o > 0.75
    if hi_mask_o.sum() > 100:
        hi_o = lum_o[hi_mask_o].mean()
        hi_e = lum_e[hi_mask_o].mean()
        highlights = (hi_e - hi_o) * 400.0
        highlights = float(np.clip(highlights, -100, 100))
    else:
        highlights = 0.0

    # 6. saturation: mean HSV-S 变化百分比
    s_o = float(rgb_to_hsv_s(o).mean() + eps)
    s_e = float(rgb_to_hsv_s(e).mean() + eps)
    saturation = (s_e / s_o - 1.0) * 100.0
    saturation = float(np.clip(saturation, -100, 100))

    return {
        'ev_compensation': round(ev, 3),
        'white_balance': round(wb_k, 0),
        'contrast': round(contrast, 1),
        'shadows': round(shadows, 1),
        'highlights': round(highlights, 1),
        'saturation': round(saturation, 1),
        # debug 字段
        '_debug_mean_luma_orig': round(float(lum_o.mean()), 3),
        '_debug_mean_luma_edit': round(float(lum_e.mean()), 3),
        '_debug_temp_orig': round(temp_o, 0),
        '_debug_temp_edit': round(temp_e, 0),
        '_debug_saturation_orig': round(s_o, 3),
        '_debug_saturation_edit': round(s_e, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/aug_ip2p_pilot_v1_reparsed.json')
    parser.add_argument('--image_dir', default='outputs/ip2p_pilot_100')
    parser.add_argument('--output', default='outputs/aug_ip2p_pilot_with_params.json')
    args = parser.parse_args()

    d = json.load(open(args.input, encoding='utf-8'))
    img_dir = Path(args.image_dir)
    rs = d['results']

    n_ok = 0
    for r in rs:
        orig = img_dir / f'{r["idx"]:04d}_orig.png'
        edit = img_dir / f'{r["idx"]:04d}_edit.png'
        if not orig.exists() or not edit.exists():
            r['pixel_params'] = None
            continue
        try:
            r['pixel_params'] = extract_params(orig, edit)
            n_ok += 1
        except Exception as exc:
            r['pixel_params'] = {'error': str(exc)}

    # 简单统计
    print(f'Processed: {n_ok}/{len(rs)}')

    # Top improved (按 delta_v2) 的参数分布
    improved = sorted(
        [r for r in rs if r.get('delta_v2') is not None and r['delta_v2'] >= 0.5],
        key=lambda x: -x['delta_v2'])
    print(f'\n=== {len(improved)} improved pairs - pixel params ===')
    print(f'{"idx":>4}  {"img":20s}  {"delta":>6s}  {"ev":>6s}  {"wb":>5s}  '
          f'{"cont":>5s}  {"shad":>5s}  {"hi":>5s}  {"sat":>5s}')
    for r in improved[:20]:
        p = r.get('pixel_params') or {}
        print(f'{r["idx"]:>4}  {r["image"]:20s}  '
              f'{r["delta_v2"]:+6.2f}  '
              f'{p.get("ev_compensation", "-"):>6}  '
              f'{p.get("white_balance", "-"):>5}  '
              f'{p.get("contrast", "-"):>5}  '
              f'{p.get("shadows", "-"):>5}  '
              f'{p.get("highlights", "-"):>5}  '
              f'{p.get("saturation", "-"):>5}')

    # 均值统计
    if improved:
        print('\n=== 改善对的像素参数均值 ===')
        keys = ['ev_compensation', 'white_balance', 'contrast',
                'shadows', 'highlights', 'saturation']
        for k in keys:
            vals = [r['pixel_params'][k] for r in improved
                    if r.get('pixel_params') and k in r['pixel_params']]
            if vals:
                print(f'  {k:20s}  mean={sum(vals) / len(vals):+7.2f}  '
                      f'min={min(vals):+7.1f}  max={max(vals):+7.1f}')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
