"""从 fivek_venus_labels.json 筛 top N FiveK 候选作为 FireRed 1.1 API teacher edit 输入.

筛选 metric:
  1. venus_suggestion 含 tone 类关键词 (contrast/saturation/shadows/highlights/exposure/wb)
  2. 排除非 tone 建议为主的图 (cropping/composition/focus/sharpness/noise)
  3. 过滤过短 (<50 chars) 或过正面 ("flawless/perfect/excellent") 的 suggestion
  4. 含建议性 pattern ("could benefit"/"adjust"/"enhance"/"improve")
  5. 按 tone keyword category 数量 desc 排序, take top N

caption 改写 (rule-based, instruction-style):
  - 扫描 tone 类别 + 方向词 (lift/recover/warmer/cooler/etc)
  - 拼接 verb-led "Increase X and Y. Apply Z. Keep ... unchanged."

输出:
  data/teacher_edits_fivek_top<N>_instr.json (run_firered_online.py 兼容)

用法:
  python tools/data/data_prep/select_fivek_teacher_candidates.py --top 20
  python tools/data/data_prep/select_fivek_teacher_candidates.py --top 50 --exclude_used a0006-IMG_2787.jpg
"""
import argparse
import json
import re
from pathlib import Path

# 6 类 tone 关键词
TONE_KEYWORDS = {
    'contrast':    ['contrast'],
    'saturation':  ['saturation', 'saturated', 'vivid color', 'vivid colors', 'vibrancy', 'vibrance'],
    'shadows':     ['shadow', 'underexposed shadows', 'shadow detail'],
    'highlights':  ['highlight', 'overexposure', 'overexposed', 'blown highlights'],
    'exposure':    ['exposure', 'brightness'],
    'wb':          ['white balance', 'warmer', 'cooler', 'warmth', 'cool tone', 'warm tone',
                    'color temperature', 'tint'],
}

# 负向关键词 (说明建议不是 tone 调整为主)
NEGATIVE_KEYWORDS = [
    'cropping', 'composition could be improved', 'subject placement', 'framing',
    'sharper focus', 'depth of field', 'noise reduction', 'sharpness could',
    'sharp focus could', 'precision in focusing', 'centering', 're-framing',
]

# 建议性 pattern
SUGGESTION_PATTERNS = [
    'could benefit', 'would benefit', 'could be improved', 'enhance the',
    'adjustment', 'adjusting', 'could be elevated', 'could elevate',
    'slight increase', 'slight decrease', 'could be enhanced',
    'could be adjusted', 'recommend', 'room for improvement',
    'could be addressed', 'further enhance', 'could further',
]

# 过度正面 -> 没改进空间
POSITIVE_FILTER_KEYWORDS = [
    'flawless', 'perfect', 'excellent', 'skillfully', 'masterfully',
    'aesthetically pleasing portrayal', 'well-executed',
]


def score_caption(text: str) -> tuple[int, list[str]]:
    """统计 tone 类别命中数, 返回 (count, hit_categories)."""
    t = text.lower()
    hits = set()
    for cat, kws in TONE_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                hits.add(cat)
                break
    return len(hits), sorted(hits)


def has_negative(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in NEGATIVE_KEYWORDS)


def has_suggestion(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in SUGGESTION_PATTERNS)


def is_too_positive(text: str) -> bool:
    t = text.lower()
    n = sum(1 for kw in POSITIVE_FILTER_KEYWORDS if kw in t)
    return n >= 2


def rewrite_to_instruction(suggestion: str, hits: list[str]) -> str:
    """把 suggestion 规则化改写成 instruction-style verb-led caption."""
    actions = []
    t = suggestion.lower()

    if 'contrast' in hits:
        if any(x in t for x in ['reduce contrast', 'lower contrast', 'less contrast']):
            actions.append('reduce contrast')
        else:
            actions.append('increase contrast')

    if 'saturation' in hits:
        if any(x in t for x in ['oversaturated', 'reduce saturation', 'lower saturation', 'desaturate']):
            actions.append('decrease saturation')
        else:
            actions.append('enhance saturation')

    if 'shadows' in hits:
        if any(x in t for x in ['lift', 'underexposed shadow', 'shadow detail', 'open shadow', 'reveal shadow']):
            actions.append('lift shadows')
        elif 'deepen shadow' in t or 'darker shadow' in t:
            actions.append('deepen shadows')
        else:
            actions.append('lift shadows')

    if 'highlights' in hits:
        if any(x in t for x in ['recover', 'overexposure', 'overexposed', 'blown', 'bright highlight']):
            actions.append('recover highlights')
        else:
            actions.append('soften highlights')

    # exposure 跟 shadows 重叠时跳过
    if 'exposure' in hits and 'shadows' not in hits:
        if any(x in t for x in ['reduce exposure', 'overexposed', 'too bright']):
            actions.append('reduce exposure')
        elif any(x in t for x in ['increase exposure', 'underexposed', 'too dark', 'brighten']):
            actions.append('brighten exposure')

    if 'wb' in hits:
        if 'warm' in t and 'cool' not in t:
            actions.append('apply warmer white balance')
        elif 'cool' in t and 'warm' not in t:
            actions.append('apply cooler white balance')
        elif 'warm' in t and 'cool' in t:
            actions.append('neutralize white balance')
        else:
            actions.append('balance white balance')

    if not actions:
        return ''

    # 拼接
    if len(actions) == 1:
        verb_part = actions[0].capitalize()
    elif len(actions) == 2:
        verb_part = (actions[0] + ' and ' + actions[1]).capitalize()
    else:
        verb_part = (', '.join(actions[:-1]) + ', and ' + actions[-1]).capitalize()

    return verb_part + '. Keep the original composition and subject unchanged.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/fivek_venus_labels.json')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--min_score', type=int, default=2,
                    help='至少命中几个 tone 类别 (default 2)')
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--exclude_used', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_json = args.out_json or f'data/teacher_edits_fivek_top{args.top}_instr.json'

    with open(args.source, encoding='utf-8') as f:
        d = json.load(f)
    labels = d['labels']
    print(f'[INFO] loaded {len(labels)} fivek venus labels')

    excluded = set(args.exclude_used)
    if excluded:
        print(f'[EXCLUDE] {len(excluded)} images: {sorted(excluded)[:5]}...')

    # 过滤
    n_too_short, n_too_positive, n_no_suggestion, n_negative, n_low_score = 0, 0, 0, 0, 0
    candidates = []
    for lbl in labels:
        suggestion = lbl.get('raw_response', '')
        if not suggestion or len(suggestion) < 50:
            n_too_short += 1; continue
        if is_too_positive(suggestion):
            n_too_positive += 1; continue
        if not has_suggestion(suggestion):
            n_no_suggestion += 1; continue
        if has_negative(suggestion):
            n_negative += 1; continue
        if lbl['image'] in excluded:
            continue
        score, hits = score_caption(suggestion)
        if score < args.min_score:
            n_low_score += 1; continue
        caption = rewrite_to_instruction(suggestion, hits)
        if not caption:
            continue
        candidates.append({
            'image': lbl['image'],
            'raw_response': suggestion,
            'score': score,
            'hits': hits,
            'caption': caption,
        })

    print(f'[FILTER] kept {len(candidates)} / {len(labels)} candidates')
    print(f'         too_short={n_too_short}, too_positive={n_too_positive}, '
          f'no_suggestion={n_no_suggestion}, has_negative={n_negative}, '
          f'low_score={n_low_score}')

    if not candidates:
        print('[ERR] no candidates passed filters'); return 1

    # sort by score desc, tie-break by image name (deterministic)
    candidates.sort(key=lambda x: (-x['score'], x['image']))
    top = candidates[:args.top]

    print(f'\n[TOP {args.top}]')
    samples = []
    for rank, c in enumerate(top, start=1):
        m = re.match(r'a0*(\d+)-', c['image'])
        idx = int(m.group(1)) if m else rank
        samples.append({
            'rank': rank,
            'idx': idx,
            'source_image': c['image'],
            'orig_path': f'E:/Data/dataset/fivek_jpeg/{c["image"]}',
            'venus_suggestion': c['raw_response'][:300],
            'new_caption': c['caption'],
            'tone_target': '+'.join(c['hits']),
            'score': c['score'],
        })
        cap_preview = c['caption'][:75] + ('...' if len(c['caption']) > 75 else '')
        print(f'  {rank:>2}  idx={idx:<5} score={c["score"]} hits={c["hits"]}')
        print(f'      {c["image"]}')
        print(f'      cap: {cap_preview}')

    out = {
        'metadata': {
            'purpose': f'FiveK teacher edit candidates top {args.top}',
            'source': args.source,
            'n_samples': len(samples),
            'caption_style': 'instruction-style verb-led + Keep ... unchanged anchor',
            'selector_version': 'v1 rule-based',
            'min_score': args.min_score,
            'excluded_images': sorted(excluded),
        },
        'samples': samples,
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n[SAVED] {out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
