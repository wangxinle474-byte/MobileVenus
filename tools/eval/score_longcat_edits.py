"""\u7528 Qwen3-VL-4B-Instruct \u5bf9 LongCat \u7f16\u8f91\u7ed3\u679c\u6253\u5206 (orig vs edited \u591a\u56fe\u8f93\u5165).

\u6253\u5206\u89c4\u5219 (4 \u7ef4\u5ea6 + Overall + Reason):
  1. instruction_follow [1-10]: \u6307\u4ee4\u9075\u5faa\u5ea6 (\u662f\u5426\u5b9e\u73b0 caption \u8981\u6c42\u7684\u4fee\u6539)
  2. aesthetic         [1-10]: \u7f8e\u5b66\u8d28\u91cf (\u5149\u5f71/\u8272\u5f69/\u6784\u56fe)
  3. identity_preserve [1-10]: \u4e3b\u4f53\u4fdd\u7559 (\u4e0e\u539f\u56fe\u4e3b\u4f53\u4e00\u81f4\u6027)
  4. realism           [1-10]: \u771f\u5b9e\u611f (\u662f\u5426\u4f2a\u5f71/\u5931\u771f)
  5. overall           [1-10]: 4 \u9879\u5747\u503c\u56db\u820d\u4e94\u5165
  6. reason            str:   \u4e2d\u6587 1-2 \u53e5\u8bdd\u8bf4\u660e

\u7528\u6cd5 (AutoDL):
  python tools/eval/score_longcat_edits.py \\
    --input_dir outputs/longcat_compare_sceneA \\
    --captions data/compare_5_captions.json \\
    --out outputs/longcat_score_sceneA.json \\
    --group_label sceneA
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = '/root/autodl-tmp/models/Qwen3-VL-4B-Instruct'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCORE_RE = re.compile(r'<score>\s*(\{.*?\})\s*</score>', re.DOTALL)
NUM_DIMS = ['instruction_follow', 'aesthetic', 'identity_preserve', 'realism', 'overall']


SYSTEM_PROMPT = """You are an expert image-editing quality evaluator. \
Given an original image, an edited image, and the edit instruction, \
you must rate the edit on 4 dimensions (1=fail, 10=perfect), then output \
strict JSON wrapped in <score>...</score> tags."""


def build_user_prompt(caption: str) -> str:
    return f"""I will show you two images:
[Image 1] Original photo (before editing)
[Image 2] Edited photo (after applying the instruction below)

Edit instruction: "{caption}"

Score the edit on each dimension (integer 1-10, where 10=perfect, 8-9=excellent, 6-7=good, 4-5=mediocre, 2-3=poor, 1=failure):
- instruction_follow: Did the edit faithfully implement the instruction?
- aesthetic: Is the edited image visually pleasing (light/color/composition)?
- identity_preserve: Does the edited image preserve the original subject/scene?
- realism: Are there any artifacts, distortions, or unrealistic elements? (10=clean, 1=heavy artifacts)

Use the FULL 1-10 range; avoid clustering all scores at the top.

Then compute overall = round(mean of the 4 dimensions) and give a 1-2 sentence \u4e2d\u6587 reason.

Output STRICTLY as one JSON object inside <score>...</score>, no extra text. \
Example FORMAT only (do NOT copy these values; rate based on the actual images):
<score>
{{"instruction_follow": 0, "aesthetic": 0, "identity_preserve": 0, "realism": 0, "overall": 0, "reason": "(\u4e00\u4e24\u53e5\u4e2d\u6587\u8bf4\u660e\u4f60\u7684\u8bc4\u5206\u4f9d\u636e)"}}
</score>"""


def parse_score(text: str) -> dict:
    """\u89e3\u6790 <score>{...}</score>, \u8fd4\u56de\u5305\u542b parse \u72b6\u6001\u7684 dict."""
    res = {
        'raw': text,
        'parsed_ok': False,
        'parse_issue': None,
        'scores': None,
    }
    m = SCORE_RE.search(text)
    if not m:
        res['parse_issue'] = 'no_score_block'
        return res
    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        res['parse_issue'] = f'invalid_json:{e.msg}'
        return res

    missing = [k for k in NUM_DIMS if k not in obj]
    if missing:
        res['parse_issue'] = f'missing_keys:{missing}'
        return res

    bad_vals = []
    for k in NUM_DIMS:
        v = obj.get(k)
        if not isinstance(v, (int, float)) or not (1 <= v <= 10):
            bad_vals.append(f'{k}={v}')
    if bad_vals:
        res['parse_issue'] = f'bad_vals:{bad_vals}'
        return res

    if not isinstance(obj.get('reason', ''), str):
        res['parse_issue'] = 'reason_not_str'
        return res

    res['scores'] = obj
    res['parsed_ok'] = True
    return res


def score_one(model, processor, orig_img, edit_img, orig_path, edit_path, caption, max_new_tokens=400):
    """\u5bf9\u4e00\u5bf9 (orig, edit) \u8c03\u7528\u4e00\u6b21 Qwen3-VL \u6253\u5206.

    \u5fc5\u987b\u540c\u65f6\u5728 message dict \u4e2d\u6307\u5b9a 'image': path/PIL,
    chat_template \u624d\u4f1a\u63d2\u5165\u56fe\u50cf\u5360\u4f4d\u7b26 token.
    \u53ea\u4f20 images=[...] \u4f1a\u88ab\u5ffd\u7565, \u5bfc\u81f4\u6a21\u578b\u770b\u4e0d\u5230\u56fe (\u4ee5\u524d 10 \u4e2a\u6837\u672c\u5206\u6570\u4e00\u6837\u5c31\u662f\u8fd9\u4e2a bug).
    """
    user_text = build_user_prompt(caption)
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
        {'role': 'user', 'content': [
            {'type': 'image', 'image': str(orig_path)},
            {'type': 'image', 'image': str(edit_path)},
            {'type': 'text', 'text': user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        images=[orig_img, edit_img],
        return_tensors='pt',
    ).to(model.device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    out_ids = gen[0, inputs['input_ids'].shape[-1]:]
    out_text = processor.tokenizer.decode(out_ids, skip_special_tokens=False).strip()
    return parse_score(out_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='\u7f16\u8f91\u56fe\u76ee\u5f55 (\u65b0\u5e03\u5c40: <input_dir>/<idx>.png; \u65e7: <input_dir>/<idx>_<suffix>.png)')
    ap.add_argument('--originals_dir', default=None,
                    help='\u539f\u56fe\u76ee\u5f55 (\u65b0\u5e03\u5c40: <originals_dir>/<idx>.png). \u4e0d\u4f20\u5c31\u4ece input_dir/<idx>_orig.png \u8bfb (\u65e7\u5e03\u5c40 fallback)')
    ap.add_argument('--captions', required=True,
                    help='compare_5_captions(_edit).json')
    ap.add_argument('--out', required=True, help='\u8f93\u51fa JSON \u8def\u5f84')
    ap.add_argument('--group_label', default='unknown',
                    help='\u7ec4\u540d (sceneA / editB / firered), \u4ec5\u7528\u4e8e meta)')
    ap.add_argument('--edit_suffix', default='longcat',
                    help='\u65e7\u5e03\u5c40 fallback \u7528: \u7f16\u8f91\u56fe\u540e\u7f00 (longcat / firered)')
    ap.add_argument('--base_model', default=BASE_MODEL)
    ap.add_argument('--max_new_tokens', type=int, default=400)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    originals_dir = None
    if args.originals_dir:
        originals_dir = Path(args.originals_dir)
        if not originals_dir.is_absolute():
            originals_dir = PROJECT_ROOT / originals_dir
    captions_path = Path(args.captions)
    if not captions_path.is_absolute():
        captions_path = PROJECT_ROOT / captions_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print(f'[load] processor + model: {args.base_model}')
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map='cuda',
        trust_remote_code=True,
    )
    model.eval()

    print(f'[load] captions: {captions_path}')
    cfg = json.load(open(captions_path, encoding='utf-8'))
    samples = cfg['samples']
    print(f'[run] {len(samples)} pairs from {input_dir}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        idx = s['idx']
        caption = s['new_caption']
        # 原图: 优先 originals_dir/<idx>.png, fallback input_dir/<idx>_orig.png
        if originals_dir is not None:
            orig_p = originals_dir / f'{idx:04d}.png'
        else:
            orig_p = input_dir / f'{idx:04d}_orig.png'
        # 编辑图: 优先 input_dir/<idx>.png (新), fallback input_dir/<idx>_<suffix>.png (旧)
        edit_p = input_dir / f'{idx:04d}.png'
        if not edit_p.exists():
            edit_p = input_dir / f'{idx:04d}_{args.edit_suffix}.png'
        if not orig_p.exists() or not edit_p.exists():
            print(f'[{i+1}/{len(samples)}] SKIP idx={idx}: missing image '
                  f'(orig={orig_p}, edit={edit_p})')
            continue

        orig_img = Image.open(orig_p).convert('RGB')
        edit_img = Image.open(edit_p).convert('RGB')

        try:
            parsed = score_one(model, processor, orig_img, edit_img,
                               orig_p, edit_p, caption,
                               max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f'[{i+1}/{len(samples)}] FAIL idx={idx}: {type(e).__name__}: {e}')
            parsed = {'raw': str(e), 'parsed_ok': False,
                      'parse_issue': f'exception:{type(e).__name__}', 'scores': None}

        results.append({
            'idx': idx,
            'source_image': s.get('source_image'),
            'caption': caption,
            'orig_path': str(orig_p),
            'edit_path': str(edit_p),
            'pred': parsed,
        })

        ok = '\u2713' if parsed['parsed_ok'] else '\u2717'
        sc = parsed.get('scores') or {}
        print(f'[{i+1}/{len(samples)}] {ok} idx={idx}  '
              f'IF={sc.get("instruction_follow", "?")}  '
              f'Aes={sc.get("aesthetic", "?")}  '
              f'IdP={sc.get("identity_preserve", "?")}  '
              f'Real={sc.get("realism", "?")}  '
              f'Overall={sc.get("overall", "?")}  '
              f'issue={parsed["parse_issue"]}')
        if parsed['parsed_ok'] and sc.get('reason'):
            print(f'         reason: {sc["reason"]}')

    dt = time.time() - t0
    n_ok = sum(1 for r in results if r['pred']['parsed_ok'])
    # \u805a\u5408\u5747\u503c
    means = {}
    if n_ok > 0:
        for k in NUM_DIMS:
            vals = [r['pred']['scores'][k] for r in results if r['pred']['parsed_ok']]
            means[k] = round(sum(vals) / len(vals), 2)

    print(f'\n[DONE] {n_ok}/{len(results)} parsed_ok in {dt:.1f}s')
    print(f'[MEAN] group={args.group_label}  ' + '  '.join(
        f'{k}={v}' for k, v in means.items()))

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'group_label': args.group_label,
                'input_dir': str(input_dir),
                'captions': str(captions_path),
                'base_model': args.base_model,
                'n_total': len(results),
                'n_parsed_ok': n_ok,
                'parse_rate': n_ok / max(1, len(results)),
                'runtime_sec': dt,
                'means': means,
            },
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f'[OUT] {out_path}')


if __name__ == '__main__':
    main()
