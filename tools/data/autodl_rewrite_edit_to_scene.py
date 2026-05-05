"""\u7528 Qwen3-VL-4B-Instruct \u628a\u201c\u7f16\u8f91\u6307\u4ee4\u201d\u91cd\u5199\u4e3a\u201c\u5b8c\u6574\u573a\u666f\u63cf\u8ff0\u201d.

\u8f93\u5165: data/compare_5_captions_edit.json (\u52a8\u8bcd\u5f0f\u6307\u4ee4)
\u8f93\u51fa: data/compare_5_captions_edit_rewritten.json (\u63cf\u8ff0\u5f0f\u6700\u7ec8\u573a\u666f)

\u8bbe\u8ba1\u6c14\u8def: LongCat-Turbo \u5403\u201c\u63cf\u8ff0\u6700\u7ec8\u753b\u9762\u201d\u6bd4\u201c\u4e0b\u6307\u4ee4\u201d\u6548\u679c\u597d
(sceneA 9.2 vs editB 8.8). \u8ba9 Qwen3-VL \u770b\u539f\u56fe + \u539f\u59cb\u6307\u4ee4, \u751f\u6210\u4e00\u53e5\u9c9c\u6d3b\u63cf\u8ff0,
\u7ed3\u5408\u539f\u573a\u666f + \u6307\u4ee4\u8981\u6c42, \u6253\u5305\u4e3a "scene-style prompt" \u9001\u7ed9 LongCat\u3002

\u8fd0\u884c (AutoDL):
  python tools/data/autodl_rewrite_edit_to_scene.py \\
    --captions_in data/compare_5_captions_edit.json \\
    --out data/compare_5_captions_edit_rewritten.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = '/root/autodl-tmp/models/Qwen3-VL-4B-Instruct'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


SYSTEM_PROMPT = """You are a professional photography prompt engineer. \
You translate user-given edit instructions into rich, descriptive captions \
suitable for a text-conditioned image-editing diffusion model. \
You ALWAYS describe the final edited scene as if it already exists, \
never use imperative verbs like 'apply', 'increase', 'add'."""


USER_TEMPLATE = """I will show you the ORIGINAL photo. The user wants to edit it with this instruction:

"{instruction}"

Your task: Write ONE rich English caption (40-70 words) describing what the FINAL edited photo should look like.

Strict rules:
- Describe the concrete subject and setting that you actually see in the original (preserve existing content faithfully).
- Incorporate the requested edits as final-state visual attributes (lighting, color, mood, contrast, composition).
- Descriptive style, NOT imperative. Write "the mountains glow with warm golden-hour light", NOT "apply warm light to mountains".
- Single paragraph. 40-70 words.
- Keep specific aesthetic qualifiers if present in the instruction (golden hour, dramatic shadows, B&W, bokeh, etc.).
- Output ONLY the caption text. No quotes, no preamble like "Caption:", no explanation."""


@torch.no_grad()
def rewrite_one(model, processor, orig_img, orig_path, instruction, max_new_tokens=200):
    user_text = USER_TEMPLATE.format(instruction=instruction)
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
        {'role': 'user', 'content': [
            {'type': 'image', 'image': str(orig_path)},
            {'type': 'text', 'text': user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        images=[orig_img],
        return_tensors='pt',
    ).to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    out_ids = gen[0, inputs['input_ids'].shape[-1]:]
    out_text = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    # \u53bb\u6389\u53ef\u80fd\u7684\u5f15\u53f7\u5305\u88c5
    if out_text.startswith('"') and out_text.endswith('"'):
        out_text = out_text[1:-1].strip()
    if out_text.startswith('Caption:'):
        out_text = out_text[len('Caption:'):].strip()
    return out_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions_in', required=True,
                    help='\u8f93\u5165 captions JSON (\u542b\u52a8\u8bcd\u5f0f\u6307\u4ee4)')
    ap.add_argument('--out', required=True,
                    help='\u8f93\u51fa rewritten JSON \u8def\u5f84')
    ap.add_argument('--originals_dir', default='outputs/compare_5/originals',
                    help='\u539f\u56fe\u76ee\u5f55 (\u65b0\u5e03\u5c40 <idx>.png)')
    ap.add_argument('--input_dir', default=None,
                    help='[\u65e7\u5e03\u5c40 fallback] \u542b <idx>_orig.png \u7684\u76ee\u5f55')
    ap.add_argument('--base_model', default=BASE_MODEL)
    ap.add_argument('--max_new_tokens', type=int, default=200)
    args = ap.parse_args()

    cap_path = Path(args.captions_in)
    if not cap_path.is_absolute():
        cap_path = PROJECT_ROOT / cap_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    originals_dir = Path(args.originals_dir)
    if not originals_dir.is_absolute():
        originals_dir = PROJECT_ROOT / originals_dir
    input_dir = None
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = PROJECT_ROOT / input_dir

    print(f'[load] processor + model: {args.base_model}')
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map='cuda',
        trust_remote_code=True,
    )
    model.eval()

    cfg = json.load(open(cap_path, encoding='utf-8'))
    samples = cfg['samples']
    print(f'[run] {len(samples)} samples')

    t0 = time.time()
    new_samples = []
    for i, s in enumerate(samples):
        idx = s['idx']
        instruction = s['new_caption']

        # 找原图: 新布局 originals_dir/<idx>.png 优先, fallback 旧 input_dir/<idx>_orig.png, 再 fallback orig_path 字段
        cand_new = originals_dir / f'{idx:04d}.png'
        cand_old = (input_dir / f'{idx:04d}_orig.png') if input_dir is not None else None
        cand_field = PROJECT_ROOT / s.get('orig_path', '')
        if cand_new.exists():
            orig_p = cand_new
        elif cand_old is not None and cand_old.exists():
            orig_p = cand_old
        elif cand_field.exists():
            orig_p = cand_field
        else:
            print(f'[{i+1}/{len(samples)}] SKIP idx={idx}: no orig found '
                  f'(tried {cand_new}, {cand_old}, {cand_field})')
            new_samples.append({**s, 'rewritten_caption': None,
                                'rewrite_error': 'orig_image_not_found'})
            continue

        orig_img = Image.open(orig_p).convert('RGB')
        ti = time.time()
        try:
            rewritten = rewrite_one(model, processor, orig_img, orig_p,
                                    instruction, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f'[{i+1}/{len(samples)}] FAIL idx={idx}: {type(e).__name__}: {e}')
            new_samples.append({**s, 'rewritten_caption': None,
                                'rewrite_error': f'{type(e).__name__}:{e}'})
            continue
        dt = time.time() - ti

        # \u4fdd\u7559\u539f\u6709\u5b57\u6bb5, \u65b0\u589e rewritten + \u8986\u76d6 new_caption
        new_s = {
            **s,
            'original_caption': instruction,
            'rewritten_caption': rewritten,
            'new_caption': rewritten,  # \u8ba9 LongCat \u811a\u672c\u76f4\u63a5\u8bfb new_caption
            'rewrite_time_sec': round(dt, 1),
        }
        new_samples.append(new_s)
        print(f'[{i+1}/{len(samples)}] idx={idx}  ({dt:.1f}s)')
        print(f'  ORIG: {instruction}')
        print(f'  REWR: {rewritten}')

    total = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        'metadata': {
            **cfg.get('metadata', {}),
            'rewriter_model': args.base_model,
            'rewriter_system_prompt': SYSTEM_PROMPT,
            'rewriter_user_template': USER_TEMPLATE,
            'rewriter_runtime_sec': round(total, 1),
            'caption_style': 'rewritten scene description (from edit instruction)',
        },
        'samples': new_samples,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for s in new_samples if s.get('rewritten_caption'))
    print(f'\n[DONE] {n_ok}/{len(samples)} rewritten in {total:.1f}s')
    print(f'[OUT] {out_path}')


if __name__ == '__main__':
    main()
