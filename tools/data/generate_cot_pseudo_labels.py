"""\u7528 JarvisEvo-8B (vLLM) \u4e3a ArtEdit-Bench-Lr \u7684 800 \u6837\u672c\u751f\u6210 CoT pseudo \u6807\u7b7e\u3002

\u903b\u8f91:
  \u5bf9\u6bcf\u4e2a\u6837\u672c (input.jpg, user_want.txt):
    1. \u7ec4\u88c5 system + user \u6d88\u606f (\u6309 JarvisEvo SYSTEM_PROMPT)
    2. \u53d1\u9001\u5230\u672c\u5730 vLLM \u670d\u52a1 (OpenAI \u517c\u5bb9 API)
    3. \u63a5\u6536 round-1 \u8f93\u51fa <think>...</think> + <tool_call>...</tool_call>
    4. \u4fdd\u5b58\u4e3a JSONL

\u4f7f\u7528\u573a\u666f: \u6211\u4eec\u4e0d\u8dd1\u591a\u8f6e\u3001\u4e0d\u8c03 Adobe LR, \u4ec5\u8981 round-1 \u6587\u672c \u505a SFT \u6807\u7b7e\u3002

\u524d\u63d0: vLLM \u670d\u52a1\u5df2\u5728 localhost:8086 \u542f\u52a8
  VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /root/autodl-tmp/checkpoints/pretrained/JarvisEvo \\
    --tensor-parallel-size 1 --port 8086 --api-key 0 --served-model-name jarvisevo \\
    --max_model_len 20480 --limit-mm-per-prompt.image 5 --gpu-memory-utilization 0.90

\u8f93\u51fa: pseudo_labels.jsonl  \u6bcf\u884c = {"image_dir", "prompt", "response", "think", "tool_call"}
"""
import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# SYSTEM_PROMPT v3: \u9488\u5bf9 smoke v1 \u53d1\u73b0\u7684\u95ee\u9898 (JSON \u5355\u5f15/\u7f3a\u503c/\u5d4c\u5957) \u4e25\u683c\u7ea6\u675f
# v1 bug: model \u8f93\u51fa<tool_call>block</tool_call> (\u5360\u4f4d\u7b26\u9677\u9631)
# v2 bug: model \u8f93\u51fa\u590d\u6742\u5d4c\u5957 + \u5355\u5f15\u53f7 'Custom' + tone curve {1:0} \u975e\u6cd5 JSON
# v3: \u5f3a\u5236 flat \u7eaf\u6570\u503c dict + 2 \u4e2a\u5b8c\u6574\u4f8b\u5b50 + \u660e\u4ee4 ban \u590d\u6742\u7ed3\u6784
ALLOWED_LR_KEYS_DOC = """temp, tint, exposure, contrast, highlights, shadows, whites, blacks, clarity, vibrance, saturation, texture, dehaze"""

SYSTEM_PROMPT_SHORT = """You are a Lightroom adjustment specialist. Given an input image and user instruction, you output 4-8 Lightroom slider values.

# OUTPUT FORMAT (no other text allowed)

<think>
[2-3 sentences in user's language: state photo's current look and what 4-8 sliders to move.]
</think>
<tool_call>{"temp": <int>, "exposure": <float>, "contrast": <int>, ...}</tool_call>

# CONCRETE EXAMPLES (copy this style EXACTLY)

## Example 1 (English)
User: "Make this landscape more dramatic with deeper shadows and warmer highlights."
<think>The landscape has flat lighting. I will warm the white balance, deepen shadows for drama, and add contrast and clarity to bring out detail.</think>
<tool_call>{"temp": 18, "exposure": -0.1, "contrast": 22, "shadows": -28, "blacks": -22, "clarity": 14, "vibrance": 10, "texture": 8}</tool_call>

## Example 2 (Chinese)
User: "\u8ba9\u4eba\u50cf\u80a4\u8272\u66f4\u67d4\u548c\u3002"
<think>\u539f\u7247\u80a4\u8272\u504f\u51b7\u3002\u6211\u4f1a\u63d0\u9ad8\u8272\u6e29\u8ba9\u80a4\u8272\u53d8\u6696\uff0c\u7565\u538b\u9ad8\u5149\uff0c\u52a0\u4e00\u70b9\u9690\u73b0\u8425\u9020\u67d4\u8f6f\u611f\u3002</think>
<tool_call>{"temp": 8, "tint": 4, "exposure": 0.05, "highlights": -18, "shadows": 6, "clarity": -8, "vibrance": 5, "texture": -4}</tool_call>

# THE ONLY ALLOWED KEYS (13 total -- NEVER use any other key)
temp, tint, exposure, contrast, highlights, shadows, whites, blacks, clarity, vibrance, saturation, texture, dehaze

# FORBIDDEN -- DO NOT INCLUDE THESE EVER
- NO ProcessVersion, whiteBalance, HasSettings, Look, Profile, CameraProfile
- NO ToneCurve*, ToneCurvePV2012*, ParametricShadow*, ParametricMidtone*, ParametricHighlight*
- NO Sharpness, SharpenRadius, SharpenDetail, SharpenEdgeMasking
- NO ColorNoiseReduction*, Defringe*, PerspectiveScale
- NO HueAdjustment*, SaturationAdjustment*, LuminanceAdjustment* (any per-color HSL)
- NO SplitToning*, ColorGrade*, PointColors

# JSON SYNTAX (must be valid parseable JSON)
- Double quotes "temp" not 'temp'
- Numbers only as values (int or float), no strings, no null, no nested objects, no arrays
- No trailing comma, no comments

# LENGTH
- Output 4-8 keys (rarely all 13). DO NOT pad with zeros for unused keys.
- Keep <think> short (2-3 sentences)
- Total response should be under 800 characters
"""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('utf-8')


def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return m.group(1).strip() if m else None


ALLOWED_LR_KEYS = {
    'temp', 'tint', 'exposure', 'contrast', 'highlights', 'shadows',
    'whites', 'blacks', 'clarity', 'vibrance', 'saturation', 'texture', 'dehaze',
}


def repair_json(s: str) -> str:
    """修复 JarvisEvo 输出的常见 JSON 语法错误:
    1. 单引号 -> 双引号
    2. 整数 key {1: 0} -> 双引号 "1"
    3. 缺值 "k":, -> 删除该 key
    4. 改造 尾部 逗号
    返回修复后的字符串 (仍不保证 可解析).
    """
    if not s:
        return s
    # 1. 单引号包裹的字符串 -> 双引号
    s = re.sub(r"'([^'\\]*?)'", r'"\1"', s)
    # 2. 整数键: {1:0,2:0} -> {"1":0,"2":0}
    s = re.sub(r'(\{|,)\s*(\d+)\s*:', r'\1"\2":', s)
    # 3. 缺值的键值对: "k": ,  或  "k":}  -> 删除整个 key:value
    s = re.sub(r'"[^"]+"\s*:\s*(?=[,}])', '', s)
    # 4. 连续逗号 ,, -> ,  和 尾部逗号 ,] / ,}
    s = re.sub(r',\s*,', ',', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # 5. 第一个元素前面多余的逗号: { , -> {
    s = re.sub(r'([{\[])\s*,', r'\1', s)
    return s


def parse_tool_call(tc_text: str | None) -> tuple[dict, list[str]]:
    """尝试解析 tool_call JSON, 返回 (filtered_dict, issues).
    filtered_dict 只含 ALLOWED_LR_KEYS + numeric 值.
    issues 是描述修复/过滤动作的记录.
    """
    issues = []
    if not tc_text:
        return {}, ['empty tool_call']
    if tc_text.strip() == 'block':
        return {}, ['CRITICAL: literal "block" placeholder']

    # 去掉可能的 markdown fence
    text = tc_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception as e1:
        # 试修复
        repaired = repair_json(text)
        try:
            parsed = json.loads(repaired)
            issues.append(f'json_repaired (raw err: {str(e1)[:80]})')
        except Exception as e2:
            return {}, [f'json_unparseable: {str(e2)[:120]}']

    if not isinstance(parsed, dict):
        return {}, [f'tool_call not dict: {type(parsed).__name__}']

    # 过滤: 只保留 ALLOWED_LR_KEYS 且值为数字
    out = {}
    dropped = []
    for k, v in parsed.items():
        kn = k.strip().lower() if isinstance(k, str) else str(k).lower()
        if kn not in ALLOWED_LR_KEYS:
            dropped.append(k)
            continue
        if isinstance(v, bool):
            dropped.append(f'{k}=bool')
            continue
        if not isinstance(v, (int, float)):
            dropped.append(f'{k}={type(v).__name__}')
            continue
        out[kn] = v
    if dropped:
        issues.append(f'dropped {len(dropped)} non-allowed keys/values')
    return out, issues


def label_one(client: OpenAI, sample_dir: Path, task_type: str = 'lightroom',
              model: str = 'jarvisevo', timeout: int = 120) -> dict:
    """Generate CoT label for ONE sample. Returns dict on success, {} on error."""
    input_img = sample_dir / 'input.jpg'
    user_want_f = sample_dir / 'user_want.txt'
    if not input_img.exists() or not user_want_f.exists():
        return {}

    user_prompt = user_want_f.read_text(encoding='utf-8').strip()
    msg_content = [
        {'type': 'image_url',
         'image_url': {'url': f'data:image/jpeg;base64,{encode_image(input_img)}'}},
        {'type': 'text',
         'text': f'<task_type>{task_type}</task_type>. Instruction: {user_prompt}'},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT_SHORT},
                {'role': 'user', 'content': msg_content},
            ],
            timeout=timeout,
            temperature=0.1,  # \u5fae\u63d0\u9632\u6b62\u65e9\u505c
            max_tokens=4096,  # v4: 2048 \u88ab\u622a\u65ad (sample 1) \u8c03\u9ad8
        )
        text = resp.choices[0].message.content
        tc_raw = extract_tag(text, 'tool_call')
        parsed, issues = parse_tool_call(tc_raw)
        return {
            'image_dir': str(sample_dir),
            'sample_id': sample_dir.name,
            'lang': sample_dir.parent.name,  # CN or EN
            'user_want': user_prompt,
            'response': text,
            'think': extract_tag(text, 'think'),
            'tool_call': tc_raw,
            'parsed_tool_call': parsed,  # dict, filtered to allowed keys
            'parse_issues': issues,
            'has_box': '<box>' in user_prompt,
            'parsed_ok': len(parsed) >= 2,  # 至少 2 个参数才算有效样本
        }
    except Exception as e:
        return {'image_dir': str(sample_dir), 'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp/datasets/ArtEdit-Bench/ArtEdit-Bench-Lr',
                    help='ArtEdit-Bench-Lr \u6839\u76ee\u5f55 (\u4e0b\u5c5e CN/ EN/)')
    ap.add_argument('--out', default='/root/autodl-tmp/datasets/ArtEdit-Bench/pseudo_labels.jsonl')
    ap.add_argument('--api_url', default='http://localhost:8086/v1')
    ap.add_argument('--model', default='jarvisevo')
    ap.add_argument('--api_key', default='0')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--limit', type=int, default=0, help='\u6700\u591a\u6837\u672c\u6570 (0=\u5168\u90e8)')
    ap.add_argument('--langs', default='CN,EN', help='\u5904\u7406\u54ea\u4e9b\u8bed\u8a00\u76ee\u5f55')
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # \u6536\u96c6\u6240\u6709\u6837\u672c\u76ee\u5f55
    sample_dirs = []
    for lang in args.langs.split(','):
        lang_dir = root / lang.strip()
        if lang_dir.exists():
            sample_dirs.extend(sorted([d for d in lang_dir.iterdir() if d.is_dir()]))
    print(f'[INFO] {len(sample_dirs)} samples collected from langs={args.langs}')
    if args.limit:
        sample_dirs = sample_dirs[:args.limit]

    # \u8df3\u8fc7\u5df2\u5904\u7406\u7684
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['image_dir'])
                except Exception:
                    pass
        print(f'[INFO] {len(done_ids)} samples already labeled, skipping')

    todo = [d for d in sample_dirs if str(d) not in done_ids]
    print(f'[INFO] {len(todo)} samples to label')

    client = OpenAI(api_key=args.api_key, base_url=args.api_url, timeout=args.timeout)

    # \u68c0\u67e5 vLLM \u670d\u52a1\u53ef\u8fbe
    try:
        models = client.models.list()
        print(f'[INFO] vLLM API OK, models: {[m.id for m in models.data]}')
    except Exception as e:
        print(f'[ERR] vLLM \u4e0d\u901a: {e}')
        print(f'[HINT] \u542f\u52a8: VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... --port 8086 ...')
        sys.exit(1)

    # \u5e76\u884c\u8c03\u7528
    n_success = n_fail = 0
    t0 = time.time()
    with open(out_path, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {pool.submit(label_one, client, d, 'lightroom', args.model, args.timeout): d
                       for d in todo}
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                # \u6709\u6548\u6837\u672c = parsed_tool_call \u81f3\u5c11\u542b 2 \u4e2a\u5408\u6cd5 LR \u53c2\u6570
                if result.get('parsed_ok'):
                    n_success += 1
                else:
                    n_fail += 1
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 0.1)
                    remain = (len(todo) - i - 1) / max(rate, 0.001)
                    print(f'  [{i+1}/{len(todo)}] ok={n_success} fail={n_fail}  '
                          f'rate={rate:.2f}/s ETA={remain/60:.1f} min')

    print(f'\n[DONE] {n_success} labels OK / {n_fail} fail / {len(todo)} total')
    print(f'[OUT] {out_path}')


if __name__ == '__main__':
    main()
