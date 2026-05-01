"""
AutoDL 端: 用 Venus 对 FiveK 图片生成美学分析文本，作为 Stage A 训练数据
消除域偏移: Stage A 数据 = FiveK 图片 (与 Stage B 同域)

输出格式: Stage1 JSON (与 train_Stage1.json 相同格式)
  [{"id": "fivek_0001", "conversations": [
      {"from": "user",  "value": "Picture 1: <img>...</img>\nPROMPT"},
      {"from": "assistant", "value": "VENUS_ANALYSIS_TEXT"}
  ]}, ...]

用法 (AutoDL):
  python gen_fivek_stage_a.py
  python gen_fivek_stage_a.py --max_images 2000 --output /root/autodl-tmp/fivek_stage_a.json
  python gen_fivek_stage_a.py --resume  # 断点续跑
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

VENUS_DIR = "/root/autodl-tmp/Venus-Q-Stage1"
FIVEK_DIR = "/root/autodl-tmp/fivek_jpeg"

# Stage A 分析 prompt: 要求 Venus 给出详细的美学分析 (用于语义对齐)
ANALYSIS_PROMPT = """As a professional photography critic, provide a detailed aesthetic analysis of this photo.
Describe specifically:
1. Lighting quality: brightness level, shadows, highlights, whether exposure is correct
2. Color characteristics: white balance tendency (warm/cool/neutral), color saturation, color harmony
3. Contrast and tonal range: is the contrast appropriate, are details preserved in dark/light areas
4. Subject clarity: is the main subject clear and well-exposed
5. Overall adjustment suggestion: what Lightroom parameters would most improve this photo

Be specific and descriptive. Use 3-5 sentences total."""


def load_venus():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    venus_dir = VENUS_DIR
    if not os.path.exists(os.path.join(venus_dir, 'config.json')):
        for alt in ['/root/Venus-Q-Stage1', '/root/autodl-pub/Venus-Q-Stage1']:
            if os.path.exists(os.path.join(alt, 'config.json')):
                venus_dir = alt
                break
        else:
            raise FileNotFoundError(f"Venus not found at {VENUS_DIR}")

    logger.info(f"Loading Venus from {venus_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        venus_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        venus_dir, device_map={"": "cuda:0"}, trust_remote_code=True,
        bf16=True, local_files_only=True
    ).eval()
    logger.info("Venus loaded")
    return model, tokenizer


def query_venus(model, tokenizer, img_path: str, prompt: str, max_new_tokens: int = 256) -> str:
    import torch
    query = tokenizer.from_list_format([
        {'image': img_path},
        {'text': prompt},
    ])
    with torch.no_grad():
        response, _ = model.chat(
            tokenizer, query=query, history=None,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return response.strip()


def get_image_list(fivek_dir: Path, max_images: int) -> list:
    exts = {'.jpg', '.jpeg', '.png'}
    images = sorted([p for p in fivek_dir.iterdir() if p.suffix.lower() in exts])
    if max_images > 0:
        images = images[:max_images]
    return images


def load_existing(output_path: Path) -> dict:
    """加载已有结果用于断点续跑"""
    if not output_path.exists():
        return {}
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item for item in data}


def save_results(items: list, output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fivek_dir', default=FIVEK_DIR)
    parser.add_argument('--output', default='/root/autodl-tmp/fivek_stage_a.json')
    parser.add_argument('--max_images', type=int, default=2000,
                        help='最多处理图片数 (-1=全部5120张)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='每N张保存一次中间结果')
    parser.add_argument('--resume', action='store_true',
                        help='断点续跑，跳过已处理的图片')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    args = parser.parse_args()

    fivek_dir = Path(args.fivek_dir)
    output_path = Path(args.output)

    if not fivek_dir.exists():
        logger.error(f"FiveK 目录不存在: {fivek_dir}")
        sys.exit(1)

    images = get_image_list(fivek_dir, args.max_images)
    logger.info(f"FiveK 图片总数: {len(images)}")

    # 断点续跑
    existing = {}
    if args.resume:
        existing = load_existing(output_path)
        logger.info(f"已有结果: {len(existing)} 张，跳过")

    model, tokenizer = load_venus()

    results = list(existing.values())
    skipped = 0
    errors = 0
    t_start = time.time()

    for i, img_path in enumerate(images, 1):
        img_id = img_path.stem  # e.g. "fivek_0001"

        if img_id in existing:
            skipped += 1
            continue

        try:
            t0 = time.time()
            response = query_venus(model, tokenizer, str(img_path),
                                   ANALYSIS_PROMPT, args.max_new_tokens)
            elapsed = time.time() - t0

            item = {
                "id": img_id,
                "conversations": [
                    {
                        "from": "user",
                        "value": f"Picture 1: <img>{str(img_path)}</img>\n{ANALYSIS_PROMPT}"
                    },
                    {
                        "from": "assistant",
                        "value": response
                    }
                ]
            }
            results.append(item)

            done = len(results)
            total_elapsed = time.time() - t_start
            avg_per_img = total_elapsed / max(done - skipped, 1)
            remaining = (len(images) - i) * avg_per_img / 60

            logger.info(
                f"[{i}/{len(images)}] {img_id} | {elapsed:.1f}s | "
                f"avg={avg_per_img:.1f}s | ETA≈{remaining:.0f}min"
            )

            if done % args.save_every == 0:
                save_results(results, output_path)
                logger.info(f"  → 中间保存: {output_path} ({done} 条)")

        except Exception as e:
            errors += 1
            logger.error(f"[{i}] {img_id} 失败: {e}")
            if errors > 20:
                logger.error("错误过多，终止")
                break

    save_results(results, output_path)
    logger.info("=" * 60)
    logger.info(f"完成! 生成: {len(results)} 条, 跳过: {skipped}, 错误: {errors}")
    logger.info(f"输出: {output_path} ({output_path.stat().st_size/1024/1024:.1f}MB)")
    logger.info("下一步: 下载此文件到本地，运行 embed_texts.py 生成 .npz 文件")


if __name__ == '__main__':
    main()
