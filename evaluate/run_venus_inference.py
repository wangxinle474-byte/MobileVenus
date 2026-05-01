"""
Venus 推理脚本 — 获取 Venus 对测试图像的文本输出

用法:
  python evaluate/run_venus_inference.py \
    --model_path /path/to/Venus-Q-Stage2 \
    --image_dir /path/to/test/images \
    --output_path evaluate/results/venus_outputs.json

输出格式:
  {
    "image_001.jpg": "This image has good composition but poor lighting...",
    "image_002.jpg": "The colors are vibrant and well-balanced...",
    ...
  }
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 美学分析 prompt (与 Venus 评估脚本一致)
AESTHETIC_PROMPT = "Please critically analyze this image from an aesthetic perspective."

# 参数提取 prompt (额外)
PARAMETER_PROMPT = (
    "Analyze this image's aesthetic quality and suggest specific camera parameter adjustments. "
    "Consider: exposure compensation (EV), white balance (color temperature in Kelvin), "
    "focus point position, whether HDR should be enabled, and the best shooting mode "
    "(auto/portrait/night/landscape/macro). Be specific with numbers."
)


def run_inference(
    model_path: str,
    image_dir: str,
    output_path: str,
    prompt: str = AESTHETIC_PROMPT,
    max_images: int = -1,
):
    """
    运行 Venus 推理

    Args:
        model_path: Venus 模型路径 (如 Venus-Q-Stage2)
        image_dir: 测试图片目录
        output_path: 输出 JSON 路径
        prompt: 推理提示词
        max_images: 最大处理图片数 (-1 = 全部)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 加载模型
    logger.info(f"Loading Venus model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, bf16=True
    ).eval()
    logger.info("Model loaded")

    # 收集图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ])
    if max_images > 0:
        image_files = image_files[:max_images]
    logger.info(f"Processing {len(image_files)} images")

    # 推理
    outputs = {}
    for image_name in tqdm(image_files, desc="Venus Inference"):
        image_path = os.path.join(image_dir, image_name)
        try:
            query = tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt},
            ])
            response, _ = model.chat(tokenizer, query=query, history=None)
            outputs[image_name] = response
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            outputs[image_name] = ""

        # 每 10 张保存一次 (防止中途中断丢失)
        if len(outputs) % 10 == 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

    # 最终保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(outputs)} outputs to {output_path}")
    return outputs


def run_dual_prompt_inference(
    model_path: str,
    image_dir: str,
    output_dir: str,
    max_images: int = -1,
):
    """
    运行双 prompt 推理:
      1. 美学分析 prompt → 通用文本分析
      2. 参数建议 prompt → 针对性参数建议
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prompt 1: 通用美学分析
    logger.info("=== Running aesthetic analysis prompt ===")
    run_inference(
        model_path, image_dir,
        os.path.join(output_dir, "venus_aesthetic.json"),
        prompt=AESTHETIC_PROMPT,
        max_images=max_images,
    )

    # Prompt 2: 参数建议
    logger.info("=== Running parameter suggestion prompt ===")
    run_inference(
        model_path, image_dir,
        os.path.join(output_dir, "venus_params.json"),
        prompt=PARAMETER_PROMPT,
        max_images=max_images,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Venus inference for comparison")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to Venus model (e.g., Venus-Q-Stage2)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_path', type=str, default='evaluate/results/venus_outputs.json',
                        help='Output JSON path')
    parser.add_argument('--prompt', type=str, default='aesthetic',
                        choices=['aesthetic', 'parameter', 'dual'],
                        help='Which prompt to use')
    parser.add_argument('--max_images', type=int, default=-1,
                        help='Max images to process (-1 = all)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.prompt == 'dual':
        output_dir = str(Path(args.output_path).parent)
        run_dual_prompt_inference(args.model_path, args.image_dir, output_dir, args.max_images)
    else:
        prompt = AESTHETIC_PROMPT if args.prompt == 'aesthetic' else PARAMETER_PROMPT
        run_inference(args.model_path, args.image_dir, args.output_path, prompt, args.max_images)
