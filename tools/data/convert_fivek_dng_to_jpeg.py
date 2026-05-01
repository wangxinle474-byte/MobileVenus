"""
FiveK DNG → JPEG 转换 + 专家调参配对

将 5000 张 DNG 原始照片转换为 JPEG，并与 fivek_expert_params.json 中的
专家调参记录配对，生成可直接用于训练的 (图片, 参数) 数据集。

输出:
  E:\dataset\fivek_jpeg\         → 5000 张 JPEG 图片
  data/fivek_paired_train.json   → 训练集 (图片路径 + 真实专家参数)
  data/fivek_paired_val.json     → 验证集

用法:
  python tools/convert_fivek_dng_to_jpeg.py
  python tools/convert_fivek_dng_to_jpeg.py --max_size 512 --quality 90
  python tools/convert_fivek_dng_to_jpeg.py --expert_name "expert_c"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random

import numpy as np
from PIL import Image

try:
    import rawpy
except ImportError:
    print("需要安装: pip install rawpy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# DNG → JPEG 转换
# ============================================================

def convert_dng_to_jpeg(
    dng_path: str,
    output_path: str,
    max_size: int = 512,
    quality: int = 90,
) -> bool:
    """
    将单张 DNG 转换为 JPEG
    
    Args:
        dng_path: DNG 文件路径
        output_path: 输出 JPEG 路径
        max_size: 长边最大像素数 (节省空间和加速训练)
        quality: JPEG 质量 (1-100)
    
    Returns:
        True if success
    """
    try:
        raw = rawpy.imread(dng_path)
        # use_camera_wb: 使用相机原始白平衡 (重要: 保留原始色温信息)
        # half_size: 快速解码, 尺寸减半
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=True,
            no_auto_bright=True,  # 不自动调亮度, 保留原始曝光
            output_bps=8,
        )
        raw.close()
        
        img = Image.fromarray(rgb)
        
        # 缩放到 max_size
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        img.save(output_path, 'JPEG', quality=quality)
        return True
        
    except Exception as e:
        logger.warning(f"转换失败 {dng_path}: {e}")
        return False


# ============================================================
# 收集所有 DNG 文件
# ============================================================

def collect_dng_files(fivek_root: str) -> List[Path]:
    """收集所有 DNG 文件"""
    root = Path(fivek_root) / "raw_photos"
    dng_files = []
    
    for subdir in sorted(root.iterdir()):
        if subdir.name.startswith("HQa"):
            photos_dir = subdir / "photos"
            if photos_dir.exists():
                for f in sorted(photos_dir.glob("*.dng")):
                    dng_files.append(f)
    
    return dng_files


# ============================================================
# 加载专家调参并按图片名索引
# ============================================================

def load_expert_params(json_path: str) -> Dict[str, List[dict]]:
    """
    加载专家调参, 按 image_name 分组
    
    Returns:
        dict: { "a0001-jmac_DSC1459.dng": [expert_a_params, expert_b_params, ...] }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    params_by_image = defaultdict(list)
    for sample in data['samples']:
        params_by_image[sample['image_name']].append(sample)
    
    return dict(params_by_image)


# ============================================================
# 创建训练数据集
# ============================================================

def create_training_dataset(
    jpeg_dir: str,
    expert_params: Dict[str, List[dict]],
    expert_name: str = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    配对 JPEG 图片和专家参数, 生成训练/验证集
    
    Args:
        jpeg_dir: JPEG 图片目录
        expert_params: 按图片名分组的专家参数
        expert_name: 指定使用哪个专家 (None=全部)
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        (train_samples, val_samples)
    """
    jpeg_dir = Path(jpeg_dir)
    samples = []
    matched = 0
    unmatched = 0
    
    for jpeg_path in sorted(jpeg_dir.glob("*.jpg")):
        # JPEG 文件名 → DNG 文件名
        dng_name = jpeg_path.stem + ".dng"
        
        if dng_name not in expert_params:
            unmatched += 1
            continue
        
        matched += 1
        expert_records = expert_params[dng_name]
        
        # 筛选专家
        if expert_name:
            expert_records = [r for r in expert_records if r['expert'] == expert_name]
        
        for record in expert_records:
            sample = {
                'id': f"fivek_{jpeg_path.stem}_{record['expert']}",
                'image': str(jpeg_path.absolute()),
                'image_name': dng_name,
                'source': 'fivek_expert',
                'expert': record['expert'],
                'targets': {
                    'ev_compensation': float(record.get('ev_compensation', 0)),
                    'white_balance': float(record.get('white_balance', 5500)),
                    'contrast': float(record.get('contrast', 0)),
                    'brightness': float(record.get('brightness', 0)),
                    'shadows': float(record.get('shadows', 0)),
                    'highlights': float(record.get('highlights', 0)),
                    'saturation': float(record.get('saturation', 0)),
                    'vibrance': float(record.get('vibrance', 0)),
                },
            }
            samples.append(sample)
    
    logger.info(f"配对结果: {matched} 张匹配, {unmatched} 张未匹配, 共 {len(samples)} 条记录")
    
    # 按图片分 train/val (避免同一图片的不同专家分到不同集)
    image_names = sorted(set(s['image_name'] for s in samples))
    random.seed(seed)
    random.shuffle(image_names)
    
    val_count = int(len(image_names) * val_ratio)
    val_images = set(image_names[:val_count])
    
    train_samples = [s for s in samples if s['image_name'] not in val_images]
    val_samples = [s for s in samples if s['image_name'] in val_images]
    
    logger.info(f"训练集: {len(train_samples)} 条 ({len(image_names) - val_count} 张图)")
    logger.info(f"验证集: {len(val_samples)} 条 ({val_count} 张图)")
    
    return train_samples, val_samples


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FiveK DNG → JPEG + 专家参数配对")
    parser.add_argument('--fivek_root', default=r'E:\dataset\fivek_dataset',
                        help='FiveK 数据集根目录')
    parser.add_argument('--output_jpeg_dir', default=r'E:\dataset\fivek_jpeg',
                        help='JPEG 输出目录')
    parser.add_argument('--expert_params', default=None,
                        help='专家调参 JSON 路径 (默认自动查找)')
    parser.add_argument('--expert_name', default=None,
                        help='指定专家 (默认全部)')
    parser.add_argument('--max_size', type=int, default=512,
                        help='长边最大像素')
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG 质量')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--skip_convert', action='store_true',
                        help='跳过 DNG→JPEG 转换 (已转换过)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最多转换张数 (调试用)')
    args = parser.parse_args()
    
    # 路径
    project_root = Path(__file__).parent.parent.parent
    if args.expert_params is None:
        args.expert_params = str(project_root / 'data' / 'fivek_expert_params.json')
    
    output_data_dir = project_root / 'data'
    jpeg_dir = Path(args.output_jpeg_dir)
    
    # Step 1: DNG → JPEG
    if not args.skip_convert:
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        dng_files = collect_dng_files(args.fivek_root)
        logger.info(f"找到 {len(dng_files)} 张 DNG 文件")
        
        if args.max_images:
            dng_files = dng_files[:args.max_images]
            logger.info(f"限制转换 {args.max_images} 张")
        
        # 检查已转换的
        existing = set(f.stem for f in jpeg_dir.glob("*.jpg"))
        to_convert = [f for f in dng_files if f.stem not in existing]
        logger.info(f"已转换: {len(existing)}, 待转换: {len(to_convert)}")
        
        success = 0
        fail = 0
        for i, dng_path in enumerate(to_convert):
            output_path = jpeg_dir / f"{dng_path.stem}.jpg"
            if convert_dng_to_jpeg(str(dng_path), str(output_path), args.max_size, args.quality):
                success += 1
            else:
                fail += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"进度: {i+1}/{len(to_convert)} (成功: {success}, 失败: {fail})")
        
        logger.info(f"转换完成: 成功 {success}, 失败 {fail}")
    else:
        logger.info(f"跳过转换, 直接使用: {jpeg_dir}")
    
    # Step 2: 加载专家参数
    logger.info(f"加载专家调参: {args.expert_params}")
    expert_params = load_expert_params(args.expert_params)
    logger.info(f"共 {len(expert_params)} 张图片的专家参数")
    
    # Step 3: 配对 + 划分
    train_samples, val_samples = create_training_dataset(
        str(jpeg_dir), expert_params, args.expert_name, args.val_ratio
    )
    
    # Step 4: 保存
    train_path = output_data_dir / 'fivek_paired_train.json'
    val_path = output_data_dir / 'fivek_paired_val.json'
    
    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"保存: {train_path} ({len(train_samples)} 条)")
    logger.info(f"保存: {val_path} ({len(val_samples)} 条)")
    
    # 统计
    if train_samples:
        evs = [s['targets']['ev_compensation'] for s in train_samples]
        wbs = [s['targets']['white_balance'] for s in train_samples]
        logger.info(f"EV 范围: [{min(evs):.2f}, {max(evs):.2f}], 均值: {np.mean(evs):.2f}")
        logger.info(f"WB 范围: [{min(wbs):.0f}, {max(wbs):.0f}], 均值: {np.mean(wbs):.0f}")


if __name__ == '__main__':
    main()
