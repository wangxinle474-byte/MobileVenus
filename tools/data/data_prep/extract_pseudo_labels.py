"""
从 JPEG 图像自动提取伪标签 (5参数版)

无需 Lightroom 参数，直接从图像统计量推断相机参数标签:
  1. EV补偿    ← 图像平均亮度
  2. 白平衡    ← R/B 通道比率 (色温估计)
  3. 对焦点    ← 拉普拉斯算子峰值位置 (清晰度最高区域)
  4. HDR       ← 动态范围分析 (高光/阴影占比)
  5. 拍摄模式  ← 场景特征组合推断

同时生成 9 类问题检测标签 + 严重程度。

用法:
  python tools/extract_pseudo_labels.py --image_dir data/fivek_jpeg --output_dir data/fivek_pseudo
  python tools/extract_pseudo_labels.py --image_dir data/fivek_jpeg --output_dir data/fivek_pseudo --max_samples 500
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import cv2
except ImportError:
    print("需要安装: pip install pillow opencv-python")
    exit(1)


# ============================================================
# 核心: 从图像提取 5 个相机参数伪标签
# ============================================================

def extract_ev_compensation(img_rgb: np.ndarray) -> float:
    """
    从图像亮度估计 EV 补偿值
    
    逻辑: 理想曝光的平均亮度约 118 (18% 灰)
          偏暗 → 需要正 EV 补偿
          偏亮 → 需要负 EV 补偿
    
    Returns:
        ev: float in [-3, 3]
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    
    # 118 = 18% 灰的亮度值 (gamma 2.2 下)
    # EV = log2(target / current), 简化为线性映射
    target = 118.0
    if mean_brightness < 1:
        mean_brightness = 1.0
    
    # 亮度偏差 → EV 补偿
    ratio = target / mean_brightness
    ev = np.log2(ratio)
    ev = float(np.clip(ev, -3.0, 3.0))
    
    return round(ev, 3)


def extract_white_balance(img_rgb: np.ndarray) -> int:
    """
    从 R/B 通道比率估计色温 (K)
    
    逻辑: R/B 高 → 暖色 → 低色温 (需要提高 WB 补偿)
          R/B 低 → 冷色 → 高色温 (需要降低 WB 补偿)
    
    Returns:
        wb: int in [2000, 10000]
    """
    r_mean = img_rgb[:, :, 0].mean()
    b_mean = img_rgb[:, :, 2].mean()
    
    if b_mean < 1:
        b_mean = 1.0
    
    rb_ratio = r_mean / b_mean
    
    # R/B ratio → 色温 (经验映射)
    # rb_ratio ~0.7 → ~9000K (冷色)
    # rb_ratio ~1.0 → ~5500K (中性)
    # rb_ratio ~1.5 → ~3000K (暖色)
    wb = int(5500 + (1.0 - rb_ratio) * 5000)
    wb = int(np.clip(wb, 2000, 10000))
    
    return wb


def extract_focus_point(img_rgb: np.ndarray) -> List[float]:
    """
    用拉普拉斯算子找到清晰度最高的区域作为对焦点
    
    将图像分成 7x7 网格，计算每个网格的拉普拉斯方差，
    取方差最高的位置作为对焦点。
    
    Returns:
        [x, y] in [0, 1]²
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    grid_rows, grid_cols = 7, 7
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    best_var = -1
    best_x, best_y = 0.5, 0.5
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell = gray[y1:y2, x1:x2]
            
            if cell.size == 0:
                continue
            
            lap_var = cv2.Laplacian(cell, cv2.CV_64F).var()
            
            if lap_var > best_var:
                best_var = lap_var
                best_x = (c + 0.5) / grid_cols
                best_y = (r + 0.5) / grid_rows
    
    return [round(float(best_x), 4), round(float(best_y), 4)]


def extract_hdr_need(img_rgb: np.ndarray) -> int:
    """
    分析动态范围判断是否需要 HDR
    
    逻辑: 如果高光过曝和阴影欠曝同时存在 → 需要 HDR
    
    Returns:
        hdr: 0 or 1
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 高光占比 (>240) 和阴影占比 (<15)
    highlight_ratio = (gray > 240).sum() / gray.size
    shadow_ratio = (gray < 15).sum() / gray.size
    
    # 同时有过曝和欠曝 → 高动态范围场景
    if highlight_ratio > 0.05 and shadow_ratio > 0.05:
        return 1
    
    # 动态范围很大
    p5, p95 = np.percentile(gray, [5, 95])
    if (p95 - p5) > 200:
        return 1
    
    return 0


def extract_shooting_mode(
    img_rgb: np.ndarray,
    ev: float,
    wb: int,
    focus_point: List[float]
) -> int:
    """
    基于图像特征推断拍摄模式
    
    Returns:
        mode: 0=auto, 1=portrait, 2=night, 3=landscape, 4=macro
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mean_brightness = gray.mean()
    
    # 颜色饱和度
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    
    # 中心区域 vs 边缘区域的清晰度差异 (景深指示)
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    border_region = np.concatenate([
        gray[:h//4, :].flatten(),
        gray[3*h//4:, :].flatten(),
        gray[:, :w//4].flatten(),
        gray[:, 3*w//4:].flatten()
    ])
    center_sharpness = cv2.Laplacian(center_region, cv2.CV_64F).var()
    border_sharpness = cv2.Laplacian(
        border_region.reshape(-1, 1), cv2.CV_64F
    ).var() if border_region.size > 0 else center_sharpness
    
    depth_ratio = center_sharpness / (border_sharpness + 1e-6)
    
    # 夜景: 很暗
    if mean_brightness < 50:
        return 2  # night
    
    # 风景: 高饱和度 + 高边缘密度 + 均匀清晰度
    if saturation > 80 and edge_density > 0.1 and depth_ratio < 3:
        return 3  # landscape
    
    # 人像: 浅景深 (中心清晰、边缘模糊) + 暖色温
    if depth_ratio > 5 and wb < 6000:
        return 1  # portrait
    
    # 微距: 极浅景深 + 高中心清晰度
    if depth_ratio > 10 and center_sharpness > 500:
        return 4  # macro
    
    return 0  # auto


# ============================================================
# 问题检测
# ============================================================

def detect_problems(
    img_rgb: np.ndarray,
    ev: float,
    wb: int
) -> Tuple[List[int], List[int]]:
    """
    从图像统计量检测 9 类问题
    
    Returns:
        problem_labels: [9] 二值向量
        severity_labels: [9] 严重程度 (0=mild, 1=moderate, 2=severe)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    mean_bright = gray.mean()
    std_bright = gray.std()
    saturation = hsv[:, :, 1].mean()
    
    problems = [0] * 9
    severities = [0] * 9
    
    # 0: underexposure (欠曝)
    if mean_bright < 60:
        problems[0] = 1
        severities[0] = 2 if mean_bright < 30 else (1 if mean_bright < 45 else 0)
    
    # 1: overexposure (过曝)
    if mean_bright > 200:
        problems[1] = 1
        severities[1] = 2 if mean_bright > 230 else (1 if mean_bright > 215 else 0)
    
    # 2: color_cast (色偏)
    r_mean = img_rgb[:, :, 0].mean()
    g_mean = img_rgb[:, :, 1].mean()
    b_mean = img_rgb[:, :, 2].mean()
    channel_std = np.std([r_mean, g_mean, b_mean])
    if channel_std > 25:
        problems[2] = 1
        severities[2] = 2 if channel_std > 45 else (1 if channel_std > 35 else 0)
    
    # 3: poor_composition (构图不佳) - 用边缘分布不均衡度估计
    h, w = gray.shape
    left_energy = cv2.Laplacian(gray[:, :w//2], cv2.CV_64F).var()
    right_energy = cv2.Laplacian(gray[:, w//2:], cv2.CV_64F).var()
    balance = min(left_energy, right_energy) / (max(left_energy, right_energy) + 1e-6)
    if balance < 0.2:
        problems[3] = 1
        severities[3] = 1
    
    # 4: blur (模糊)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        problems[4] = 1
        severities[4] = 2 if lap_var < 15 else (1 if lap_var < 30 else 0)
    
    # 5: noise (噪点) - 高ISO特征: 暗图+高频噪声
    if mean_bright < 80:
        # 高频分量比例
        blur_img = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.abs(gray.astype(float) - blur_img.astype(float)).mean()
        if noise_level > 8:
            problems[5] = 1
            severities[5] = 2 if noise_level > 15 else (1 if noise_level > 10 else 0)
    
    # 6: backlight (逆光)
    top_half = gray[:h//2, :].mean()
    bottom_half = gray[h//2:, :].mean()
    if top_half > 180 and bottom_half < 80:
        problems[6] = 1
        severities[6] = 1
    
    # 7: low_contrast (对比度低)
    if std_bright < 30:
        problems[7] = 1
        severities[7] = 2 if std_bright < 15 else (1 if std_bright < 22 else 0)
    
    # 8: background_messy (背景杂乱) - 边缘区域纹理复杂度
    border = np.concatenate([
        gray[:h//6, :].flatten(),
        gray[5*h//6:, :].flatten(),
    ])
    if border.size > 0:
        border_std = border.std()
        if border_std > 60:
            problems[8] = 1
            severities[8] = 1
    
    return problems, severities


# ============================================================
# 主处理流程
# ============================================================

def process_image(image_path: str, image_id: str) -> Optional[Dict]:
    """
    处理单张图像，提取所有伪标签
    
    Returns:
        sample: 统一格式的训练样本 (与 convert_fivek.py 输出格式一致)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # 为加速，将大图缩小到 640px
        h, w = img_np.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_np = cv2.resize(img_np, (new_w, new_h))
        
        # 提取 5 个参数
        ev = extract_ev_compensation(img_np)
        wb = extract_white_balance(img_np)
        focus = extract_focus_point(img_np)
        hdr = extract_hdr_need(img_np)
        mode = extract_shooting_mode(img_np, ev, wb, focus)
        
        # 问题检测
        problems, severities = detect_problems(img_np, ev, wb)
        
        sample = {
            "id": f"fivek_{image_id}",
            "image": str(image_path),
            "source": "fivek_pseudo",
            "targets": {
                "ev_compensation": ev,
                "white_balance": wb,
                "focus_point": focus,
                "hdr": hdr,
                "mode": mode,
            },
            "problem_labels": problems,
            "severity_labels": severities,
        }
        
        return sample
        
    except Exception as e:
        logger.warning(f"处理失败: {image_path}, {e}")
        return None


def process_dataset(
    image_dir: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    val_split: float = 0.1
) -> Tuple[List[Dict], List[Dict]]:
    """
    处理整个图像目录
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录
        max_samples: 最大样本数
        val_split: 验证集比例
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有图像
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(image_path.rglob(ext)))
    
    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到图像文件")
        return [], []
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"开始提取伪标签...\n")
    
    # 处理每张图像
    samples = []
    stats = {'modes': {}, 'problems': {}, 'hdr_count': 0}
    mode_names = ['auto', 'portrait', 'night', 'landscape', 'macro']
    problem_names = [
        'underexposure', 'overexposure', 'color_cast',
        'poor_composition', 'blur', 'noise',
        'backlight', 'low_contrast', 'background_messy'
    ]
    
    for i, img_file in enumerate(image_files):
        image_id = img_file.stem
        sample = process_image(str(img_file), image_id)
        
        if sample is None:
            continue
        
        samples.append(sample)
        
        # 统计
        m = mode_names[sample['targets']['mode']]
        stats['modes'][m] = stats['modes'].get(m, 0) + 1
        if sample['targets']['hdr'] == 1:
            stats['hdr_count'] += 1
        for j, p in enumerate(sample['problem_labels']):
            if p == 1:
                pname = problem_names[j]
                stats['problems'][pname] = stats['problems'].get(pname, 0) + 1
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(image_files)} ...")
    
    print(f"\n处理完成: {len(samples)} 个有效样本")
    
    # 划分训练集/验证集
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    val_size = max(int(len(samples) * val_split), 1)
    train_size = len(samples) - val_size
    
    train_samples = [samples[i] for i in indices[:train_size]]
    val_samples = [samples[i] for i in indices[train_size:]]
    
    # 保存
    train_file = output_path / "pseudo_5params_train.json"
    val_file = output_path / "pseudo_5params_val.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"伪标签提取完成 (5参数版)")
    print(f"{'='*60}")
    print(f"  总样本: {len(samples)}")
    print(f"  训练集: {len(train_samples)} → {train_file}")
    print(f"  验证集: {len(val_samples)} → {val_file}")
    print(f"\n  拍摄模式分布: {stats['modes']}")
    print(f"  HDR 开启: {stats['hdr_count']}/{len(samples)} "
          f"({100*stats['hdr_count']/max(len(samples),1):.1f}%)")
    print(f"  问题类型: {stats['problems']}")
    
    # 打印参数范围
    evs = [s['targets']['ev_compensation'] for s in samples]
    wbs = [s['targets']['white_balance'] for s in samples]
    print(f"\n  EV 范围: [{min(evs):.2f}, {max(evs):.2f}], 均值={np.mean(evs):.2f}")
    print(f"  WB 范围: [{min(wbs)}, {max(wbs)}], 均值={np.mean(wbs):.0f}")
    
    # 样本示例
    print(f"\n  样本示例:")
    s = samples[0]
    print(f"    image: {s['image']}")
    print(f"    targets: {s['targets']}")
    print(f"    problems: {s['problem_labels']}")
    
    return train_samples, val_samples


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 JPEG 图像提取伪标签 (5参数版)"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="图像目录")
    parser.add_argument("--output_dir", type=str, default="data/fivek_pseudo",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    process_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_split=args.val_split
    )
