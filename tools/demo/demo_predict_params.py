"""
FiveK SemanticBridge 推理 Demo
加载训练好的模型，对任意图片预测 EV 补偿和白平衡参数，并可视化效果。

Usage:
  python tools/demo_predict_params.py                          # 随机抽 8 张 FiveK
  python tools/demo_predict_params.py --image path/to/img.jpg  # 指定单张图片
  python tools/demo_predict_params.py --dir path/to/folder     # 批量推理
"""
import sys
import json
import argparse
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_fivek_semantic import FiveKParamModel, FiveKExpertDataset


def load_model(ckpt_path: str, device: torch.device) -> FiveKParamModel:
    model = FiveKParamModel(image_size=224)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def predict_single(model, image_path: str, device: torch.device) -> dict:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
    
    ev = float(outputs['ev_compensation'].squeeze())
    wb = float(outputs['white_balance'].squeeze())
    confidence = outputs['param_confidence'].squeeze().cpu().numpy()
    
    return {
        'ev_compensation': round(ev, 2),
        'white_balance': round(wb, 0),
        'confidence': confidence,
    }


def apply_ev_simulation(img: Image.Image, ev: float) -> Image.Image:
    """模拟 EV 补偿效果"""
    factor = 2.0 ** (ev * 0.3)  # 缩小效果避免过曝
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_wb_simulation(img: Image.Image, wb: float) -> Image.Image:
    """模拟白平衡调整 (简化版: 偏暖/偏冷色调)"""
    arr = np.array(img, dtype=np.float32)
    # wb < 5500 偏暖(加红减蓝), wb > 5500 偏冷(加蓝减红)
    shift = (wb - 5500) / 5000.0  # [-0.7, 0.9]
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shift * 30, 0, 255)  # Blue
    arr[:, :, 0] = np.clip(arr[:, :, 0] - shift * 20, 0, 255)  # Red
    return Image.fromarray(arr.astype(np.uint8))


def create_comparison(img_path: str, pred: dict, gt: dict = None, size: int = 400) -> Image.Image:
    """创建原图 vs 调整后的对比图"""
    img = Image.open(img_path).convert('RGB')
    
    # 保持宽高比缩放
    w, h = img.size
    ratio = size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 应用参数
    adjusted = apply_ev_simulation(img_resized, pred['ev_compensation'])
    adjusted = apply_wb_simulation(adjusted, pred['white_balance'])
    
    # 拼接
    gap = 10
    text_h = 80
    canvas_w = new_w * 2 + gap
    canvas_h = new_h + text_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
    
    canvas.paste(img_resized, (0, 0))
    canvas.paste(adjusted, (new_w + gap, 0))
    
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_small = font
    
    # 标签
    y = new_h + 5
    draw.text((5, y), "Original", fill=(200, 200, 200), font=font)
    draw.text((new_w + gap + 5, y), "Adjusted", fill=(120, 255, 120), font=font)
    
    y += 20
    ev = pred['ev_compensation']
    wb = pred['white_balance']
    ev_sign = '+' if ev > 0 else ''
    draw.text((5, y), f"Predicted: EV={ev_sign}{ev:.2f}  WB={wb:.0f}K", 
              fill=(255, 220, 100), font=font_small)
    
    if gt:
        y += 16
        gt_ev = gt.get('ev_compensation', 0)
        gt_wb = gt.get('white_balance', 5500)
        gt_sign = '+' if gt_ev > 0 else ''
        draw.text((5, y), f"Expert:    EV={gt_sign}{gt_ev:.2f}  WB={gt_wb:.0f}K", 
                  fill=(150, 150, 255), font=font_small)
        
        y += 16
        ev_err = abs(ev - gt_ev)
        wb_err = abs(wb - gt_wb)
        draw.text((5, y), f"Error:     EV={ev_err:.2f}  WB={wb_err:.0f}K",
                  fill=(255, 150, 150), font=font_small)
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description='FiveK SemanticBridge Inference Demo')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--dir', type=str, help='Directory of images')
    parser.add_argument('--num', type=int, default=8, help='Number of random FiveK samples')
    parser.add_argument('--ckpt', type=str, 
                       default=str(PROJECT_ROOT / 'checkpoints' / 'fivek_semantic' / 'best.pt'))
    parser.add_argument('--output', type=str,
                       default=str(PROJECT_ROOT / 'outputs' / 'demo_predictions.png'))
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    model = load_model(args.ckpt, device)
    print(f"Model loaded from {args.ckpt}")
    
    # 收集图片
    images = []
    gt_params = {}
    
    if args.image:
        images = [args.image]
    elif args.dir:
        d = Path(args.dir)
        images = sorted([str(f) for f in d.glob('*.jpg')])[:args.num]
    else:
        # 随机抽 FiveK 样本 + 加载 GT
        jpeg_dir = Path(r'E:\dataset\fivek_jpeg')
        if not jpeg_dir.exists():
            print(f"FiveK JPEG not found at {jpeg_dir}, use --image or --dir")
            return
        
        # 加载专家参数做对比
        params_file = PROJECT_ROOT / 'data' / 'fivek_expert_params.json'
        if params_file.exists():
            with open(params_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            from collections import defaultdict
            img_means = defaultdict(lambda: defaultdict(list))
            for s in data['samples']:
                name = s['image_name'].replace('.dng', '')
                img_means[name]['ev'].append(s.get('ev_compensation', 0))
                img_means[name]['wb'].append(s.get('white_balance', 5500))
            
            for name in img_means:
                gt_params[name] = {
                    'ev_compensation': np.mean(img_means[name]['ev']),
                    'white_balance': np.mean(img_means[name]['wb']),
                }
        
        all_jpgs = sorted(jpeg_dir.glob('*.jpg'))
        random.seed(42)
        selected = random.sample(all_jpgs, min(args.num, len(all_jpgs)))
        images = [str(f) for f in selected]
    
    print(f"Processing {len(images)} images...")
    
    # 逐张推理
    results = []
    for img_path in images:
        pred = predict_single(model, img_path, device)
        name = Path(img_path).stem
        gt = gt_params.get(name, None)
        results.append({
            'image': img_path,
            'name': name,
            'prediction': pred,
            'ground_truth': gt,
        })
        
        ev = pred['ev_compensation']
        wb = pred['white_balance']
        ev_sign = '+' if ev > 0 else ''
        line = f"  {name}: EV={ev_sign}{ev:.2f}, WB={wb:.0f}K"
        if gt:
            line += f"  (GT: EV={gt['ev_compensation']:+.2f}, WB={gt['white_balance']:.0f}K)"
        print(line)
    
    # 生成对比图
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparisons = []
    for r in results:
        comp = create_comparison(r['image'], r['prediction'], r['ground_truth'])
        comparisons.append(comp)
    
    # 拼成网格
    if comparisons:
        cols = min(2, len(comparisons))
        rows = (len(comparisons) + cols - 1) // cols
        
        cw, ch = comparisons[0].size
        grid_gap = 6
        grid = Image.new('RGB', 
                        (cols * cw + (cols - 1) * grid_gap, rows * ch + (rows - 1) * grid_gap),
                        (20, 20, 20))
        
        for i, comp in enumerate(comparisons):
            r, c = divmod(i, cols)
            x = c * (cw + grid_gap)
            y = r * (ch + grid_gap)
            grid.paste(comp, (x, y))
        
        grid.save(str(output_path), quality=95)
        print(f"\nSaved comparison grid: {output_path}")
    
    # 汇总统计
    if any(r['ground_truth'] for r in results):
        ev_errs = []
        wb_errs = []
        for r in results:
            if r['ground_truth']:
                ev_errs.append(abs(r['prediction']['ev_compensation'] - r['ground_truth']['ev_compensation']))
                wb_errs.append(abs(r['prediction']['white_balance'] - r['ground_truth']['white_balance']))
        
        print(f"\nSummary ({len(ev_errs)} samples with GT):")
        print(f"  EV MAE:  {np.mean(ev_errs):.3f} (range [-3,+3])")
        print(f"  WB MAE:  {np.mean(wb_errs):.0f}K (range [2000,10000])")


if __name__ == '__main__':
    main()
