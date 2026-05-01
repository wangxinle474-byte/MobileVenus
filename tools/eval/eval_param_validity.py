"""
参数有效性验证: 用 AADB 美学评分器独立评估
证明预测参数确实提升了图像质量

逻辑:
  原图 → 美学分 A
  原图 + 预测参数 → ISP渲染 → 美学分 B
  B > A → 参数有效

用法:
  python tools/eval_param_validity.py --num_images 200
  python tools/eval_param_validity.py --model distill_v2 --num_images 500
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.isp_pipeline import apply_lightroom_params
from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.fivek_8param.model import FiveK8ParamModel
from training.train_aadb_aesthetic import AestheticModel


def load_param_model(model_type, device):
    """加载参数预测模型"""
    if model_type == 'baseline':
        model = FiveK8ParamModel()
        state = torch.load('checkpoints/fivek_8param/best.pt',
                           map_location='cpu', weights_only=False)
        key = 'model_state_dict' if 'model_state_dict' in state else 'model'
        model.load_state_dict(state[key])
        label = 'Baseline'
    elif model_type == 'distill_v2':
        from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
        stage_a = SemanticDistillModel()
        sa = torch.load('checkpoints/semantic_distill/stage_a/best.pt',
                        map_location='cpu', weights_only=False)
        stage_a.load_state_dict(sa.get('model_state_dict', sa.get('model')))
        model = DistillParamModel(stage_a_model=stage_a)
        sb = torch.load('checkpoints/semantic_distill_v2/stage_b/best.pt',
                        map_location='cpu', weights_only=False)
        model.load_state_dict(sb.get('model_state_dict', sb.get('model')))
        label = 'Distill v2'
    elif model_type in ('distill_v4', 'distill_v5'):
        from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
        ver = model_type.split('_')[1]  # 'v4' or 'v5'
        stage_a = SemanticDistillModel()
        sa = torch.load(f'checkpoints/distill_{ver}/stage_a/best.pt',
                        map_location='cpu', weights_only=False)
        stage_a.load_state_dict(sa.get('model_state_dict', sa.get('model')))
        model = DistillParamModel(stage_a_model=stage_a)
        sb = torch.load(f'checkpoints/distill_{ver}/stage_b/best.pt',
                        map_location='cpu', weights_only=False)
        model.load_state_dict(sb.get('model_state_dict', sb.get('model')))
        label = f'Distill {ver.upper()}'
    else:
        raise ValueError(f"Unknown: {model_type}")
    model.to(device).eval()
    return model, label


def load_aesthetic_model(device):
    """加载 AADB 美学评分模型"""
    ckpt = Path('checkpoints/aadb_aesthetic_full/best.pt')
    model = AestheticModel(image_size=224)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    return model


def aesthetic_score(aes_model, img: Image.Image, transform, device):
    """给单张图片打美学分"""
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = aes_model(t)
    return {
        'overall': out['weighted_score'].item(),
        'dims': out['scores'].squeeze().cpu().numpy(),  # (5,)
    }


def predict_params(model, img_tensor, device):
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
    return {p: out['raw_params'][p].squeeze().cpu().item() for p in PARAM_NAMES}


def get_val_names(data_file, val_ratio=0.1, seed=42):
    with open(data_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    samples = raw['samples'] if isinstance(raw, dict) and 'samples' in raw else raw
    names = [item['image_name'].replace('.dng', '') for item in samples]
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    n_val = int(len(names) * val_ratio)
    return names[-n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='baseline',
                        choices=['baseline', 'distill_v2', 'distill_v4', 'distill_v5', 'all'])
    parser.add_argument('--data_file', default='data/fivek_expert_params.json')
    parser.add_argument('--jpeg_dir', default=r'E:\dataset\fivek_jpeg')
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--output', default='outputs/eval/param_validity.txt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 加载美学评分器
    print("加载 AADB 美学评分模型...")
    aes_model = load_aesthetic_model(device)

    # 获取验证集
    val_names = get_val_names(args.data_file)
    rng = np.random.RandomState(123)
    eval_names = list(rng.choice(val_names,
                                 min(args.num_images, len(val_names)),
                                 replace=False))
    print(f"评估 {len(eval_names)} 张验证集图片\n")

    model_types = ['baseline', 'distill_v2', 'distill_v4', 'distill_v5'] if args.model == 'all' else [args.model]
    dim_names = ['composition', 'lighting', 'color', 'clarity', 'subject']

    all_results = {}

    for mt in model_types:
        print(f"{'='*60}")
        try:
            param_model, label = load_param_model(mt, device)
        except Exception as e:
            print(f"跳过 {mt}: {e}")
            continue
        print(f"  参数预测模型: {label}")
        print(f"{'='*60}\n")

        scores_before = []  # 原图美学分
        scores_after = []   # 调参后美学分
        scores_dims_before = []
        scores_dims_after = []
        predicted_params_all = []

        for name in tqdm(eval_names, desc=f"[{label}] 评估中"):
            img_path = Path(args.jpeg_dir) / f"{name}.jpg"
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)

            # 1. 原图美学分
            s_before = aesthetic_score(aes_model, img, transform, device)
            scores_before.append(s_before['overall'])
            scores_dims_before.append(s_before['dims'])

            # 2. 预测参数 → ISP渲染 → 美学分
            pred_p = predict_params(param_model, img_tensor, device)
            img_enhanced = apply_lightroom_params(img, pred_p)
            s_after = aesthetic_score(aes_model, img_enhanced, transform, device)
            scores_after.append(s_after['overall'])
            scores_dims_after.append(s_after['dims'])
            predicted_params_all.append(pred_p)

        # === 统计 ===
        before = np.array(scores_before)
        after = np.array(scores_after)
        improved = (after > before).sum()
        total = len(before)

        print(f"\n{'='*60}")
        print(f"  {label} — 参数有效性验证结果")
        print(f"{'='*60}")
        print(f"\n  === 总体美学分 ===")
        print(f"  原图平均:     {before.mean():.3f} (std={before.std():.3f})")
        print(f"  调参后平均:   {after.mean():.3f} (std={after.std():.3f})")
        print(f"  平均提升:     {(after-before).mean():+.3f}")
        print(f"  提升率:       {improved}/{total} ({100*improved/total:.1f}%)")
        print()

        # 各维度分析
        dims_b = np.array(scores_dims_before)  # (N, 5)
        dims_a = np.array(scores_dims_after)   # (N, 5)
        print(f"  === 各维度美学分提升 ===")
        print(f"  {'维度':<15} {'原图':>8} {'调参后':>8} {'提升':>8} {'提升率':>8}")
        print(f"  {'-'*50}")
        for i, dname in enumerate(dim_names):
            b_mean = dims_b[:, i].mean()
            a_mean = dims_a[:, i].mean()
            delta = a_mean - b_mean
            imp = (dims_a[:, i] > dims_b[:, i]).sum()
            print(f"  {dname:<15} {b_mean:>8.3f} {a_mean:>8.3f} {delta:>+8.3f} "
                  f"{100*imp/total:>7.1f}%")

        # 参数统计
        print(f"\n  === 预测参数统计 ===")
        print(f"  {'参数':<20} {'均值':>8} {'标准差':>8} {'范围':>18}")
        print(f"  {'-'*56}")
        for p in PARAM_NAMES:
            vals = [pp[p] for pp in predicted_params_all]
            lo, hi = PARAM_RANGES[p]
            print(f"  {p:<20} {np.mean(vals):>8.2f} {np.std(vals):>8.2f} "
                  f"[{np.min(vals):>6.1f}, {np.max(vals):>6.1f}]")

        all_results[label] = {
            'before_mean': float(before.mean()),
            'after_mean': float(after.mean()),
            'delta': float((after - before).mean()),
            'improve_rate': float(improved / total),
            'n': total,
        }
        print()

    # === 模型间对比 ===
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  模型间对比")
        print(f"{'='*60}")
        print(f"  {'模型':<15} {'原图':>8} {'调参后':>8} {'提升':>8} {'提升率':>8}")
        print(f"  {'-'*50}")
        for label, r in all_results.items():
            print(f"  {label:<15} {r['before_mean']:>8.3f} {r['after_mean']:>8.3f} "
                  f"{r['delta']:>+8.3f} {100*r['improve_rate']:>7.1f}%")

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("参数有效性验证结果\n\n")
        for label, r in all_results.items():
            f.write(f"{label}:\n")
            f.write(f"  原图美学分: {r['before_mean']:.3f}\n")
            f.write(f"  调参后:     {r['after_mean']:.3f}\n")
            f.write(f"  提升:       {r['delta']:+.3f}\n")
            f.write(f"  提升率:     {100*r['improve_rate']:.1f}%\n\n")
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
