"""
本地生成 Venus 评估用图
对 FiveK val 集随机抽取 N 张，用各模型增强后保存到目录结构，供上传 AutoDL 运行 Venus 评分。

输出目录结构:
  venus_eval/
    original/      ← 原图
    baseline/      ← Baseline 增强
    distill_v2/    ← Distill v2 增强
    distill_v4/    ← Distill v4 增强
    distill_v5/    ← Distill v5 增强

用法:
  python tools/data/generate_venus_eval_images.py --num_images 50
  python tools/data/generate_venus_eval_images.py --models baseline distill_v5 --num_images 50
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
from training.fivek_8param.config import PARAM_NAMES
from training.fivek_8param.model import FiveK8ParamModel


# ─────────────────────────── 模型加载 ───────────────────────────

def _load_distill(stage_a_ckpt, stage_b_ckpt):
    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    sa = SemanticDistillModel()
    st_a = torch.load(stage_a_ckpt, map_location='cpu', weights_only=False)
    sa.load_state_dict(st_a.get('model_state_dict', st_a.get('model')))
    model = DistillParamModel(stage_a_model=sa)
    st_b = torch.load(stage_b_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(st_b.get('model_state_dict', st_b.get('model')))
    return model


def load_model(model_type, device):
    if model_type == 'baseline':
        model = FiveK8ParamModel()
        st = torch.load('checkpoints/fivek_8param/best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(st.get('model_state_dict', st.get('model')))
        label = 'baseline'
    elif model_type == 'distill_v2':
        model = _load_distill('checkpoints/semantic_distill/stage_a/best.pt',
                               'checkpoints/semantic_distill_v2/stage_b/best.pt')
        label = 'distill_v2'
    elif model_type in ('distill_v4', 'distill_v5'):
        ver = model_type.split('_')[1]
        model = _load_distill(f'checkpoints/distill_{ver}/stage_a/best.pt',
                               f'checkpoints/distill_{ver}/stage_b/best.pt')
        label = model_type
    else:
        raise ValueError(f"Unknown: {model_type}")
    model.to(device).eval()
    return model, label


# ─────────────────────────── 数据加载 ───────────────────────────

def get_val_names(data_file, val_ratio=0.1, seed=42):
    with open(data_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    samples = raw['samples'] if isinstance(raw, dict) and 'samples' in raw else raw
    names = [item['image_name'].replace('.dng', '') for item in samples]
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    return names[-int(len(names) * val_ratio):]


def predict_params(model, img: Image.Image, transform, device):
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    return {p: out['raw_params'][p].squeeze().cpu().item() for p in PARAM_NAMES}


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['baseline', 'distill_v2', 'distill_v4', 'distill_v5'],
                        choices=['baseline', 'distill_v2', 'distill_v4', 'distill_v5'],
                        help='要生成的模型列表')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--data_file', default='data/fivek_expert_params.json')
    parser.add_argument('--jpeg_dir', default=r'E:\dataset\fivek_jpeg')
    parser.add_argument('--output_dir', default='outputs/data/venus_eval')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 抽取图片
    val_names = get_val_names(args.data_file)
    rng = np.random.RandomState(args.seed)
    eval_names = list(rng.choice(val_names, min(args.num_images, len(val_names)), replace=False))
    print(f"评估图片: {len(eval_names)} 张")

    out_dir = Path(args.output_dir)

    # ── 保存原图 ──
    orig_dir = out_dir / 'original'
    orig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n保存原图 → {orig_dir}")
    saved_names = []
    for name in tqdm(eval_names, ncols=80):
        img_path = Path(args.jpeg_dir) / f"{name}.jpg"
        if not img_path.exists():
            continue
        orig = Image.open(img_path).convert('RGB')
        orig.save(orig_dir / f"{name}.jpg", quality=95)
        saved_names.append(name)

    print(f"原图保存: {len(saved_names)} 张")

    # ── 各模型增强图 ──
    for model_type in args.models:
        print(f"\n[{model_type}] 加载模型...")
        try:
            model, label = load_model(model_type, device)
        except Exception as e:
            print(f"  跳过 {model_type}: {e}")
            continue

        save_dir = out_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        # 跳过已生成的
        existing = {f.stem for f in save_dir.glob('*.jpg')}
        todo = [n for n in saved_names if n not in existing]
        if not todo:
            print(f"  [{label}] 已全部生成，跳过")
            continue

        print(f"  [{label}] 生成 {len(todo)} 张 → {save_dir}")
        for name in tqdm(todo, desc=f'[{label}]', ncols=80):
            img_path = Path(args.jpeg_dir) / f"{name}.jpg"
            orig = Image.open(img_path).convert('RGB')
            params = predict_params(model, orig, transform, device)
            enhanced = apply_lightroom_params(orig, params)
            enhanced.save(save_dir / f"{name}.jpg", quality=95)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 输出打包提示 ──
    print(f"\n{'='*60}")
    print(f"  图片已生成: {out_dir}")
    print(f"  上传 AutoDL 命令:")
    print(f"    # 先压缩")
    print(f"    cd {out_dir.parent}")
    print(f"    zip -r venus_eval.zip venus_eval/")
    print(f"    # 上传 (AutoDL JupyterLab 拖拽 或 scp)")
    print(f"    # AutoDL 解压后运行:")
    print(f"    python venus_aesthetic_eval.py \\")
    print(f"      --eval_dir /root/venus_eval \\")
    print(f"      --groups original baseline distill_v2 distill_v4 distill_v5 \\")
    print(f"      --max_images {len(saved_names)} \\")
    print(f"      --output /root/venus_eval_results_all.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
