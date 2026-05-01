"""
批量生成评估图片: 原图 + Baseline增强 + Distill增强
生成后上传到 AutoDL, 用 Venus 做美学评价

用法:
  python tools/gen_eval_images.py --num_images 50
  python tools/gen_eval_images.py --num_images 100 --output_dir outputs/venus_eval
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


def load_model(model_type, device):
    if model_type == 'baseline':
        model = FiveK8ParamModel()
        state = torch.load('checkpoints/fivek_8param/best.pt',
                           map_location='cpu', weights_only=False)
        key = 'model_state_dict' if 'model_state_dict' in state else 'model'
        model.load_state_dict(state[key])
        return model.to(device).eval(), 'baseline'
    elif model_type == 'distill_v2':
        from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
        sa_model = SemanticDistillModel()
        sa = torch.load('checkpoints/semantic_distill/stage_a/best.pt',
                        map_location='cpu', weights_only=False)
        sa_model.load_state_dict(sa.get('model_state_dict', sa.get('model')))
        model = DistillParamModel(stage_a_model=sa_model)
        sb = torch.load('checkpoints/semantic_distill_v2/stage_b/best.pt',
                        map_location='cpu', weights_only=False)
        model.load_state_dict(sb.get('model_state_dict', sb.get('model')))
        return model.to(device).eval(), 'distill_v2'
    else:
        raise ValueError(f"Unknown: {model_type}")


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
    parser.add_argument('--data_file', default='data/fivek_expert_params.json')
    parser.add_argument('--jpeg_dir', default=r'E:\dataset\fivek_jpeg')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--output_dir', default='outputs/venus_eval')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 输出目录
    out_dir = Path(args.output_dir)
    (out_dir / 'original').mkdir(parents=True, exist_ok=True)
    (out_dir / 'baseline').mkdir(parents=True, exist_ok=True)
    (out_dir / 'distill_v2').mkdir(parents=True, exist_ok=True)

    # 验证集
    val_names = get_val_names(args.data_file)
    rng = np.random.RandomState(123)
    eval_names = list(rng.choice(val_names, min(args.num_images, len(val_names)), replace=False))

    # 加载模型
    models = {}
    for mt in ['baseline', 'distill_v2']:
        try:
            m, label = load_model(mt, device)
            models[label] = m
            print(f"加载模型: {label}")
        except Exception as e:
            print(f"跳过 {mt}: {e}")

    # 生成图片
    params_log = {}
    for name in tqdm(eval_names, desc="生成评估图片"):
        img_path = Path(args.jpeg_dir) / f"{name}.jpg"
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        # 保存原图 (resize 到 480p 统一尺寸)
        img_480 = img.copy()
        w, h = img_480.size
        if max(w, h) > 720:
            scale = 720 / max(w, h)
            img_480 = img_480.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img_480.save(out_dir / 'original' / f"{name}.jpg", quality=95)

        params_log[name] = {}

        # 各模型预测+渲染
        for label, model in models.items():
            pred_p = predict_params(model, img_tensor, device)
            img_enhanced = apply_lightroom_params(img, pred_p)

            # 统一尺寸
            w, h = img_enhanced.size
            if max(w, h) > 720:
                scale = 720 / max(w, h)
                img_enhanced = img_enhanced.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS)
            img_enhanced.save(out_dir / label / f"{name}.jpg", quality=95)
            params_log[name][label] = pred_p

    # 保存参数记录
    with open(out_dir / 'predicted_params.json', 'w') as f:
        json.dump(params_log, f, indent=2)

    print(f"\n生成完成!")
    print(f"  原图:       {out_dir / 'original'}")
    print(f"  Baseline:   {out_dir / 'baseline'}")
    print(f"  Distill v2: {out_dir / 'distill_v2'}")
    print(f"  参数记录:   {out_dir / 'predicted_params.json'}")
    print(f"  图片数:     {len(params_log)}")
    print(f"\n下一步: 打包上传到 AutoDL, 运行 venus_aesthetic_eval.py")


if __name__ == '__main__':
    main()
