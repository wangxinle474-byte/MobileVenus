"""
FiveK 多专家一致性评估
评估模型预测与不同标注专家之间的 PSNR/SSIM 一致性
用法:
  python tools/eval_multi_expert.py --model distill_v4
  python tools/eval_multi_expert.py --model all
"""
import sys, json, argparse, warnings
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import piq

from models.isp_pipeline import apply_lightroom_params
from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.fivek_8param.model import FiveK8ParamModel
from training.semantic_distill.model import DistillParamModel


EXPERT_LABELS = {
    'default': 'Expert-Default',
    '42962A54-F9BA-11DB-B851-000D93313A24': 'Expert-A',
    'ED7AD140-FA03-11DB-AB5E-00145166C8C8': 'Expert-B',
}


def load_all_expert_params(data_file: str):
    """
    按图片名 + 专家分组加载所有参数
    返回 {image_name: {expert_label: {param: val}}}
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    samples = raw['samples'] if isinstance(raw, dict) and 'samples' in raw else raw

    per_image = defaultdict(dict)
    for item in samples:
        name = item['image_name'].replace('.dng', '')
        expert_id = item.get('expert', 'default')
        label = EXPERT_LABELS.get(expert_id, expert_id[:8])
        params = {p: float(item.get(p, 0.0)) for p in PARAM_NAMES}
        per_image[name][label] = params

    return dict(per_image)


def get_val_split(data_file: str, val_ratio=0.1, seed=42):
    with open(data_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    samples = raw['samples'] if isinstance(raw, dict) and 'samples' in raw else raw
    # 只用 default expert 的图片做 val split（同 eval_psnr_ssim.py）
    names = list({item['image_name'].replace('.dng', '')
                  for item in samples if item.get('expert', 'default') == 'default'})
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    n_val = int(len(names) * val_ratio)
    return names[-n_val:]


def _load_distill_model(stage_a_ckpt, stage_b_ckpt):
    from training.semantic_distill.model import SemanticDistillModel
    stage_a_model = SemanticDistillModel()
    state_a = torch.load(stage_a_ckpt, map_location='cpu', weights_only=False)
    stage_a_model.load_state_dict(state_a.get('model_state_dict', state_a.get('model')))
    model = DistillParamModel(stage_a_model=stage_a_model)
    state_b = torch.load(stage_b_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state_b.get('model_state_dict', state_b.get('model')))
    return model


def load_model(model_type, device):
    if model_type == 'baseline':
        model = FiveK8ParamModel()
        state = torch.load('checkpoints/fivek_8param/best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state.get('model')))
        label = 'Baseline'
    elif model_type == 'distill_v2':
        model = _load_distill_model('checkpoints/semantic_distill/stage_a/best.pt',
                                    'checkpoints/semantic_distill_v2/stage_b/best.pt')
        label = 'Distill v2'
    elif model_type == 'distill_v4':
        model = _load_distill_model('checkpoints/distill_v4/stage_a/best.pt',
                                    'checkpoints/distill_v4/stage_b/best.pt')
        label = 'Distill v4'
    elif model_type == 'distill_v5':
        model = _load_distill_model('checkpoints/distill_v5/stage_a/best.pt',
                                    'checkpoints/distill_v5/stage_b/best.pt')
        label = 'Distill v5'
    else:
        raise ValueError(f"Unknown model: {model_type}")
    model.to(device).eval()
    return model, label


def predict_params(model, img_tensor, device):
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
    return {p: out['raw_params'][p].squeeze().cpu().item() for p in PARAM_NAMES}


def compute_metrics(img_gt, img_pred, device):
    a, b = np.array(img_gt), np.array(img_pred)
    p = psnr(a, b, data_range=255)
    s = ssim(a, b, channel_axis=2, data_range=255)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tg = torch.from_numpy(a.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
        tp = torch.from_numpy(b.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
        ms = piq.multi_scale_ssim(tg, tp, data_range=1.0).item()
    return p, s, ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all',
                        choices=['baseline', 'distill_v2', 'distill_v4', 'distill_v5', 'all'])
    parser.add_argument('--data_file', default='data/fivek_expert_params.json')
    parser.add_argument('--jpeg_dir', default=r'E:\dataset\fivek_jpeg')
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--output', default='outputs/eval/eval_multi_expert.txt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_expert_params = load_all_expert_params(args.data_file)
    val_names = get_val_split(args.data_file)

    # 优先选有多个专家标注的图片
    multi_expert_imgs = [n for n in val_names if len(all_expert_params.get(n, {})) > 1]
    single_expert_imgs = [n for n in val_names if len(all_expert_params.get(n, {})) == 1]

    rng = np.random.RandomState(123)
    rng.shuffle(multi_expert_imgs)
    rng.shuffle(single_expert_imgs)

    # 尽量多用多专家图片
    eval_names = multi_expert_imgs[:args.num_images]
    if len(eval_names) < args.num_images:
        eval_names += single_expert_imgs[:args.num_images - len(eval_names)]
    eval_names = eval_names[:args.num_images]

    all_expert_labels = sorted({lbl for n in eval_names
                                 for lbl in all_expert_params.get(n, {}).keys()})

    print(f"多专家一致性评估 | 图片数: {len(eval_names)}")
    print(f"专家标注者: {all_expert_labels}")
    print(f"多专家覆盖图片: {len(multi_expert_imgs)} / {len(val_names)} val images")
    print()

    if args.model == 'all':
        model_types = ['baseline', 'distill_v2']
        if Path('checkpoints/distill_v4/stage_b/best.pt').exists():
            model_types.append('distill_v4')
        if Path('checkpoints/distill_v5/stage_b/best.pt').exists():
            model_types.append('distill_v5')
    else:
        model_types = [args.model]

    all_results = {}

    for mt in model_types:
        try:
            model, label = load_model(mt, device)
        except Exception as e:
            print(f"跳过 {mt}: {e}")
            continue

        # per_expert_metrics[expert_label] = [(psnr, ssim, ms_ssim), ...]
        per_expert = {el: {'psnr': [], 'ssim': [], 'ms': []} for el in all_expert_labels}
        n_done = 0

        for name in eval_names:
            img_path = Path(args.jpeg_dir) / f"{name}.jpg"
            if not img_path.exists():
                continue
            experts_for_img = all_expert_params.get(name, {})
            if not experts_for_img:
                continue

            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            pred_p = predict_params(model, img_tensor, device)

            # 预测渲染
            img_pred = apply_lightroom_params(img, pred_p)

            for expert_lbl, gt_p in experts_for_img.items():
                if expert_lbl not in per_expert:
                    continue
                img_gt = apply_lightroom_params(img, gt_p)
                p_val, s_val, ms_val = compute_metrics(img_gt, img_pred, device)
                per_expert[expert_lbl]['psnr'].append(p_val)
                per_expert[expert_lbl]['ssim'].append(s_val)
                per_expert[expert_lbl]['ms'].append(ms_val)

            n_done += 1
            if n_done % 50 == 0:
                print(f"  [{label}] {n_done}/{len(eval_names)} done...")

        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        print(f"  {'专家':<20} {'N':>5} {'PSNR':>8} {'SSIM':>8} {'MS-SSIM':>9}")
        print(f"  {'-'*55}")

        model_results = {}
        for el in sorted(per_expert.keys()):
            m = per_expert[el]
            if not m['psnr']:
                continue
            mp = np.mean(m['psnr'])
            ms = np.mean(m['ssim'])
            mm = np.mean(m['ms'])
            n = len(m['psnr'])
            print(f"  {el:<20} {n:>5} {mp:>8.2f} {ms:>8.4f} {mm:>9.4f}")
            model_results[el] = {'psnr': mp, 'ssim': ms, 'ms_ssim': mm, 'n': n}

        # 计算专家间 PSNR 标准差（一致性指标）
        psnr_vals = [model_results[el]['psnr'] for el in model_results if model_results[el]['n'] > 10]
        if len(psnr_vals) > 1:
            print(f"  {'─'*55}")
            print(f"  专家间 PSNR std: {np.std(psnr_vals):.3f} dB  "
                  f"(越小=模型对各专家越一致)")

        all_results[label] = model_results

    # 保存文本结果
    lines = ['FiveK 多专家一致性评估\n', f'图片数: {len(eval_names)}\n\n']
    for model_label, res in all_results.items():
        lines.append(f'=== {model_label} ===\n')
        for el, m in res.items():
            lines.append(f"  {el}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}, "
                         f"MS-SSIM={m['ms_ssim']:.4f} (n={m['n']})\n")
        lines.append('\n')
    Path(args.output).write_text(''.join(lines), encoding='utf-8')
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
