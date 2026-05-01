"""用 Neural ISP 替代 diff_isp 评估各版本模型"""
import sys, json, logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/root/autodl-tmp')
CODE_DIR = ROOT / 'IntelligenceCamera'
sys.path.insert(0, str(CODE_DIR))

EVAL_DIR   = ROOT / 'unified_eval_nisp'
JPEG_DIR   = ROOT / 'fivek_jpeg'
CKPT_DIR   = ROOT / 'checkpoints'
NISP_CKPT  = CKPT_DIR / 'neural_isp' / 'best.pt'


def load_param_model(version, device):
    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.semantic_distill.config import DistillConfig
    cfg = DistillConfig()
    sa_path = CKPT_DIR / 'distill_v6' / 'stage_a' / 'best.pt'
    stage_a = SemanticDistillModel(image_size=cfg.image_size, visual_dim=cfg.visual_dim,
                                    semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim)
    sa_state = torch.load(str(sa_path), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])
    model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)
    sb_path = CKPT_DIR / f'distill_{version}' / 'stage_b' / 'best.pt'
    if not sb_path.exists():
        logger.warning(f'{version} not found: {sb_path}')
        return None
    sb_state = torch.load(str(sb_path), map_location='cpu', weights_only=False)
    model.load_state_dict(sb_state['model_state_dict'])
    model.to(device).eval()
    logger.info(f'{version} loaded')
    return model


def load_neural_isp(device):
    from models.neural_isp import NeuralISP
    ckpt = torch.load(str(NISP_CKPT), map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    nisp = NeuralISP(param_dim=cfg['param_dim'], base_ch=cfg['base_ch'],
                     n_res_blocks=cfg['n_res_blocks'])
    nisp.load_state_dict(ckpt['model_state_dict'])
    nisp.to(device).eval()
    logger.info(f'NeuralISP loaded (val_loss={ckpt.get("val_loss","?")}, aes={ckpt.get("val_aes","?")})')
    return nisp


def generate_with_nisp(param_model, nisp, names, jpeg_dir, out_dir, device):
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in out_dir.glob('*.jpg'))
    todo = [n for n in names if n not in existing]
    if not todo:
        logger.info('  already done, skip'); return

    transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    raw_transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

    def to_norm_params(raw_params):
        """raw_params (B, 6) physical → normalized [-1,1]"""
        result = []
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            result.append((raw_params[:, i] - lo) / (hi - lo) * 2.0 - 1.0)
        return torch.stack(result, dim=1)

    def denorm(params_norm):
        result = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            result[name] = lo + (params_norm[:, i] + 1.0) / 2.0 * (hi - lo)
        return result

    for name in todo:
        img_path = jpeg_dir / f'{name}.jpg'
        if not img_path.exists(): continue
        pil = Image.open(img_path).convert('RGB')
        t = transform(pil).unsqueeze(0).to(device)
        raw = raw_transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = param_model(t)
            params_norm = out['params_norm']  # (1, 6) [-1,1]
            enhanced = nisp(raw.float(), params_norm)
            enhanced = enhanced.clamp(0, 1)

        arr = (enhanced[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir/f'{name}.jpg', quality=95)
    logger.info(f'  generated {len(todo)}')


def evaluate_musiq(groups, names, device):
    try:
        import pyiqa
    except: logger.warning('pyiqa not installed'); return {}
    metric_fn = pyiqa.create_metric('musiq-ava', device=device)
    to_tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])
    results = {}
    for gname, gdir in groups.items():
        scores = []
        for name in names:
            p = gdir / f'{name}.jpg'
            if not p.exists(): continue
            try:
                t = to_tensor(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad(): s = metric_fn(t).item()
                scores.append(s)
            except: continue
        if scores:
            results[gname] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'n': len(scores)}
    del metric_fn; torch.cuda.empty_cache()
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备原图
    orig_dir = EVAL_DIR / 'original'
    old_orig = ROOT / 'unified_eval' / 'original'
    if old_orig.exists() and not orig_dir.exists():
        import shutil; shutil.copytree(str(old_orig), str(orig_dir))
    elif not orig_dir.exists():
        orig_dir.mkdir(parents=True, exist_ok=True)
        import random; random.seed(42)
        all_jpgs = sorted(JPEG_DIR.glob('*.jpg'))
        selected = random.sample(all_jpgs, min(50, len(all_jpgs)))
        for src in selected:
            Image.open(src).convert('RGB').resize((512,512)).save(orig_dir/src.name, quality=95)

    names = sorted(f.stem for f in orig_dir.glob('*.jpg'))
    logger.info(f'{len(names)} eval images')

    nisp = load_neural_isp(device)
    versions = ['v6','v7','v8','v9']
    groups = {'original': orig_dir}

    # diff_isp 结果 (旧)
    for ver in versions:
        old_dir = ROOT / 'unified_eval' / f'distill_{ver}'
        if old_dir.exists():
            groups[f'{ver}_diff_isp'] = old_dir

    # Neural ISP 结果 (新)
    for ver in versions:
        model = load_param_model(ver, device)
        if model is None: continue
        out_dir = EVAL_DIR / f'{ver}_nisp'
        generate_with_nisp(model, nisp, names, JPEG_DIR, out_dir, device)
        groups[f'{ver}_nisp'] = out_dir
        del model; torch.cuda.empty_cache()

    # MUSIQ
    logger.info('\n' + '='*60)
    logger.info('  MUSIQ AVA: diff_isp vs Neural ISP')
    logger.info('='*60)
    results = evaluate_musiq(groups, names, device)

    orig_mean = results.get('original',{}).get('mean', 0)
    logger.info(f"{'Group':<20} {'Mean':>8} {'Std':>8} {'Delta':>8}")
    logger.info('-'*48)
    for gname in groups:
        if gname in results:
            r = results[gname]
            d = r['mean'] - orig_mean if gname != 'original' else 0
            d_str = f'{d:+.3f}' if gname != 'original' else ''
            logger.info(f"{gname:<20} {r['mean']:>8.3f} {r['std']:>8.3f} {d_str:>8}")

    json.dump(results, open(EVAL_DIR/'results.json','w'), indent=2)
    logger.info(f'\nSaved: {EVAL_DIR/"results.json"}')

if __name__ == '__main__':
    main()
