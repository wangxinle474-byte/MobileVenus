"""高分辨率评估: 模型224预测参数, diff_isp在512渲染"""
import sys, json, logging
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/root/autodl-tmp')
sys.path.insert(0, str(ROOT / 'IntelligenceCamera'))

EVAL_DIR = ROOT / 'unified_eval_hires'
JPEG_DIR = ROOT / 'fivek_jpeg'
CKPT_DIR = ROOT / 'checkpoints'

def load_model(version, device):
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
    if not sb_path.exists(): return None
    sb_state = torch.load(str(sb_path), map_location='cpu', weights_only=False)
    model.load_state_dict(sb_state['model_state_dict'])
    model.to(device).eval()
    return model

def generate_hires(model, names, jpeg_dir, out_dir, device):
    """224预测参数, 512渲染"""
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    from models.diff_isp import apply_diff_isp
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in out_dir.glob('*.jpg'))
    todo = [n for n in names if n not in existing]
    if not todo: logger.info('  skip'); return

    model_tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    hires_tf = T.Compose([T.Resize((512,512)), T.ToTensor()])  # 512渲染!

    def denorm(params_norm):
        result = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            result[name] = lo + (params_norm[:, i] + 1.0) / 2.0 * (hi - lo)
        return result

    for name in todo:
        p = jpeg_dir / f'{name}.jpg'
        if not p.exists(): continue
        pil = Image.open(p).convert('RGB')
        t = model_tf(pil).unsqueeze(0).to(device)
        raw_hires = hires_tf(pil).unsqueeze(0).to(device)  # 512x512

        with torch.no_grad():
            out = model(t)
            params = denorm(out['params_norm'])
            enhanced = apply_diff_isp(raw_hires.float(), params).clamp(0, 1)
            enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0)

        arr = (enhanced[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir/f'{name}.jpg', quality=95)
    logger.info(f'  generated {len(todo)} @ 512x512')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    orig_dir = EVAL_DIR / 'original'
    old_orig = ROOT / 'unified_eval' / 'original'
    if old_orig.exists() and not orig_dir.exists():
        import shutil; shutil.copytree(str(old_orig), str(orig_dir))

    names = sorted(f.stem for f in orig_dir.glob('*.jpg'))
    logger.info(f'{len(names)} images')

    groups = {'original': orig_dir}
    for ver in ['v6','v7','v8','v9']:
        logger.info(f'--- {ver} ---')
        model = load_model(ver, device)
        if model is None: continue
        out_dir = EVAL_DIR / f'{ver}_hires'
        generate_hires(model, names, JPEG_DIR, out_dir, device)
        groups[f'{ver}_hires'] = out_dir
        # 旧的224结果也加入对比
        old_dir = ROOT / 'unified_eval' / f'distill_{ver}'
        if old_dir.exists(): groups[f'{ver}_224'] = old_dir
        del model; torch.cuda.empty_cache()

    # MUSIQ
    import pyiqa
    metric_fn = pyiqa.create_metric('musiq-ava', device=device)
    to_tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])

    logger.info('\n' + '='*60)
    logger.info('  MUSIQ: 224 render vs 512 render')
    logger.info('='*60)
    results = {}
    for gname, gdir in groups.items():
        scores = []
        for name in names:
            fp = gdir / f'{name}.jpg'
            if not fp.exists(): continue
            try:
                t = to_tensor(Image.open(fp).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad(): s = metric_fn(t).item()
                scores.append(s)
            except: continue
        if scores: results[gname] = {'mean':float(np.mean(scores)), 'std':float(np.std(scores))}

    orig_m = results.get('original',{}).get('mean',0)
    logger.info(f"{'Group':<20} {'Mean':>8} {'Std':>8} {'Delta':>8}")
    logger.info('-'*48)
    for g in groups:
        if g in results:
            r = results[g]
            d = r['mean']-orig_m if g!='original' else 0
            logger.info(f"{g:<20} {r['mean']:>8.3f} {r['std']:>8.3f} {d:>+8.3f}" if g!='original' else f"{g:<20} {r['mean']:>8.3f} {r['std']:>8.3f}")

    json.dump(results, open(EVAL_DIR/'results.json','w'), indent=2)
    logger.info(f'Saved: {EVAL_DIR/"results.json"}')

if __name__ == '__main__':
    main()
