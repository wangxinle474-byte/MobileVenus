"""Step 1: Expert C 天花板评估 — 看专家图 MUSIQ 上限"""
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
EXPERT_C = ROOT / 'fivek_expert_c'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 用之前的50张评估图
    orig_dir = EVAL_DIR / 'original'
    names = sorted(f.stem for f in orig_dir.glob('*.jpg'))
    logger.info(f'{len(names)} eval images')
    
    # 准备 Expert C 512x512
    ec_dir = EVAL_DIR / 'expert_c'
    ec_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in ec_dir.glob('*.jpg'))
    for name in names:
        if name in existing: continue
        # 尝试多种命名格式
        candidates = [
            EXPERT_C / f'{name}.jpg',
            EXPERT_C / f'{name}.png',
            EXPERT_C / f'{name}.tif',
        ]
        found = None
        for c in candidates:
            if c.exists(): found = c; break
        if found is None:
            # 试试不区分大小写
            for f in EXPERT_C.iterdir():
                if f.stem.lower() == name.lower():
                    found = f; break
        if found:
            Image.open(found).convert('RGB').resize((512,512)).save(ec_dir/f'{name}.jpg', quality=95)
    
    ec_count = len(list(ec_dir.glob('*.jpg')))
    logger.info(f'Expert C: {ec_count}/{len(names)} found')
    
    if ec_count == 0:
        logger.error('No Expert C images found! Check fivek_expert_c directory')
        # 列出目录内容
        if EXPERT_C.exists():
            files = list(EXPERT_C.iterdir())[:10]
            logger.info(f'fivek_expert_c contains: {[f.name for f in files]}...')
            logger.info(f'Total files: {len(list(EXPERT_C.iterdir()))}')
        else:
            logger.error(f'{EXPERT_C} does not exist!')
        return
    
    # MUSIQ 评估: original vs expert_c vs 各模型版本
    import pyiqa
    metric_fn = pyiqa.create_metric('musiq-ava', device=device)
    to_tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])
    
    groups = {
        'original': orig_dir,
        'expert_c': ec_dir,
    }
    # 加入之前的 hires 结果
    for ver in ['v6','v7','v8','v9']:
        hires_dir = EVAL_DIR / f'{ver}_hires'
        if hires_dir.exists():
            groups[f'{ver}_hires'] = hires_dir
    
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
        if scores:
            results[gname] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'n': len(scores)}
    
    # 打印
    logger.info('\n' + '='*60)
    logger.info('  MUSIQ AVA: Expert C 天花板 vs 模型')
    logger.info('='*60)
    orig_m = results.get('original',{}).get('mean', 0)
    ec_m = results.get('expert_c',{}).get('mean', 0)
    logger.info(f"{'Group':<20} {'Mean':>8} {'Std':>8} {'vs Orig':>10} {'vs ExpC':>10}")
    logger.info('-'*60)
    for g in groups:
        if g in results:
            r = results[g]
            d_orig = r['mean'] - orig_m
            d_ec = r['mean'] - ec_m if ec_m > 0 else 0
            d_orig_s = f"{d_orig:+.3f}" if g != 'original' else ''
            d_ec_s = f"{d_ec:+.3f}" if g != 'expert_c' else ''
            logger.info(f"{g:<20} {r['mean']:>8.3f} {r['std']:>8.3f} {d_orig_s:>10} {d_ec_s:>10}")
    
    # 关键指标
    if ec_m > 0:
        logger.info(f'\n>>> Expert C 天花板: {ec_m:.3f}')
        logger.info(f'>>> 原图基线: {orig_m:.3f}')
        logger.info(f'>>> 专家提升空间: {ec_m - orig_m:+.3f}')
        best_model = max([(g, results[g]['mean']) for g in results if 'hires' in g], key=lambda x: x[1], default=None)
        if best_model:
            logger.info(f'>>> 当前最佳模型: {best_model[0]} = {best_model[1]:.3f} (距天花板 {best_model[1]-ec_m:+.3f})')
    
    json.dump(results, open(EVAL_DIR/'ceiling_results.json','w'), indent=2)

if __name__ == '__main__':
    main()
