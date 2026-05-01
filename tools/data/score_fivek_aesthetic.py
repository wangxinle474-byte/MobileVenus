"""
用 AADB 美学评分器给 FiveK JPEG 图片打分
输出: data/fivek_aesthetic_scores.json
"""
import sys
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_aadb_aesthetic import AestheticModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 加载 AADB 美学评分模型
    ckpt_path = PROJECT_ROOT / 'checkpoints' / 'aadb_aesthetic_full' / 'best.pt'
    if not ckpt_path.exists():
        logger.error(f"找不到 AADB 权重: {ckpt_path}")
        return
    
    model = AestheticModel(image_size=224)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    logger.info(f"已加载 AADB 模型 (epoch {state.get('epoch', '?')})")
    
    # 图片变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 扫描 FiveK JPEG
    jpeg_dir = Path(r'E:\dataset\fivek_jpeg')
    jpg_files = sorted(jpeg_dir.glob('*.jpg'))
    logger.info(f"找到 {len(jpg_files)} 张 FiveK JPEG")
    
    # 批量推理
    results = {}
    batch_size = 16
    dim_names = ['composition', 'lighting', 'color', 'clarity', 'subject']
    
    with torch.no_grad():
        for i in tqdm(range(0, len(jpg_files), batch_size), desc="评分中"):
            batch_files = jpg_files[i:i+batch_size]
            images = []
            names = []
            for f in batch_files:
                try:
                    img = Image.open(f).convert('RGB')
                    images.append(transform(img))
                    names.append(f.name)
                except Exception as e:
                    logger.warning(f"跳过 {f.name}: {e}")
            
            if not images:
                continue
            
            batch = torch.stack(images).to(device)
            outputs = model(batch)
            scores = outputs['scores'].cpu()       # (B, 5)
            overall = outputs['weighted_score'].cpu()  # (B,)
            
            for j, name in enumerate(names):
                results[name] = {
                    'overall': round(float(overall[j]), 3),
                    'dimensions': {
                        dim_names[k]: round(float(scores[j, k]), 3)
                        for k in range(5)
                    }
                }
    
    # 统计
    all_scores = [v['overall'] for v in results.values()]
    import numpy as np
    logger.info(f"评分完成: {len(results)} 张")
    logger.info(f"  overall: mean={np.mean(all_scores):.2f}, "
                f"min={np.min(all_scores):.2f}, max={np.max(all_scores):.2f}, "
                f"std={np.std(all_scores):.2f}")
    
    # 保存
    output = {
        'metadata': {
            'scorer': 'AADB_aesthetic_full',
            'num_images': len(results),
            'score_range': [0, 10],
        },
        'scores': results
    }
    
    out_path = PROJECT_ROOT / 'data' / 'fivek_aesthetic_scores.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
