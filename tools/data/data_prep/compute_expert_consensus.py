"""
计算每张 FiveK 图片的专家一致性分数 (Expert Consensus Score)

原理: 对每张图片, 计算5位专家在各参数上的标准差
     → 标准差小 = 专家一致 = 高权重
     → 标准差大 = 专家分歧 = 低权重

输出: data/fivek_expert_consensus.json
  {
    "image_name": {
      "consensus_score": 0.82,       # 综合一致性 [0,1], 越高越一致
      "per_param": { "ev": 0.95, "wb": 0.78, ... },
      "num_experts": 5
    }
  }
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 有效参数 (去掉 clarity, 全零无意义)
PARAMS = [
    'ev_compensation', 'white_balance', 'contrast', 'brightness',
    'shadows', 'highlights', 'saturation', 'vibrance'
]

# 各参数的物理范围 (用于归一化 STD)
PARAM_RANGES = {
    'ev_compensation': 6.0,    # -3 ~ +3
    'white_balance': 8000.0,   # 2000 ~ 10000
    'contrast': 100.0,         # -100 ~ 0 ~ 100
    'brightness': 100.0,
    'shadows': 100.0,
    'highlights': 100.0,
    'saturation': 200.0,       # -100 ~ 100
    'vibrance': 200.0,
    'clarity': 100.0,
}


def main():
    # 加载专家参数
    data_path = PROJECT_ROOT / 'data' / 'fivek_expert_params.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按图片分组
    img_params = defaultdict(lambda: defaultdict(list))
    img_experts = defaultdict(set)
    
    for s in data['samples']:
        name = s['image_name']
        img_experts[name].add(s.get('expert', 'unknown'))
        for p in PARAMS:
            img_params[name][p].append(float(s.get(p, 0)))
    
    print(f"总图片数: {len(img_params)}")
    
    # 计算每张图的一致性分数
    results = {}
    all_consensus = []
    
    for img_name in sorted(img_params.keys()):
        per_param_consensus = {}
        
        for p in PARAMS:
            values = img_params[img_name][p]
            if len(values) > 1:
                std = np.std(values)
                # 归一化: STD/范围 → [0,1], 然后反转 (1 = 完全一致, 0 = 完全分歧)
                normalized_std = std / PARAM_RANGES[p]
                consensus = max(0.0, 1.0 - normalized_std * 5)  # 放大差异, 20%以上的std → 0分
            else:
                consensus = 1.0  # 只有一个专家, 无法评估, 默认满分
            
            per_param_consensus[p] = round(consensus, 4)
        
        # 综合一致性: 各参数一致性的加权平均
        overall = np.mean(list(per_param_consensus.values()))
        
        results[img_name] = {
            'consensus_score': round(float(overall), 4),
            'per_param': per_param_consensus,
            'num_experts': len(img_experts[img_name]),
        }
        all_consensus.append(overall)
    
    # 统计
    all_consensus = np.array(all_consensus)
    print(f"\n专家一致性分数统计:")
    print(f"  mean = {np.mean(all_consensus):.3f}")
    print(f"  std  = {np.std(all_consensus):.3f}")
    print(f"  min  = {np.min(all_consensus):.3f}")
    print(f"  max  = {np.max(all_consensus):.3f}")
    
    # 分段统计
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    labels = ['很低', '低  ', '中  ', '高  ', '很高']
    print(f"\n分段分布:")
    for (lo, hi), label in zip(bins, labels):
        count = np.sum((all_consensus >= lo) & (all_consensus < hi))
        pct = count / len(all_consensus) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{lo:.1f}, {hi:.1f}) {label}: {count:>5} 张 ({pct:5.1f}%) {bar}")
    
    # 展示最高/最低一致性的图片
    sorted_imgs = sorted(results.items(), key=lambda x: x[1]['consensus_score'])
    
    print(f"\n一致性最低的 5 张图片 (专家分歧最大):")
    for name, info in sorted_imgs[:5]:
        print(f"  {name}: score={info['consensus_score']:.3f}")
    
    print(f"\n一致性最高的 5 张图片 (专家意见统一):")
    for name, info in sorted_imgs[-5:]:
        print(f"  {name}: score={info['consensus_score']:.3f}")
    
    # 保存
    output = {
        'metadata': {
            'description': 'FiveK per-image expert consensus scores',
            'method': '1 - 5*normalized_std, averaged across 8 params (excl. clarity)',
            'score_range': [0, 1],
            'higher_is_better': True,
            'num_images': len(results),
            'params': PARAMS,
            'stats': {
                'mean': round(float(np.mean(all_consensus)), 4),
                'std': round(float(np.std(all_consensus)), 4),
                'min': round(float(np.min(all_consensus)), 4),
                'max': round(float(np.max(all_consensus)), 4),
            }
        },
        'scores': results
    }
    
    out_path = PROJECT_ROOT / 'data' / 'fivek_expert_consensus.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == '__main__':
    main()
