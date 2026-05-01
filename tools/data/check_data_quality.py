#!/usr/bin/env python3
"""
检查训练数据质量

使用方法:
    python tools/check_data_quality.py \
        --data_path data/fivek_training/train.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import Counter


def load_data(data_path: str) -> List[Dict]:
    """加载训练数据"""
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    return data


def check_data_quality(data: List[Dict], data_dir: str) -> Dict:
    """检查数据质量"""
    
    stats = {
        "total_samples": len(data),
        "valid_samples": 0,
        "invalid_samples": 0,
        "missing_images": 0,
        "missing_fields": 0,
        "problem_types": Counter(),
        "severity_levels": Counter(),
        "parameter_ranges": {
            "ev": {"min": float('inf'), "max": float('-inf'), "avg": 0},
            "iso": {"min": float('inf'), "max": float('-inf'), "avg": 0},
            "wb": {"min": float('inf'), "max": float('-inf'), "avg": 0}
        },
        "issues": []
    }
    
    ev_sum = 0
    iso_sum = 0
    wb_sum = 0
    valid_count = 0
    
    for idx, sample in enumerate(data):
        is_valid = True
        
        # 检查必需字段
        required_fields = ["image", "conversations"]
        for field in required_fields:
            if field not in sample:
                stats["missing_fields"] += 1
                stats["issues"].append(f"Sample {idx}: Missing field '{field}'")
                is_valid = False
        
        # 检查图片文件
        if "image" in sample:
            image_path = os.path.join(data_dir, sample["image"])
            if not os.path.exists(image_path):
                stats["missing_images"] += 1
                stats["issues"].append(f"Sample {idx}: Image not found: {image_path}")
                is_valid = False
        
        # 检查对话内容
        if "conversations" in sample:
            convs = sample["conversations"]
            if len(convs) < 2:
                stats["issues"].append(f"Sample {idx}: Insufficient conversations")
                is_valid = False
            else:
                # 解析回复内容
                assistant_reply = convs[1].get("value", "")
                
                # 统计问题类型
                if "underexposure" in assistant_reply.lower() or "曝光不足" in assistant_reply:
                    stats["problem_types"]["underexposure"] += 1
                elif "overexposure" in assistant_reply.lower() or "曝光过度" in assistant_reply:
                    stats["problem_types"]["overexposure"] += 1
                elif "white balance" in assistant_reply.lower() or "色温" in assistant_reply:
                    stats["problem_types"]["white_balance"] += 1
                elif "contrast" in assistant_reply.lower() or "对比度" in assistant_reply:
                    stats["problem_types"]["low_contrast"] += 1
                
                # 统计严重程度
                if "slight" in assistant_reply.lower() or "轻微" in assistant_reply:
                    stats["severity_levels"]["slight"] += 1
                elif "moderate" in assistant_reply.lower() or "中等" in assistant_reply:
                    stats["severity_levels"]["moderate"] += 1
                elif "severe" in assistant_reply.lower() or "严重" in assistant_reply:
                    stats["severity_levels"]["severe"] += 1
                
                # 提取参数范围
                try:
                    import re
                    
                    # EV
                    ev_match = re.search(r'EV[:\s]+([+-]?\d+\.?\d*)', assistant_reply)
                    if ev_match:
                        ev = float(ev_match.group(1))
                        stats["parameter_ranges"]["ev"]["min"] = min(
                            stats["parameter_ranges"]["ev"]["min"], ev
                        )
                        stats["parameter_ranges"]["ev"]["max"] = max(
                            stats["parameter_ranges"]["ev"]["max"], ev
                        )
                        ev_sum += ev
                        valid_count += 1
                    
                    # ISO
                    iso_match = re.search(r'ISO[:\s]+(\d+)', assistant_reply)
                    if iso_match:
                        iso = int(iso_match.group(1))
                        stats["parameter_ranges"]["iso"]["min"] = min(
                            stats["parameter_ranges"]["iso"]["min"], iso
                        )
                        stats["parameter_ranges"]["iso"]["max"] = max(
                            stats["parameter_ranges"]["iso"]["max"], iso
                        )
                        iso_sum += iso
                    
                    # WB
                    wb_match = re.search(r'(\d+)K', assistant_reply)
                    if wb_match:
                        wb = int(wb_match.group(1))
                        stats["parameter_ranges"]["wb"]["min"] = min(
                            stats["parameter_ranges"]["wb"]["min"], wb
                        )
                        stats["parameter_ranges"]["wb"]["max"] = max(
                            stats["parameter_ranges"]["wb"]["max"], wb
                        )
                        wb_sum += wb
                
                except Exception as e:
                    stats["issues"].append(f"Sample {idx}: Error parsing parameters: {e}")
        
        if is_valid:
            stats["valid_samples"] += 1
        else:
            stats["invalid_samples"] += 1
    
    # 计算平均值
    if valid_count > 0:
        stats["parameter_ranges"]["ev"]["avg"] = ev_sum / valid_count
        stats["parameter_ranges"]["iso"]["avg"] = iso_sum / valid_count
        stats["parameter_ranges"]["wb"]["avg"] = wb_sum / valid_count
    
    return stats


def print_stats(stats: Dict):
    """打印统计信息"""
    
    print("\n" + "="*60)
    print("数据质量检查报告")
    print("="*60)
    
    print(f"\n总样本数: {stats['total_samples']}")
    print(f"有效样本: {stats['valid_samples']} ({stats['valid_samples']/stats['total_samples']*100:.1f}%)")
    print(f"无效样本: {stats['invalid_samples']} ({stats['invalid_samples']/stats['total_samples']*100:.1f}%)")
    
    if stats['missing_images'] > 0:
        print(f"\n⚠️  缺失图片: {stats['missing_images']}")
    
    if stats['missing_fields'] > 0:
        print(f"⚠️  缺失字段: {stats['missing_fields']}")
    
    print("\n问题类型分布:")
    for problem, count in stats['problem_types'].most_common():
        print(f"  - {problem}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n严重程度分布:")
    for severity, count in stats['severity_levels'].most_common():
        print(f"  - {severity}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n参数范围:")
    for param, ranges in stats['parameter_ranges'].items():
        if ranges['min'] != float('inf'):
            print(f"  - {param.upper()}:")
            print(f"      最小值: {ranges['min']}")
            print(f"      最大值: {ranges['max']}")
            print(f"      平均值: {ranges['avg']:.2f}")
    
    if stats['issues']:
        print(f"\n⚠️  发现 {len(stats['issues'])} 个问题:")
        for issue in stats['issues'][:10]:  # 只显示前 10 个
            print(f"  - {issue}")
        if len(stats['issues']) > 10:
            print(f"  ... 还有 {len(stats['issues']) - 10} 个问题")
    
    print("\n" + "="*60)
    
    # 给出建议
    if stats['invalid_samples'] > stats['total_samples'] * 0.1:
        print("\n⚠️  警告: 无效样本超过 10%，建议检查数据质量")
    
    if len(stats['problem_types']) < 3:
        print("\n⚠️  警告: 问题类型过少，建议增加数据多样性")
    
    if stats['valid_samples'] < 1000:
        print("\n⚠️  警告: 有效样本少于 1000，建议增加数据量")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="检查训练数据质量")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="训练数据 JSON 文件路径"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="输出统计信息到 JSON 文件（可选）"
    )
    
    args = parser.parse_args()
    
    # 获取数据目录
    data_dir = os.path.dirname(args.data_path)
    
    # 加载数据
    data = load_data(args.data_path)
    
    # 检查质量
    stats = check_data_quality(data, data_dir)
    
    # 打印统计
    print_stats(stats)
    
    # 保存 JSON
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            # 转换 Counter 为普通 dict
            output_stats = {
                **stats,
                "problem_types": dict(stats["problem_types"]),
                "severity_levels": dict(stats["severity_levels"])
            }
            json.dump(output_stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
