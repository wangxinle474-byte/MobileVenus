"""
[DEPRECATED] 建议生成器 — 旧架构，已废弃

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
此文件属于早期方案 (TinyLLaMA + 建议生成)，
已被当前 Stage C 文本条件化方案替代。

当前文本交互请参考:
  - training/text_condition/model.py::TextConditionedModel
  - training/text_condition/model.py::FiLMFusion

保留此文件仅作为历史参考，不参与当前训练管线。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原始设计: 根据视觉特征和评分生成美学改进建议
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import random


class SuggestionGenerator(nn.Module):
    """
    建议生成器
    根据视觉特征和评分生成改进建议
    
    Args:
        hidden_size: 隐藏层大小
        num_suggestion_types: 建议类型数量
    """
    
    def __init__(self, hidden_size=2048, num_suggestion_types=10):
        super().__init__()
        
        self.num_suggestion_types = num_suggestion_types
        
        # 条件编码器 (视觉特征 + 评分)
        self.condition_encoder = nn.Sequential(
            nn.Linear(512 + 5, hidden_size),  # visual_feat(512) + scores(5)
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU()
        )
        
        # 建议类型分类器
        self.suggestion_classifier = nn.Linear(hidden_size // 2, num_suggestion_types)
        
        # 建议参数预测器
        self.param_predictor = nn.ModuleDict({
            'direction': nn.Linear(hidden_size // 2, 4),  # 上下左右
            'intensity': nn.Linear(hidden_size // 2, 3),  # 轻微/中等/严重
            'action': nn.Linear(hidden_size // 2, 8),     # 各种动作
        })
        
        # 建议模板库
        self.suggestion_templates = self._build_templates()
        
    def _build_templates(self) -> Dict[int, Dict]:
        """构建建议模板库"""
        return {
            0: {  # 构图问题
                'name': 'composition',
                'templates': [
                    "主体位置偏{direction}，建议{action}",
                    "构图{intensity}失衡，建议应用{rule}",
                    "视觉引导线不明显，建议{action}",
                ]
            },
            1: {  # 光线问题
                'name': 'lighting',
                'templates': [
                    "光线{intensity}{issue}，建议{action}",
                    "逆光导致主体欠曝，建议{action}",
                    "对比度{intensity}不足，建议{action}",
                ]
            },
            2: {  # 色彩问题
                'name': 'color',
                'templates': [
                    "色彩{intensity}{issue}，建议{action}",
                    "色温偏{direction}，建议{action}",
                    "饱和度{intensity}过高，建议{action}",
                ]
            },
            3: {  # 清晰度问题
                'name': 'clarity',
                'templates': [
                    "画面{intensity}模糊，建议{action}",
                    "焦点不准确，建议{action}",
                    "噪点{intensity}明显，建议{action}",
                ]
            },
            4: {  # 主体问题
                'name': 'subject',
                'templates': [
                    "主体不够突出，建议{action}",
                    "背景{intensity}杂乱，建议{action}",
                    "主体与背景对比度不足，建议{action}",
                ]
            },
            5: {  # 曝光问题
                'name': 'exposure',
                'templates': [
                    "曝光{intensity}{issue}，建议{action} {value}EV",
                    "高光{intensity}溢出，建议{action}",
                    "暗部细节丢失，建议{action}",
                ]
            },
            6: {  # 角度问题
                'name': 'angle',
                'templates': [
                    "拍摄角度{intensity}不佳，建议{action}",
                    "水平线倾斜，建议{action}",
                    "视角{intensity}平淡，建议{action}",
                ]
            },
            7: {  # 距离问题
                'name': 'distance',
                'templates': [
                    "拍摄距离{intensity}{issue}，建议{action}",
                    "主体{intensity}过小，建议{action}",
                    "画面{intensity}拥挤，建议{action}",
                ]
            },
            8: {  # 时机问题
                'name': 'timing',
                'templates': [
                    "光线条件{intensity}不佳，建议{action}",
                    "等待更好的光线时机",
                    "建议在{time}拍摄",
                ]
            },
            9: {  # 综合建议
                'name': 'general',
                'templates': [
                    "整体{intensity}{issue}，建议{action}",
                    "可尝试{action}以提升画面质量",
                    "建议{action}后重新构图",
                ]
            }
        }
    
    def forward(self, visual_features: torch.Tensor, scores: torch.Tensor) -> List[List[str]]:
        """
        生成建议
        
        Args:
            visual_features: (B, 512) 视觉特征
            scores: (B, 5) 五维度评分
            
        Returns:
            suggestions: List[List[str]] 每个样本的建议列表
        """
        batch_size = visual_features.shape[0]
        
        # 拼接条件
        condition = torch.cat([visual_features, scores], dim=1)  # (B, 517)
        condition_emb = self.condition_encoder(condition)  # (B, hidden_size//2)
        
        # 分类建议类型
        suggestion_logits = self.suggestion_classifier(condition_emb)  # (B, num_types)
        
        # 预测参数
        params = {
            name: predictor(condition_emb)
            for name, predictor in self.param_predictor.items()
        }
        
        # 生成建议文本
        suggestions = []
        for batch_idx in range(batch_size):
            # 选择 top-3 建议类型
            top_types = torch.topk(suggestion_logits[batch_idx], k=3).indices
            
            batch_suggestions = []
            for type_idx in top_types:
                suggestion = self._generate_suggestion(
                    type_idx.item(),
                    scores[batch_idx].cpu().numpy(),
                    {k: v[batch_idx] for k, v in params.items()}
                )
                if suggestion:
                    batch_suggestions.append(suggestion)
            
            suggestions.append(batch_suggestions)
        
        return suggestions
    
    def _generate_suggestion(
        self,
        suggestion_type: int,
        scores: 'np.ndarray',
        params: Dict[str, torch.Tensor]
    ) -> str:
        """
        根据类型和参数生成具体建议
        
        Args:
            suggestion_type: 建议类型索引
            scores: 五维度评分
            params: 预测的参数
            
        Returns:
            suggestion: 建议文本
        """
        template_info = self.suggestion_templates.get(suggestion_type)
        if not template_info:
            return ""
        
        # 随机选择一个模板
        template = random.choice(template_info['templates'])
        
        # 根据评分和参数填充模板
        fill_dict = self._get_fill_dict(scores, params)
        
        try:
            suggestion = template.format(**fill_dict)
            return suggestion
        except KeyError:
            # 如果模板缺少某些键，返回简化版本
            return template
    
    def _get_fill_dict(self, scores, params) -> Dict[str, str]:
        """生成模板填充字典"""
        # 方向
        direction_map = ['左', '右', '上', '下']
        direction_idx = torch.argmax(params['direction']).item()
        direction = direction_map[direction_idx]
        
        # 强度
        intensity_map = ['轻微', '中等', '严重']
        intensity_idx = torch.argmax(params['intensity']).item()
        intensity = intensity_map[intensity_idx]
        
        # 动作
        action_map = [
            '向右移动取景框',
            '调整拍摄角度',
            '增加曝光补偿',
            '降低曝光',
            '使用HDR模式',
            '开启人像模式',
            '调整白平衡',
            '等待更好的光线'
        ]
        action_idx = torch.argmax(params['action']).item()
        action = action_map[action_idx]
        
        # 问题描述
        issue_map = {
            'low': '不足',
            'high': '过高',
            'unbalanced': '失衡'
        }
        
        # 根据评分判断问题
        composition_score = scores[0]
        if composition_score < 5:
            issue = issue_map['unbalanced']
        elif composition_score < 7:
            issue = issue_map['low']
        else:
            issue = '良好'
        
        # 构图规则
        rules = ['三分法', '对称构图', '引导线构图', '框架构图']
        rule = random.choice(rules)
        
        # 曝光值
        value = '+1' if scores[1] < 5 else '-1'
        
        # 时间建议
        times = ['清晨', '傍晚', '黄金时刻', '蓝调时刻']
        time = random.choice(times)
        
        return {
            'direction': direction,
            'intensity': intensity,
            'action': action,
            'issue': issue,
            'rule': rule,
            'value': value,
            'time': time
        }


class SimpleSuggestionGenerator:
    """
    简化版建议生成器（基于规则）
    不需要神经网络，适合快速部署
    """
    
    def __init__(self):
        self.dimension_names = ['composition', 'lighting', 'color', 'clarity', 'subject']
        
    def generate(self, scores: Dict[str, float]) -> List[str]:
        """
        基于规则生成建议
        
        Args:
            scores: 评分字典
            
        Returns:
            suggestions: 建议列表
        """
        suggestions = []
        
        # 构图建议
        if scores['composition'] < 7.0:
            suggestions.append("💡 构图建议：尝试应用三分法，将主体放在画面交叉点")
        
        # 光线建议
        if scores['lighting'] < 6.0:
            suggestions.append("☀️ 光线建议：当前光线条件不佳，建议调整拍摄角度或等待更好的光线")
        elif scores['lighting'] < 7.5:
            suggestions.append("☀️ 光线建议：可以尝试增加曝光补偿 +0.5EV")
        
        # 色彩建议
        if scores['color'] < 7.0:
            suggestions.append("🎨 色彩建议：色彩饱和度偏低，可以调整白平衡或使用滤镜")
        
        # 清晰度建议
        if scores['clarity'] < 7.0:
            suggestions.append("🔍 清晰度建议：画面略显模糊，建议稳定手机或使用三脚架")
        
        # 主体建议
        if scores['subject'] < 7.0:
            suggestions.append("👤 主体建议：主体不够突出，建议使用人像模式虚化背景")
        
        # 综合建议
        overall = sum(scores.values()) / len(scores)
        if overall < 6.0:
            suggestions.append("📸 综合建议：建议重新构图，调整拍摄参数后再试")
        
        return suggestions[:3]  # 最多返回3条建议


if __name__ == "__main__":
    # 测试建议生成器
    generator = SuggestionGenerator(hidden_size=2048, num_suggestion_types=10)
    
    # 计算参数量
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / (1024**2):.2f} MB")
    
    # 测试前向传播
    dummy_features = torch.randn(2, 512)
    dummy_scores = torch.rand(2, 5) * 10  # [0, 10]
    
    suggestions = generator(dummy_features, dummy_scores)
    
    print(f"\nGenerated suggestions:")
    for i, batch_suggestions in enumerate(suggestions):
        print(f"\nBatch {i}:")
        for j, suggestion in enumerate(batch_suggestions):
            print(f"  {j+1}. {suggestion}")
    
    # 测试简化版
    print("\n" + "="*50)
    print("Testing SimpleSuggestionGenerator")
    print("="*50)
    
    simple_generator = SimpleSuggestionGenerator()
    test_scores = {
        'composition': 6.5,
        'lighting': 5.8,
        'color': 7.2,
        'clarity': 6.0,
        'subject': 6.8
    }
    
    simple_suggestions = simple_generator.generate(test_scores)
    print(f"\nScores: {test_scores}")
    print(f"\nSimple suggestions:")
    for i, suggestion in enumerate(simple_suggestions):
        print(f"  {i+1}. {suggestion}")
