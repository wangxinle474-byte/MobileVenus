"""
参数预测器 (5参数精简版 - 方案B)
将美学问题语义转换为具体的相机参数调整

参数集 (5个, 覆盖90%常见拍摄问题):
  1. EV补偿:   亮度控制 [-3, +3]
  2. 白平衡:   色温控制 [2000, 10000] K
  3. 对焦点:   清晰主体 (x, y) ∈ [0,1]²
  4. HDR:      动态范围 (开/关)
  5. 拍摄模式: 场景预设 (auto/portrait/night/landscape/macro)

架构:
  ProblemClassifier → CrossAttention ParameterMappingNetwork → 5参数解码
  - ProblemClassifier: 9类问题检测 + 严重程度
  - ParameterMappingNetwork: 5个可学习参数token + 交叉注意力映射
  - ParameterPredictor: 主模块, 串联以上组件 + 置信度估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 问题分类器
# ============================================================

class ProblemClassifier(nn.Module):
    """
    多标签美学问题分类器
    
    基于视觉特征和美学评分，自动检测图像中存在的多种质量问题。
    支持多标签分类（一张图可能同时存在多个问题）和严重程度评估。
    
    问题类型 (9类):
        0: underexposure    (欠曝)
        1: overexposure     (过曝)
        2: color_cast       (色偏)
        3: poor_composition (构图不佳)
        4: blur             (模糊)
        5: noise            (噪点)
        6: backlight        (逆光)
        7: low_contrast     (对比度低)
        8: background_messy (背景杂乱)
    
    Args:
        visual_dim: 视觉特征维度
        score_dim: 美学评分维度
        hidden_dim: 隐藏层维度
        num_problems: 问题类型数量
        num_severities: 严重程度级别数 (mild/moderate/severe)
    """
    
    PROBLEM_NAMES = [
        'underexposure',     # 欠曝
        'overexposure',      # 过曝
        'color_cast',        # 色偏
        'poor_composition',  # 构图不佳
        'blur',              # 模糊
        'noise',             # 噪点
        'backlight',         # 逆光
        'low_contrast',      # 对比度低
        'background_messy',  # 背景杂乱
    ]
    
    SEVERITY_NAMES = ['mild', 'moderate', 'severe']
    
    def __init__(
        self,
        visual_dim: int = 512,
        score_dim: int = 5,
        hidden_dim: int = 256,
        num_problems: int = 9,
        num_severities: int = 3
    ):
        super().__init__()
        
        self.num_problems = num_problems
        self.num_severities = num_severities
        
        # 输入融合: 视觉特征 + 美学评分
        self.input_fusion = nn.Sequential(
            nn.Linear(visual_dim + score_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 多层感知器
        self.feature_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 问题检测头 (多标签二分类)
        self.problem_head = nn.Linear(hidden_dim, num_problems)
        
        # 严重程度头 (每个问题独立的严重程度)
        self.severity_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_severities)
            for _ in range(num_problems)
        ])
    
    def forward(
        self,
        visual_features: torch.Tensor,
        scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: (B, visual_dim) 视觉特征
            scores: (B, score_dim) 美学评分
            
        Returns:
            dict:
                'problem_logits':  (B, num_problems) 问题存在概率 logits
                'problem_probs':   (B, num_problems) 问题存在概率 [0,1]
                'severity_logits': (B, num_problems, num_severities)
                'severity_probs':  (B, num_problems, num_severities)
                'hidden':          (B, hidden_dim) 中间特征
        """
        x = torch.cat([visual_features, scores], dim=1)
        x = self.input_fusion(x)
        hidden = self.feature_refiner(x)  # (B, hidden_dim)
        
        # 问题检测
        problem_logits = self.problem_head(hidden)    # (B, num_problems)
        problem_probs = torch.sigmoid(problem_logits)  # 多标签用sigmoid
        
        # 严重程度
        severity_logits = torch.stack([
            head(hidden) for head in self.severity_heads
        ], dim=1)  # (B, num_problems, num_severities)
        severity_probs = F.softmax(severity_logits, dim=-1)
        
        return {
            'problem_logits': problem_logits,
            'problem_probs': problem_probs,
            'severity_logits': severity_logits,
            'severity_probs': severity_probs,
            'hidden': hidden,
        }


# ============================================================
# 参数映射网络
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    交叉注意力模块
    
    将问题语义信息交叉注入到视觉特征中，
    让参数预测关注与问题相关的视觉区域。
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query:   (B, Nq, D) 查询（参数 token）
            context: (B, Nc, D) 上下文（视觉 + 问题特征）
        Returns:
            (B, Nq, D)
        """
        attended, _ = self.cross_attn(query, context, context)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))
        return query


class ParameterMappingNetwork(nn.Module):
    """
    参数映射网络 (5参数精简版)
    
    使用可学习的“参数 token”和交叉注意力，
    将问题语义映射为 5 个核心相机参数调整值。
    
    Token 分配:
      token 0 → EV补偿      (连续, [-3, +3])
      token 1 → 白平衡      (连续, [2000, 10000] K)
      token 2 → 对焦点      (连续, (x,y) ∈ [0,1]²)
      token 3 → HDR         (二分类, off/on)
      token 4 → 拍摄模式    (5分类)
    
    Args:
        dim: 特征维度
        num_cross_attn_layers: 交叉注意力层数
        num_heads: 注意力头数
    """
    
    NUM_PARAMS = 5
    PARAM_NAMES = ['ev_compensation', 'white_balance', 'focus_point', 'hdr', 'mode']
    
    def __init__(
        self,
        dim: int = 256,
        num_cross_attn_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.dim = dim
        self.num_param_tokens = self.NUM_PARAMS
        
        # 可学习的参数 token (5个)
        self.param_tokens = nn.Parameter(
            torch.randn(1, self.NUM_PARAMS, dim) * 0.02
        )
        
        # 交叉注意力层
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads=num_heads)
            for _ in range(num_cross_attn_layers)
        ])
        
        # ---------- 参数解码头 (5个) ----------
        # token 0: EV 补偿 [-3, +3]
        self.ev_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        # token 1: 白平衡 [2000, 10000] K
        self.wb_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        # token 2: 对焦点 (x, y) ∈ [0, 1]²
        self.focus_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 2)
        )
        # token 3: HDR [off, on]
        self.hdr_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 2)
        )
        # token 4: 拍摄模式 [auto, portrait, night, landscape, macro]
        self.mode_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 5)
        )
        
        # 置信度头: 每个参数的预测置信度
        self.confidence_head = nn.Sequential(
            nn.Linear(dim * self.NUM_PARAMS, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, self.NUM_PARAMS),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        problem_features: torch.Tensor,
        scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: (B, hidden_dim)
            problem_features: (B, hidden_dim)
            scores: (B, 5)
            
        Returns:
            dict 包含 5 个相机参数预测值和置信度
        """
        B = visual_features.shape[0]
        
        # 构建上下文: 视觉 + 问题 + 评分
        context = torch.stack([
            visual_features, problem_features,
            F.pad(scores, (0, visual_features.shape[1] - scores.shape[1]))
        ], dim=1)  # (B, 3, D)
        
        # 参数 token 扩展到 batch
        param_tokens = self.param_tokens.expand(B, -1, -1)  # (B, 5, D)
        
        # 交叉注意力: 参数 token 查询上下文
        for layer in self.cross_attn_layers:
            param_tokens = layer(param_tokens, context)
        
        # 解码 5 个参数
        params = {}
        
        # token 0: EV 补偿 [-3, +3]
        params['ev_compensation'] = torch.tanh(self.ev_head(param_tokens[:, 0])) * 3.0
        
        # token 1: 白平衡 [2000, 10000] K
        wb_raw = torch.sigmoid(self.wb_head(param_tokens[:, 1]))
        params['white_balance'] = 2000.0 + wb_raw * 8000.0
        
        # token 2: 对焦点 (x, y) ∈ [0, 1]²
        params['focus_point'] = torch.sigmoid(self.focus_head(param_tokens[:, 2]))
        
        # token 3: HDR [off=0, on=1]
        params['hdr_logits'] = self.hdr_head(param_tokens[:, 3])
        params['hdr'] = F.softmax(params['hdr_logits'], dim=-1)
        
        # token 4: 拍摄模式 (5分类)
        params['mode_logits'] = self.mode_head(param_tokens[:, 4])
        params['mode'] = F.softmax(params['mode_logits'], dim=-1)
        
        # 全局置信度
        all_tokens = param_tokens.reshape(B, -1)  # (B, 5*D)
        params['confidence'] = self.confidence_head(all_tokens)  # (B, 5)
        
        return params


# ============================================================
# 参数预测器 (主模块)
# ============================================================

class ParameterPredictor(nn.Module):
    """
    参数预测器 (5参数精简版)
    
    语义→参数转换流程:
    
    输入:  视觉特征 (B, 512) + 美学评分 (B, 5)
            ↓
    Step 1: ProblemClassifier  → 9类问题检测 + 严重程度
            ↓
    Step 2: ParameterMappingNetwork → 5参数 token + 交叉注意力
            ↓
    输出:  5个相机参数 + 置信度
    
    Args:
        config: 配置对象，包含:
            - visual_dim (int): 视觉特征维度，默认 512
            - score_dim (int): 评分维度，默认 5
            - hidden_dim (int): 隐藏层维度，默认 256
            - num_problems (int): 问题类型数，默认 9
            - num_cross_attn_layers (int): 交叉注意力层数，默认 2
    """
    
    PARAM_NAMES = ['ev_compensation', 'white_balance', 'focus_point', 'hdr', 'mode']
    MODE_NAMES = ['auto', 'portrait', 'night', 'landscape', 'macro']
    
    def __init__(self, config=None):
        super().__init__()
        
        # 解析配置
        if config is None:
            config = type('Config', (), {
                'visual_dim': 512, 'score_dim': 5,
                'hidden_dim': 256, 'num_problems': 9,
                'num_cross_attn_layers': 2
            })()
        self.config = config
        
        visual_dim = getattr(config, 'visual_dim', 512)
        score_dim = getattr(config, 'score_dim', 5)
        hidden_dim = getattr(config, 'hidden_dim', 256)
        num_problems = getattr(config, 'num_problems', 9)
        num_cross_attn_layers = getattr(config, 'num_cross_attn_layers', 2)
        
        # Step 1: 问题分类器
        self.problem_classifier = ProblemClassifier(
            visual_dim=visual_dim,
            score_dim=score_dim,
            hidden_dim=hidden_dim,
            num_problems=num_problems
        )
        
        # 视觉特征投影 (visual_dim → hidden_dim)
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Step 2: 参数映射网络 (5参数)
        self.parameter_mapping = ParameterMappingNetwork(
            dim=hidden_dim,
            num_cross_attn_layers=num_cross_attn_layers,
            num_heads=4
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        scores: torch.Tensor,
        problem_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            visual_features: (B, visual_dim) 视觉特征
            scores: (B, score_dim) 美学评分
            problem_type: (B, num_problems) 问题类型 (optional, 若不提供则自动检测)
            
        Returns:
            dict 包含:
                - 所有相机参数预测值
                - problem_detection: 问题检测结果
                - confidence: 参数置信度
        """
        # Step 1: 问题分类
        problem_result = self.problem_classifier(visual_features, scores)
        problem_features = problem_result['hidden']  # (B, hidden_dim)
        
        # Step 2: 视觉特征投影
        visual_projected = self.visual_projection(visual_features)  # (B, hidden_dim)
        
        # Step 3: 参数映射
        params = self.parameter_mapping(
            visual_projected, problem_features, scores
        )
        
        # 附加问题检测结果
        params['problem_probs'] = problem_result['problem_probs']
        params['problem_logits'] = problem_result['problem_logits']
        params['severity_probs'] = problem_result['severity_probs']
        params['severity_logits'] = problem_result['severity_logits']
        
        return params
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        problem_labels: Optional[torch.Tensor] = None,
        severity_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失 (5参数版)
        
        Args:
            predictions: forward() 的输出
            targets: 地面真值参数 (可包含: ev_compensation, white_balance,
                     focus_point, hdr, mode)
            problem_labels: (B, num_problems) 问题标签 (可选)
            severity_labels: (B, num_problems) 严重程度标签 (可选)
            
        Returns:
            losses: 各部分损失值
        """
        device = predictions['ev_compensation'].device
        losses = {}
        
        # ---------- 连续值参数: Smooth L1 Loss ----------
        # 模型输出为原始尺度 (EV [-3,3], WB [2000,10000])，
        # 训练目标已归一化到 [-1,1]，需要将预测值也归一化后再算 loss
        continuous_params = ['ev_compensation', 'white_balance']
        PARAM_RANGES = {
            'ev_compensation': (-3.0, 3.0),
            'white_balance': (2000.0, 10000.0),
        }
        param_loss = torch.tensor(0.0, device=device)
        count = 0
        for name in continuous_params:
            if name in targets:
                pred = predictions[name].squeeze(-1)
                tgt = targets[name].float()
                # 归一化预测值到 [-1, 1] 以匹配目标
                if name in PARAM_RANGES:
                    lo, hi = PARAM_RANGES[name]
                    pred = 2.0 * (pred - lo) / (hi - lo) - 1.0
                param_loss = param_loss + F.smooth_l1_loss(pred, tgt)
                count += 1
        if count > 0:
            losses['param_regression'] = param_loss / count
        
        # 对焦点: L2 Loss
        if 'focus_point' in targets:
            losses['focus_point'] = F.mse_loss(
                predictions['focus_point'],
                targets['focus_point'].float()
            )
        
        # ---------- 分类参数: Cross Entropy ----------
        for name in ['hdr', 'mode']:
            logit_key = f'{name}_logits'
            if logit_key in predictions and name in targets:
                losses[f'{name}_cls'] = F.cross_entropy(
                    predictions[logit_key],
                    targets[name].long()
                )
        
        # ---------- 问题分类损失 ----------
        if problem_labels is not None:
            losses['problem_cls'] = F.binary_cross_entropy_with_logits(
                predictions['problem_logits'],
                problem_labels.float()
            )
        
        if severity_labels is not None:
            B, P = severity_labels.shape
            sev_logits = predictions['severity_logits']  # (B, P, S)
            losses['severity_cls'] = F.cross_entropy(
                sev_logits.reshape(B * P, -1),
                severity_labels.reshape(B * P).long()
            )
        
        # ---------- 加权总损失 ----------
        total = torch.tensor(0.0, device=device)
        weights = {
            'param_regression': 1.0,
            'focus_point': 0.8,
            'hdr_cls': 0.5,
            'mode_cls': 0.3,
            'problem_cls': 0.8,
            'severity_cls': 0.3,
        }
        for name, loss in losses.items():
            total = total + weights.get(name, 1.0) * loss
        losses['total'] = total
        
        return losses
    
    def predict_readable(
        self,
        visual_features: torch.Tensor,
        scores: torch.Tensor
    ) -> List[Dict]:
        """
        生成可读的预测结果 (5参数)
        
        Args:
            visual_features: (B, visual_dim)
            scores: (B, score_dim)
            
        Returns:
            List[Dict]: 每张图片的预测结果
        """
        with torch.no_grad():
            output = self.forward(visual_features, scores)
        
        B = visual_features.shape[0]
        results = []
        
        for i in range(B):
            # 问题检测
            probs = output['problem_probs'][i]
            detected = []
            for j, name in enumerate(ProblemClassifier.PROBLEM_NAMES):
                if probs[j] > 0.5:
                    sev_idx = output['severity_probs'][i, j].argmax().item()
                    detected.append({
                        'problem': name,
                        'confidence': float(probs[j]),
                        'severity': ProblemClassifier.SEVERITY_NAMES[sev_idx]
                    })
            
            # 5个参数 + 置信度
            conf = output['confidence'][i]  # (5,)
            result = {
                'problems_detected': detected,
                'parameters': {
                    'ev_compensation': {
                        'value': float(output['ev_compensation'][i]),
                        'confidence': float(conf[0])
                    },
                    'white_balance': {
                        'value': float(output['white_balance'][i]),
                        'unit': 'K',
                        'confidence': float(conf[1])
                    },
                    'focus_point': {
                        'value': output['focus_point'][i].tolist(),
                        'confidence': float(conf[2])
                    },
                    'hdr': {
                        'value': 'on' if output['hdr'][i, 1] > 0.5 else 'off',
                        'confidence': float(conf[3])
                    },
                    'mode': {
                        'value': self.MODE_NAMES[output['mode'][i].argmax().item()],
                        'confidence': float(conf[4])
                    },
                },
                'overall_confidence': float(conf.mean())
            }
            results.append(result)
        
        return results


class RuleBasedParameterPredictor:
    """
    基于规则的参数预测器
    作为神经网络版本的备选方案
    """
    
    def __init__(self):
        # 参数映射规则库
        self.rules = self._build_rules()
        
    def _build_rules(self) -> Dict:
        """构建 5 参数映射规则"""
        return {
            'underexposure': {
                'mild':     {'ev': 0.5, 'hdr': False, 'mode': 'auto'},
                'moderate': {'ev': 1.0, 'hdr': True,  'mode': 'auto'},
                'severe':   {'ev': 1.5, 'hdr': True,  'mode': 'night'}
            },
            'overexposure': {
                'mild':     {'ev': -0.5, 'hdr': False, 'mode': 'auto'},
                'moderate': {'ev': -1.0, 'hdr': False, 'mode': 'auto'},
                'severe':   {'ev': -1.5, 'hdr': True,  'mode': 'auto'}
            },
            'color_cast': {
                'warm':    {'white_balance': 4500},
                'cool':    {'white_balance': 7000},
                'default': {'white_balance': 5500}
            },
            'poor_composition': {
                'subject_left':   {'focus_point': (0.67, 0.5)},
                'subject_right':  {'focus_point': (0.33, 0.5)},
                'subject_top':    {'focus_point': (0.5, 0.67)},
                'subject_bottom': {'focus_point': (0.5, 0.33)}
            },
            'backlight': {
                'mild':     {'ev': 0.5, 'hdr': True},
                'moderate': {'ev': 1.0, 'hdr': True},
                'severe':   {'ev': 1.5, 'hdr': True}
            },
            'background_messy': {
                'default': {'mode': 'portrait', 'focus_point': (0.5, 0.5)}
            },
            'blur': {
                'default': {'mode': 'auto', 'focus_point': (0.5, 0.5)}
            },
            'noise': {
                'default': {'ev': 0.3, 'mode': 'night'}
            }
        }
    
    def predict(
        self,
        problem_type: str,
        severity: str = 'moderate',
        scores: Optional[Dict] = None
    ) -> Dict:
        """
        基于规则预测参数
        
        Args:
            problem_type: 问题类型
            severity: 严重程度 (mild/moderate/severe)
            scores: 美学评分（可选，用于动态调整）
            
        Returns:
            parameters: 预测的参数字典
        """
        if problem_type not in self.rules:
            return {}
        
        rule = self.rules[problem_type]
        
        # 如果有严重程度分级
        if severity in rule:
            params = rule[severity].copy()
        elif 'default' in rule:
            params = rule['default'].copy()
        else:
            params = {}
        
        # 根据评分动态调整
        if scores is not None:
            params = self._adjust_by_scores(params, scores)
        
        return params
    
    def _adjust_by_scores(self, params: Dict, scores: Dict) -> Dict:
        """根据评分动态调整参数 (5参数版)"""
        adjusted = params.copy()
        
        # 如果光线评分很低，增加 EV 补偿
        if 'lighting' in scores and scores['lighting'] < 5.0:
            if 'ev' in adjusted:
                adjusted['ev'] += 0.5
        
        # 如果色彩评分很低，微调白平衡
        if 'color' in scores and scores['color'] < 5.0:
            if 'white_balance' not in adjusted:
                adjusted['white_balance'] = 5500
        
        return adjusted


class ParameterValidator:
    """参数验证器 (5参数版)，确保参数在安全范围内"""
    
    def __init__(self):
        # 5 参数限制
        self.limits = {
            'ev_compensation': (-3.0, 3.0),
            'white_balance': (2000, 10000),
            'focus_point': ((0.0, 1.0), (0.0, 1.0)),  # (x, y)
        }
        
        # 模式定义
        self.modes = ['auto', 'portrait', 'night', 'landscape', 'macro']
    
    def validate(self, params: Dict) -> Tuple[Dict, List[str]]:
        """
        验证并裁剪参数到安全范围
        
        Args:
            params: 原始参数
            
        Returns:
            validated_params: 验证后的参数
            warnings: 警告信息列表
        """
        validated = {}
        warnings = []
        
        for key, value in params.items():
            if key in self.limits:
                if key == 'focus_point':
                    # 对焦点是 (x, y) 元组
                    x_limits, y_limits = self.limits[key]
                    x = np.clip(value[0], x_limits[0], x_limits[1])
                    y = np.clip(value[1], y_limits[0], y_limits[1])
                    validated[key] = (x, y)
                    
                    if value[0] != x or value[1] != y:
                        warnings.append(f"对焦点被裁剪到安全范围: ({x:.2f}, {y:.2f})")
                else:
                    min_val, max_val = self.limits[key]
                    clipped = np.clip(value, min_val, max_val)
                    validated[key] = clipped
                    
                    if value != clipped:
                        warnings.append(f"{key} 被裁剪到安全范围: {clipped}")
            else:
                validated[key] = value
        
        # 检查参数冲突
        conflict_warnings = self._check_conflicts(validated)
        warnings.extend(conflict_warnings)
        
        return validated, warnings
    
    def _check_conflicts(self, params: Dict) -> List[str]:
        """检查参数冲突 (5参数版)"""
        warnings = []
        
        # 检查: 高 EV + HDR 开启 可能过曝
        if params.get('ev_compensation', 0) > 2.0 and params.get('hdr') is True:
            warnings.append("EV 补偿较高且 HDR 开启，可能导致过曝")
        
        # 检查: 夜景模式 + 低 EV = 照片可能过暗
        if params.get('mode') == 'night' and params.get('ev_compensation', 0) < -1.0:
            warnings.append("夜景模式下 EV 补偿过低，可能导致过暗")
        
        return warnings


class ParameterSmoother:
    """参数平滑器，避免突变"""
    
    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: 平滑因子 [0, 1]，越大变化越快
        """
        self.smoothing_factor = smoothing_factor
        self.current_params = {}
    
    def smooth(self, target_params: Dict) -> Dict:
        """
        平滑过渡到目标参数
        使用指数移动平均 (EMA)
        
        Args:
            target_params: 目标参数
            
        Returns:
            smoothed_params: 平滑后的参数
        """
        smoothed = {}
        
        for key, target_value in target_params.items():
            if key in self.current_params:
                current_value = self.current_params[key]
                
                # 对于数值参数，使用 EMA
                if isinstance(target_value, (int, float)):
                    smoothed[key] = (
                        self.smoothing_factor * target_value +
                        (1 - self.smoothing_factor) * current_value
                    )
                # 对于元组（如对焦点）
                elif isinstance(target_value, tuple):
                    smoothed[key] = tuple(
                        self.smoothing_factor * t + (1 - self.smoothing_factor) * c
                        for t, c in zip(target_value, current_value)
                    )
                # 对于分类参数（如模式），直接使用目标值
                else:
                    smoothed[key] = target_value
            else:
                # 首次设置，直接使用目标值
                smoothed[key] = target_value
        
        # 更新当前参数
        self.current_params = smoothed.copy()
        
        return smoothed
    
    def reset(self):
        """重置平滑器"""
        self.current_params = {}


class UserPreferenceLearner:
    """学习用户的参数偏好"""
    
    def __init__(self):
        self.user_adjustments = []
        self.max_history = 100
    
    def record_adjustment(
        self,
        predicted_params: Dict,
        user_params: Dict,
        accepted: bool,
        improvement: float
    ):
        """
        记录用户对预测参数的调整
        
        Args:
            predicted_params: AI 预测的参数
            user_params: 用户实际使用的参数
            accepted: 是否接受 AI 建议
            improvement: 改进幅度（评分提升）
        """
        self.user_adjustments.append({
            'predicted': predicted_params,
            'actual': user_params,
            'accepted': accepted,
            'improvement': improvement,
            'timestamp': torch.tensor(0.0)  # 实际应使用 time.time()
        })
        
        # 保持历史记录在限制内
        if len(self.user_adjustments) > self.max_history:
            self.user_adjustments.pop(0)
    
    def get_user_bias(self) -> Dict[str, float]:
        """
        计算用户的参数偏好偏差
        
        Returns:
            bias: 各参数的平均偏差
        """
        if len(self.user_adjustments) < 10:
            return {}
        
        bias = {}
        numeric_params = ['ev_compensation', 'white_balance']
        
        for param in numeric_params:
            predicted_values = []
            actual_values = []
            
            for adj in self.user_adjustments:
                if param in adj['predicted'] and param in adj['actual']:
                    predicted_values.append(adj['predicted'][param])
                    actual_values.append(adj['actual'][param])
            
            if len(predicted_values) > 0:
                bias[param] = np.mean(actual_values) - np.mean(predicted_values)
        
        return bias
    
    def get_acceptance_rate(self) -> float:
        """计算用户接受率"""
        if len(self.user_adjustments) == 0:
            return 0.0
        
        accepted_count = sum(1 for adj in self.user_adjustments if adj['accepted'])
        return accepted_count / len(self.user_adjustments)
    
    def get_average_improvement(self) -> float:
        """计算平均改进幅度"""
        if len(self.user_adjustments) == 0:
            return 0.0
        
        improvements = [adj['improvement'] for adj in self.user_adjustments]
        return np.mean(improvements)


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("MobileVenus 参数预测器 (5参数精简版) 测试")
    print("=" * 60)
    
    # ---- 1. 测试 ParameterPredictor (5参数) ----
    print("\n--- 1. ParameterPredictor (5参数版) ---")
    predictor = ParameterPredictor()
    
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"模型大小 (FP16): {total_params * 2 / (1024**2):.2f} MB")
    
    # 模拟输入
    B = 4
    visual_features = torch.randn(B, 512)
    scores = torch.rand(B, 5) * 10  # [0, 10]
    
    # 前向传播
    output = predictor(visual_features, scores)
    print(f"\n前向传播输出 (5参数):")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}: shape={str(v.shape):20s} range=[{v.min():.3f}, {v.max():.3f}]")
    
    # 可读预测
    results = predictor.predict_readable(visual_features, scores)
    print(f"\n可读预测 (第1张图):")
    print(json.dumps(results[0], indent=2, ensure_ascii=False))
    
    # ---- 2. 测试多任务损失 (5参数) ----
    print("\n--- 2. 多任务损失 ---")
    targets = {
        'ev_compensation': torch.randn(B),
        'white_balance': torch.rand(B) * 8000 + 2000,
        'focus_point': torch.rand(B, 2),
        'hdr': torch.randint(0, 2, (B,)),
        'mode': torch.randint(0, 5, (B,)),
    }
    problem_labels = (torch.rand(B, 9) > 0.5).float()
    severity_labels = torch.randint(0, 3, (B, 9))
    
    losses = predictor.compute_loss(output, targets, problem_labels, severity_labels)
    print("损失值:")
    for k, v in losses.items():
        print(f"  {k:25s}: {v.item():.4f}")
    
    # ---- 3. 测试 ProblemClassifier ----
    print("\n--- 3. ProblemClassifier ---")
    classifier = ProblemClassifier()
    cls_out = classifier(visual_features, scores)
    print(f"检测到的问题:")
    probs = cls_out['problem_probs'][0]
    for i, name in enumerate(ProblemClassifier.PROBLEM_NAMES):
        sev = ProblemClassifier.SEVERITY_NAMES[cls_out['severity_probs'][0, i].argmax().item()]
        print(f"  {name:20s}: prob={probs[i]:.3f}  severity={sev}"
    )
    
    # ---- 4. 测试规则预测器 + 验证器 + 平滑器 ----
    print("\n--- 4. 规则预测器 + 验证器 + 平滑器 ---")
    rule_pred = RuleBasedParameterPredictor()
    params = rule_pred.predict('underexposure', 'moderate')
    print(f"规则预测 (欠曝中度): {params}")
    
    validator = ParameterValidator()
    validated, warnings = validator.validate({'ev_compensation': 4.0, 'white_balance': 12000})
    print(f"验证结果: {validated}  警告: {warnings}")
    
    smoother = ParameterSmoother(smoothing_factor=0.3)
    s1 = smoother.smooth({'ev_compensation': 1.0, 'white_balance': 5500})
    s2 = smoother.smooth({'ev_compensation': 1.5, 'white_balance': 6000})
    print(f"平滑: {s1} → {s2}")
    
    print("\n" + "=" * 60)
    print("所有测试通过! (5参数版)")
    print("=" * 60)
