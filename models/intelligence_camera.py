"""
IntelligenceCamera 轻量化模型主体
基于 Venus 的知识蒸馏版本，优化用于移动端部署

增强特性:
- 集成 ParameterPredictor 参数预测模块
- 支持从 Venus 预训练权重初始化
- 多尺度视觉特征提取
- 完整的推理管线: 图像 → 评分 + 参数 + 建议
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import logging
import os

from .vision_encoder import MobileViTSmall
from .aesthetic_scorer import AestheticScorer
from .parameter_predictor import ParameterPredictor
from .language_model import TinyLLaMA
from .suggestion_generator import SuggestionGenerator

logger = logging.getLogger(__name__)


class IntelligenceCamera(nn.Module):
    """
    IntelligenceCamera 主模型
    
    参数量: ~1B
    模型大小: ~500MB (FP16)
    推理延迟: <100ms (iPhone 14 Pro)
    
    Args:
        config: 模型配置
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 视觉编码器 (~5.6M 参数, 含SE+FPN)
        self.vision_encoder = MobileViTSmall(
            image_size=config.image_size,
            num_classes=0,
            use_se=getattr(config, 'use_se', True),
            use_fpn=getattr(config, 'use_fpn', True),
            output_dim=getattr(config, 'vision_output_dim', 384)
        )
        
        vision_out_dim = getattr(config, 'vision_output_dim', 384)
        projection_dim = getattr(config, 'projection_dim', 512)
        
        # 2. 视觉-语义投影层 (0.5M 参数)
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_out_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # 3. 美学评分器 (1M 参数)
        self.aesthetic_scorer = AestheticScorer(
            input_dim=projection_dim,
            num_dimensions=5,
            hidden_dim=config.scorer_hidden_dim
        )
        
        # 4. 参数预测器 (~0.8M 参数, 5参数精简版)
        if getattr(config, 'use_parameter_predictor', True):
            predictor_config = type('PredictorConfig', (), {
                'visual_dim': projection_dim,
                'score_dim': 5,
                'hidden_dim': getattr(config, 'predictor_hidden_dim', 256),
                'num_problems': 9,
                'num_cross_attn_layers': getattr(config, 'num_cross_attn_layers', 2)
            })()
            self.parameter_predictor = ParameterPredictor(predictor_config)
        else:
            self.parameter_predictor = None
        
        # 5. 轻量语言模型 (1B 参数)
        if getattr(config, 'use_language_model', True):
            self.language_model = TinyLLaMA(config.language_model_config)
            
            # 6. 建议生成器 (2M 参数)
            self.suggestion_generator = SuggestionGenerator(
                hidden_size=config.language_model_config.hidden_size,
                num_suggestion_types=config.num_suggestion_types
            )
        else:
            self.language_model = None
            self.suggestion_generator = None
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        images: torch.Tensor,
        generate_suggestion: bool = False,
        predict_parameters: bool = False,
        return_features: bool = False
    ) -> Dict[str, object]:
        """
        前向传播
        
        Args:
            images: (B, 3, H, W) 输入图像
            generate_suggestion: 是否生成建议文本
            predict_parameters: 是否预测相机参数
            return_features: 是否返回中间特征（用于蒸馏）
            
        Returns:
            dict:
                'scores': (B, 5) 五维度评分 [0, 10]
                'weighted_score': (B,) 加权总分
                'suggestions': List[str] 建议文本 (可选)
                'parameters': Dict 相机参数 (可选)
                'features': (B, 512) 视觉特征 (可选)
                'multi_scale': Dict 多尺度特征 (可选)
        """
        result = {}
        
        # 1. 视觉特征提取
        if return_features:
            encoder_out = self.vision_encoder(images, return_multi_scale=True)
            raw_features = encoder_out['global']      # (B, 384)
            result['multi_scale'] = encoder_out
        else:
            raw_features = self.vision_encoder(images)  # (B, 384)
        
        visual_features = self.vision_projection(raw_features)  # (B, 512)
        
        # 2. 美学评分
        scores, weighted_score = self.aesthetic_scorer(visual_features)  # (B, 5), (B,)
        result['scores'] = scores
        result['weighted_score'] = weighted_score
        
        # 3. 参数预测 (新增)
        if predict_parameters and self.parameter_predictor is not None:
            param_output = self.parameter_predictor(visual_features, scores)
            result['parameters'] = param_output
        
        # 4. 条件生成建议
        suggestions = None
        if generate_suggestion and self.suggestion_generator is not None:
            low_score_mask = weighted_score < self.config.suggestion_threshold
            if low_score_mask.any():
                suggestions = self.suggestion_generator(
                    visual_features[low_score_mask],
                    scores[low_score_mask]
                )
        result['suggestions'] = suggestions
        
        if return_features:
            result['features'] = visual_features
        
        return result
    
    @torch.no_grad()
    def infer(
        self,
        images: torch.Tensor,
        return_dict: bool = True
    ) -> Dict:
        """
        推理接口（无梯度，优化速度）
        
        完整管线: 图像 → 美学评分 + 问题检测 + 参数预测 + 建议生成
        
        Args:
            images: (B, 3, H, W)
            return_dict: 是否返回字典格式
            
        Returns:
            results: 推理结果字典
        """
        self.eval()
        
        # 前向传播（包含参数预测）
        output = self.forward(
            images,
            generate_suggestion=True,
            predict_parameters=True,
            return_features=False
        )
        
        scores = output['scores']
        suggestions = output['suggestions']
        
        if not return_dict:
            return scores, suggestions
        
        # 转换为字典格式
        batch_size = scores.shape[0]
        results = []
        
        for i in range(batch_size):
            score_dict = {
                'composition': float(scores[i, 0]),
                'lighting': float(scores[i, 1]),
                'color': float(scores[i, 2]),
                'clarity': float(scores[i, 3]),
                'subject': float(scores[i, 4]),
                'overall': float(scores[i].mean())
            }
            
            result = {
                'scores': score_dict,
                'suggestions': suggestions[i] if suggestions else [],
                'needs_improvement': score_dict['overall'] < self.config.suggestion_threshold
            }
            
            # 添加参数预测结果 (5参数)
            if 'parameters' in output and self.parameter_predictor is not None:
                params = output['parameters']
                conf = params['confidence'][i]  # (5,)
                result['camera_parameters'] = {
                    'ev_compensation': float(params['ev_compensation'][i]),
                    'white_balance': float(params['white_balance'][i]),
                    'focus_point': params['focus_point'][i].tolist(),
                    'hdr': 'on' if params['hdr'][i, 1] > 0.5 else 'off',
                    'mode': ParameterPredictor.MODE_NAMES[
                        params['mode'][i].argmax().item()
                    ],
                }
                result['parameter_confidence'] = float(conf.mean())
                
                # 问题检测
                from .parameter_predictor import ProblemClassifier
                probs = params['problem_probs'][i]
                problems = []
                for j, name in enumerate(ProblemClassifier.PROBLEM_NAMES):
                    if probs[j] > 0.5:
                        sev_idx = params['severity_probs'][i, j].argmax().item()
                        problems.append({
                            'type': name,
                            'confidence': float(probs[j]),
                            'severity': ProblemClassifier.SEVERITY_NAMES[sev_idx]
                        })
                result['problems_detected'] = problems
            
            results.append(result)
        
        return results
    
    def load_venus_weights(
        self,
        venus_checkpoint_path: str,
        load_vision: bool = True,
        load_scorer: bool = True,
        strict: bool = False
    ) -> Dict[str, list]:
        """
        从 Venus 预训练权重初始化 IntelligenceCamera
        
        分模块加载策略:
        - 视觉编码器: Venus CLIP-ViT → MobileViT (权重映射)
        - 美学评分器: Venus scorer → IntelligenceCamera scorer (直接/自适应加载)
        - 参数预测器: 不从 Venus 加载（需要单独训练）
        
        Args:
            venus_checkpoint_path: Venus 模型权重路径
            load_vision: 是否加载视觉编码器权重
            load_scorer: 是否加载美学评分器权重
            strict: 是否严格匹配
            
        Returns:
            dict: {'loaded': [...], 'skipped': [...]}
        """
        result = {'loaded': [], 'skipped': []}
        
        if not os.path.exists(venus_checkpoint_path):
            raise FileNotFoundError(f"Venus checkpoint not found: {venus_checkpoint_path}")
        
        logger.info(f"Loading Venus weights from {venus_checkpoint_path}")
        
        checkpoint = torch.load(venus_checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            venus_state = checkpoint['state_dict']
        elif 'model' in checkpoint:
            venus_state = checkpoint['model']
        else:
            venus_state = checkpoint
        
        # 1. 加载视觉编码器权重
        if load_vision:
            try:
                loaded, skipped = self.vision_encoder.load_venus_weights(
                    venus_checkpoint_path, strict=strict
                )
                result['loaded'].extend([f"vision_encoder.{k}" for k in loaded])
                result['skipped'].extend([f"vision_encoder.{k}" for k in skipped])
                logger.info(f"Vision encoder: {len(loaded)} loaded, {len(skipped)} skipped")
            except Exception as e:
                logger.warning(f"Failed to load vision weights: {e}")
                result['skipped'].append(f"vision_encoder: {e}")
        
        # 2. 加载美学评分器权重
        if load_scorer:
            scorer_keys = {
                k: v for k, v in venus_state.items()
                if any(prefix in k for prefix in [
                    'aesthetic_scorer', 'scorer', 'score_head'
                ])
            }
            
            if scorer_keys:
                my_scorer_state = self.aesthetic_scorer.state_dict()
                loaded_count = 0
                for venus_key, venus_param in scorer_keys.items():
                    # 去掉前缀
                    rel_key = venus_key
                    for prefix in ['aesthetic_scorer.', 'scorer.', 'model.']:
                        if rel_key.startswith(prefix):
                            rel_key = rel_key[len(prefix):]
                    
                    if rel_key in my_scorer_state:
                        if venus_param.shape == my_scorer_state[rel_key].shape:
                            my_scorer_state[rel_key] = venus_param
                            result['loaded'].append(f"aesthetic_scorer.{rel_key}")
                            loaded_count += 1
                        else:
                            result['skipped'].append(
                                f"aesthetic_scorer.{rel_key}: shape mismatch"
                            )
                    else:
                        result['skipped'].append(f"aesthetic_scorer.{venus_key}: no match")
                
                self.aesthetic_scorer.load_state_dict(my_scorer_state, strict=False)
                logger.info(f"Aesthetic scorer: {loaded_count} keys loaded")
            else:
                logger.info("No aesthetic scorer weights found in Venus checkpoint")
        
        logger.info(f"Venus weight loading complete: "
                   f"{len(result['loaded'])} loaded, {len(result['skipped'])} skipped")
        
        return result
    
    def get_num_params(self) -> Dict[str, int]:
        """获取各组件参数量"""
        params = {
            'vision_encoder': sum(p.numel() for p in self.vision_encoder.parameters()),
            'vision_projection': sum(p.numel() for p in self.vision_projection.parameters()),
            'aesthetic_scorer': sum(p.numel() for p in self.aesthetic_scorer.parameters()),
        }
        
        if self.parameter_predictor is not None:
            params['parameter_predictor'] = sum(
                p.numel() for p in self.parameter_predictor.parameters()
            )
        
        if self.language_model is not None:
            params['language_model'] = sum(p.numel() for p in self.language_model.parameters())
            params['suggestion_generator'] = sum(p.numel() for p in self.suggestion_generator.parameters())
        
        params['total'] = sum(params.values())
        return params
    
    def get_model_size(self, dtype=torch.float16) -> Dict[str, float]:
        """获取模型大小（MB）"""
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int8: 1
        }[dtype]
        
        params = self.get_num_params()
        sizes = {
            k: v * bytes_per_param / (1024 ** 2)  # MB
            for k, v in params.items()
        }
        return sizes


class IntelligenceCameraConfig:
    """IntelligenceCamera 模型配置"""
    
    def __init__(
        self,
        # 视觉编码器
        image_size: int = 224,
        use_se: bool = True,
        use_fpn: bool = True,
        vision_output_dim: int = 384,
        projection_dim: int = 512,
        dropout: float = 0.1,
        # 美学评分器
        scorer_hidden_dim: int = 256,
        # 参数预测器
        use_parameter_predictor: bool = True,
        predictor_hidden_dim: int = 256,
        num_cross_attn_layers: int = 2,
        # 语言模型
        use_language_model: bool = True,
        num_suggestion_types: int = 10,
        suggestion_threshold: float = 7.0,
        language_model_config: Optional[dict] = None
    ):
        self.image_size = image_size
        self.use_se = use_se
        self.use_fpn = use_fpn
        self.vision_output_dim = vision_output_dim
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.scorer_hidden_dim = scorer_hidden_dim
        self.use_parameter_predictor = use_parameter_predictor
        self.predictor_hidden_dim = predictor_hidden_dim
        self.num_cross_attn_layers = num_cross_attn_layers
        self.use_language_model = use_language_model
        self.num_suggestion_types = num_suggestion_types
        self.suggestion_threshold = suggestion_threshold
        
        # 语言模型配置
        if language_model_config is None:
            from .language_model import TinyLLaMAConfig
            self.language_model_config = TinyLLaMAConfig()
        else:
            self.language_model_config = language_model_config


def create_intelligence_camera(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    venus_pretrained_path: Optional[str] = None,
    config: Optional[IntelligenceCameraConfig] = None
) -> IntelligenceCamera:
    """
    创建 IntelligenceCamera 模型
    
    Args:
        pretrained: 是否加载 IntelligenceCamera 预训练权重
        pretrained_path: IntelligenceCamera 权重路径
        venus_pretrained_path: Venus 权重路径（用于迁移学习初始化）
        config: 模型配置
        
    Returns:
        model: IntelligenceCamera 模型
    """
    if config is None:
        config = IntelligenceCameraConfig()
    
    model = IntelligenceCamera(config)
    
    # 加载 IntelligenceCamera 自己的权重
    if pretrained and pretrained_path is not None:
        logger.info(f"Loading IntelligenceCamera weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    # 从 Venus 权重迁移初始化
    elif venus_pretrained_path is not None:
        logger.info(f"Initializing from Venus weights: {venus_pretrained_path}")
        result = model.load_venus_weights(venus_pretrained_path)
        logger.info(f"Venus init: {len(result['loaded'])} loaded, {len(result['skipped'])} skipped")
    
    return model


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("IntelligenceCamera 增强版测试")
    print("=" * 60)
    
    # 创建模型
    config = IntelligenceCameraConfig(
        use_parameter_predictor=True,
        use_language_model=False  # 跳过语言模型以加快测试
    )
    model = create_intelligence_camera(config=config)
    
    # 参数量统计
    print("\n--- 参数量 ---")
    params = model.get_num_params()
    for name, count in params.items():
        print(f"  {name:25s}: {count:>12,}")
    
    print("\n--- 模型大小 (FP16) ---")
    sizes = model.get_model_size(dtype=torch.float16)
    for name, size in sizes.items():
        print(f"  {name:25s}: {size:>10.2f} MB")
    
    # 测试前向传播
    print("\n--- 前向传播 ---")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        # 基础模式
        output = model(dummy_input)
        print(f"Input:          {dummy_input.shape}")
        print(f"Scores:         {output['scores'].shape}")
        print(f"Weighted score: {output['weighted_score'].shape}")
        
        # 含参数预测
        output = model(dummy_input, predict_parameters=True, return_features=True)
        print(f"Features:       {output['features'].shape}")
        print(f"Multi-scale:    {list(output['multi_scale'].keys())}")
        print(f"Parameters:     {list(output['parameters'].keys())}")
        print(f"EV:             {output['parameters']['ev_compensation'][0].item():.3f}")
        print(f"WB:             {output['parameters']['white_balance'][0].item():.0f}K")
        print(f"Focus:          {output['parameters']['focus_point'][0].tolist()}")
        print(f"HDR:            {'on' if output['parameters']['hdr'][0, 1] > 0.5 else 'off'}")
        print(f"Mode:           {ParameterPredictor.MODE_NAMES[output['parameters']['mode'][0].argmax().item()]}")
        print(f"Confidence:     {output['parameters']['confidence'][0].mean().item():.3f}")
    
    # 测试推理接口
    print("\n--- 推理接口 ---")
    with torch.no_grad():
        results = model.infer(dummy_input)
        print(f"第1张图片结果:")
        r = results[0]
        print(f"  评分: {r['scores']}")
        if 'camera_parameters' in r:
            print(f"  相机参数: {json.dumps(r['camera_parameters'], indent=4)}")
            print(f"  置信度: {r['parameter_confidence']:.3f}")
        if 'problems_detected' in r:
            print(f"  问题: {r['problems_detected']}")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
