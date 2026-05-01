"""
智能相机系统完整示例
展示如何使用 MobileVenus + 参数预测实现智能自动调参
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image

from models import (
    MobileVenus,
    MobileVenusConfig,
    RuleBasedParameterPredictor,
    ParameterValidator,
    ParameterSmoother
)
from inference import SmartCameraController


class SmartCameraSystem:
    """
    智能相机系统
    集成美学评分、问题识别、参数预测和自动调整
    """
    
    def __init__(self, platform='ios', auto_adjust=False):
        """
        Args:
            platform: 平台 ('ios' 或 'android')
            auto_adjust: 是否自动调整参数
        """
        print("初始化智能相机系统...")
        
        # 1. 美学评分模型
        print("  - 加载 MobileVenus 模型...")
        config = MobileVenusConfig()
        self.aesthetic_model = MobileVenus(config)
        self.aesthetic_model.eval()
        
        # 2. 参数预测器
        print("  - 初始化参数预测器...")
        self.param_predictor = RuleBasedParameterPredictor()
        
        # 3. 参数验证器
        self.param_validator = ParameterValidator()
        
        # 4. 参数平滑器
        self.param_smoother = ParameterSmoother(smoothing_factor=0.3)
        
        # 5. 相机控制器
        print("  - 初始化相机控制器...")
        self.camera_controller = SmartCameraController(
            platform=platform,
            auto_adjust=auto_adjust
        )
        
        # 问题类型映射
        self.problem_mapping = {
            0: 'poor_composition',
            1: 'underexposure',
            2: 'color_cast',
            3: 'blur',
            4: 'background_messy'
        }
        
        print("✅ 智能相机系统初始化完成！\n")
    
    def analyze_and_adjust(self, image: Image.Image) -> dict:
        """
        分析图像并自动调整相机参数
        
        Args:
            image: PIL Image
            
        Returns:
            result: 分析和调整结果
        """
        print("=" * 60)
        print("开始分析图像...")
        print("=" * 60)
        
        # 1. 预处理图像
        image_tensor = self._preprocess_image(image)
        
        # 2. 美学评分
        with torch.no_grad():
            output = self.aesthetic_model(
                image_tensor,
                generate_suggestion=False,
                return_features=True
            )
        scores = output['scores']
        
        scores_dict = {
            'composition': float(scores[0, 0]),
            'lighting': float(scores[0, 1]),
            'color': float(scores[0, 2]),
            'clarity': float(scores[0, 3]),
            'subject': float(scores[0, 4]),
            'overall': float(scores[0].mean())
        }
        
        print(f"\n📊 美学评分:")
        for dim, score in scores_dict.items():
            bar = '█' * int(score) + '░' * (10 - int(score))
            print(f"  {dim:12s}: {bar} {score:.1f}/10")
        
        # 3. 识别主要问题
        problem_type, severity = self._identify_problem(scores_dict)
        
        if problem_type is None:
            print(f"\n✅ 照片质量良好 (总分: {scores_dict['overall']:.1f}/10)")
            return {
                'scores': scores_dict,
                'problem': None,
                'parameters': None,
                'applied': False
            }
        
        print(f"\n⚠️  检测到问题: {problem_type} (严重程度: {severity})")
        
        # 4. 预测参数
        predicted_params = self.param_predictor.predict(
            problem_type,
            severity,
            scores_dict
        )
        
        print(f"\n🤖 AI 建议参数:")
        for param, value in predicted_params.items():
            print(f"  {param}: {value}")
        
        # 5. 验证参数
        validated_params, warnings = self.param_validator.validate(predicted_params)
        
        if warnings:
            print(f"\n⚠️  参数验证警告:")
            for warning in warnings:
                print(f"  - {warning}")
        
        # 6. 平滑参数
        smoothed_params = self.param_smoother.smooth(validated_params)
        
        print(f"\n✨ 平滑后参数:")
        for param, value in smoothed_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.2f}")
            else:
                print(f"  {param}: {value}")
        
        # 7. 应用参数
        suggestion = self.camera_controller.suggest_parameters(
            smoothed_params,
            f"{problem_type} ({severity})"
        )
        
        print(f"\n📝 参数变化:")
        for change in suggestion['changes']:
            print(f"  - {change['description']}")
        
        return {
            'scores': scores_dict,
            'problem': {
                'type': problem_type,
                'severity': severity
            },
            'parameters': {
                'predicted': predicted_params,
                'validated': validated_params,
                'smoothed': smoothed_params
            },
            'suggestion': suggestion,
            'applied': suggestion['auto_apply']
        }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transform(image).unsqueeze(0)
    
    def _identify_problem(self, scores: dict) -> tuple:
        """
        识别主要问题
        
        Returns:
            problem_type: 问题类型
            severity: 严重程度 ('mild', 'moderate', 'severe')
        """
        # 找出评分最低的维度
        min_dim = min(
            [(dim, score) for dim, score in scores.items() if dim != 'overall'],
            key=lambda x: x[1]
        )
        
        dim_name, min_score = min_dim
        
        # 如果最低分也不错，没有问题
        if min_score >= 7.0:
            return None, None
        
        # 确定严重程度
        if min_score >= 6.0:
            severity = 'mild'
        elif min_score >= 4.0:
            severity = 'moderate'
        else:
            severity = 'severe'
        
        # 映射到问题类型
        problem_mapping = {
            'composition': 'poor_composition',
            'lighting': 'underexposure' if min_score < 6.0 else 'low_contrast',
            'color': 'color_cast',
            'clarity': 'blur',
            'subject': 'background_messy'
        }
        
        problem_type = problem_mapping.get(dim_name, 'underexposure')
        
        return problem_type, severity


def demo_basic_usage():
    """基础使用示例"""
    print("\n" + "=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60 + "\n")
    
    # 创建系统
    system = SmartCameraSystem(platform='ios', auto_adjust=False)
    
    # 创建测试图像（实际应从相机获取）
    test_image = Image.new('RGB', (640, 480), color=(50, 50, 50))  # 暗图
    
    # 分析并调整
    result = system.analyze_and_adjust(test_image)
    
    print(f"\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    if result['problem']:
        print(f"\n问题: {result['problem']['type']}")
        print(f"严重程度: {result['problem']['severity']}")
        print(f"是否已应用: {result['applied']}")


def demo_continuous_adjustment():
    """连续调整示例（模拟实时拍摄）"""
    print("\n" + "=" * 60)
    print("示例 2: 连续调整（模拟实时拍摄）")
    print("=" * 60 + "\n")
    
    system = SmartCameraSystem(platform='ios', auto_adjust=True)
    
    # 模拟 5 帧图像
    print("模拟连续 5 帧图像分析...\n")
    
    for i in range(5):
        print(f"\n{'='*60}")
        print(f"帧 {i+1}/5")
        print(f"{'='*60}")
        
        # 模拟图像逐渐变亮
        brightness = 50 + i * 30
        test_image = Image.new('RGB', (640, 480), color=(brightness, brightness, brightness))
        
        result = system.analyze_and_adjust(test_image)
        
        print(f"\n总分: {result['scores']['overall']:.1f}/10")
        
        if i < 4:
            print("\n⏸️  等待下一帧...")
            import time
            time.sleep(0.5)


def demo_parameter_comparison():
    """参数对比示例"""
    print("\n" + "=" * 60)
    print("示例 3: 参数预测对比")
    print("=" * 60 + "\n")
    
    predictor = RuleBasedParameterPredictor()
    
    # 测试不同问题类型
    test_cases = [
        ('underexposure', 'mild'),
        ('underexposure', 'moderate'),
        ('underexposure', 'severe'),
        ('color_cast', 'warm'),
        ('background_messy', 'default'),
    ]
    
    for problem_type, severity in test_cases:
        params = predictor.predict(problem_type, severity)
        print(f"\n问题: {problem_type} ({severity})")
        print(f"建议参数: {params}")


if __name__ == "__main__":
    print("\n" + "🎥" * 30)
    print("MobileVenus 智能相机系统演示")
    print("🎥" * 30)
    
    # 运行示例
    demo_basic_usage()
    
    # demo_continuous_adjustment()
    
    # demo_parameter_comparison()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60 + "\n")
