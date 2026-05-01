"""推理接口: 图片 → 6 个 ISP 参数。

支持:
- 纯视觉推理 (Stage B)
- 文本条件化推理 (Stage C)
- 批量推理
- 参数渲染预览

用法:
    from inference.predict import Predictor

    predictor = Predictor('checkpoints/distill_v8/stage_b/best.pt')
    params = predictor.predict('photo.jpg')
    preview = predictor.predict_and_render('photo.jpg')
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from training.fivek_8param.config import PARAM_NAMES
from training.semantic_distill.model import DistillParamModel
from models.isp_pipeline import render_params


class Predictor:
    """ISP 参数预测器。

    Args:
        checkpoint: Stage B 模型权重路径
        device: 推理设备
        image_size: 输入图片大小
    """

    def __init__(self, checkpoint, device='cpu', image_size=224):
        self.device = device
        self.image_size = image_size

        # 加载模型
        self.model = DistillParamModel(image_size=image_size)
        state = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(device).eval()

        # 图片预处理
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.raw_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _load_image(self, image_path):
        """加载并预处理图片。"""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        raw = self.raw_transform(img).unsqueeze(0).to(self.device)
        return tensor, raw

    @torch.no_grad()
    def predict(self, image_path):
        """预测 6 个 ISP 参数。

        Args:
            image_path: 图片路径
        Returns:
            dict: {param_name: float_value}
        """
        tensor, _ = self._load_image(image_path)
        out = self.model(tensor)

        params = {}
        for name in PARAM_NAMES:
            params[name] = out['raw_params'][name][0].item()

        return params

    @torch.no_grad()
    def predict_batch(self, image_paths):
        """批量预测。"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    @torch.no_grad()
    def predict_and_render(self, image_path, output_path=None):
        """预测参数并渲染预览图。

        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径 (None = 不保存)
        Returns:
            params: dict
            rendered: PIL.Image (渲染后)
        """
        tensor, raw = self._load_image(image_path)
        out = self.model(tensor)

        # 渲染
        rendered = render_params(raw, out['raw_params'])
        rendered_np = rendered[0].cpu().numpy().transpose(1, 2, 0)
        rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
        rendered_pil = Image.fromarray(rendered_np)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            rendered_pil.save(output_path)

        params = {}
        for name in PARAM_NAMES:
            params[name] = out['raw_params'][name][0].item()

        return params, rendered_pil

    def print_params(self, params):
        """格式化打印参数。"""
        print("Predicted ISP Parameters:")
        print("-" * 40)
        for name in PARAM_NAMES:
            val = params[name]
            if name == 'white_balance':
                print(f"  {name:20s}: {val:>8.0f} K")
            elif name == 'ev_compensation':
                print(f"  {name:20s}: {val:>+8.2f} EV")
            else:
                print(f"  {name:20s}: {val:>8.1f}")


class TextConditionedPredictor(Predictor):
    """文本条件化推理 (Stage C)。"""

    def __init__(self, checkpoint, vocab_path=None, device='cpu',
                 image_size=224):
        self.device = device
        self.image_size = image_size

        from training.text_condition.model import TextConditionedModel
        self.model = TextConditionedModel()
        state = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'], strict=False)
        self.model = self.model.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.raw_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # 简易字符 tokenizer
        self._build_vocab(vocab_path)

    def _build_vocab(self, vocab_path):
        """构建字符级词表。"""
        if vocab_path and os.path.exists(vocab_path):
            import json
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.char2id = json.load(f)
        else:
            # 基础中文 + 英文 + 数字字符
            chars = list("abcdefghijklmnopqrstuvwxyz0123456789 ，。！？"
                         "的一是不了人我在有他这中大来上个国和也地到"
                         "说时要就出会对能那好点都比太更暗亮冷暖色调"
                         "提高降低增加减少一些很非常稍微")
            self.char2id = {c: i + 1 for i, c in enumerate(chars)}
        self.char2id.setdefault('[PAD]', 0)
        self.char2id.setdefault('[CLS]', len(self.char2id))

    def tokenize(self, text, max_len=64):
        """字符级 tokenize。"""
        ids = [self.char2id.get('[CLS]', 1)]
        for ch in text.lower()[:max_len - 1]:
            ids.append(self.char2id.get(ch, 1))
        # Padding
        while len(ids) < max_len:
            ids.append(0)
        return torch.tensor([ids], dtype=torch.long)

    @torch.no_grad()
    def predict_with_text(self, image_path, text_instruction):
        """文本条件化参数预测。

        Args:
            image_path: 图片路径
            text_instruction: 口语化指令 (如 "有点暗，提亮一些")
        Returns:
            dict: {param_name: float_value}
        """
        tensor, _ = self._load_image(image_path)
        text_ids = self.tokenize(text_instruction).to(self.device)

        out = self.model(tensor, text_ids)

        params = {}
        for name in PARAM_NAMES:
            params[name] = out['raw_params'][name][0].item()
        return params


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m inference.predict <checkpoint> <image>")
        sys.exit(1)

    predictor = Predictor(sys.argv[1])
    params = predictor.predict(sys.argv[2])
    predictor.print_params(params)
