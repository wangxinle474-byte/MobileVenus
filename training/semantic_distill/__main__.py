"""
python -m training.semantic_distill embed      # 提取文本embedding
python -m training.semantic_distill stage_a    # 阶段A语义对齐
python -m training.semantic_distill stage_b    # 阶段B参数微调
python -m training.semantic_distill all        # 全流程
"""
import argparse
from .config import DistillConfig


def main():
    parser = argparse.ArgumentParser(description='Semantic Distillation Training')
    parser.add_argument('stage', choices=['embed', 'stage_a', 'stage_b', 'all'],
                        help='Which stage to run')
    cfg = DistillConfig()
    for field_name, default_val in vars(cfg).items():
        t = type(default_val)
        if t == bool:
            parser.add_argument(f'--{field_name}', type=lambda x: x.lower() == 'true',
                                default=default_val)
        else:
            parser.add_argument(f'--{field_name}', type=t, default=default_val)

    args = parser.parse_args()
    stage = args.stage
    cfg_dict = {k: v for k, v in vars(args).items() if k != 'stage'}
    cfg = DistillConfig(**cfg_dict)

    if stage in ('embed', 'all'):
        from .embed_texts import main as embed_main
        import sys
        sys.argv = ['embed_texts',
                     '--venus_json', cfg.venus_json,
                     '--output', cfg.text_embed_file,
                     '--model', cfg.text_model_name]
        embed_main()

    if stage in ('stage_a', 'all'):
        from .trainer import train_stage_a
        train_stage_a(cfg)

    if stage in ('stage_b', 'all'):
        from .trainer import train_stage_b
        train_stage_b(cfg)


if __name__ == '__main__':
    main()
