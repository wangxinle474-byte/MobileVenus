"""
python -m training.fivek_8param [--epochs N] [--batch_size N] [--jpeg_dir PATH] ...
"""
import argparse
from .config import TrainConfig
from .trainer import train


def main():
    parser = argparse.ArgumentParser(description='FiveK 8-param Training')
    cfg = TrainConfig()

    for field_name, default_val in vars(cfg).items():
        t = type(default_val)
        if t == bool:
            parser.add_argument(f'--{field_name}', type=lambda x: x.lower() == 'true',
                                default=default_val)
        else:
            parser.add_argument(f'--{field_name}', type=t, default=default_val)

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == '__main__':
    main()
