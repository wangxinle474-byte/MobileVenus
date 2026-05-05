"""\u5728 AutoDL \u4e0a\u9a8c\u8bc1 Qwen3-VL-4B \u6a21\u578b\u7684\u5b8c\u6574\u6027."""
import sys
from pathlib import Path

MODEL_DIR = Path('/root/autodl-tmp/models/Qwen3-VL-4B-Instruct')

print(f'=== Verifying {MODEL_DIR} ===', flush=True)

# \u5217\u51fa\u6587\u4ef6
files = list(MODEL_DIR.glob('*'))
safetensors = sorted(MODEL_DIR.glob('*.safetensors'))
print(f'Total files: {len(files)}', flush=True)
print(f'Safetensors: {len(safetensors)}', flush=True)
total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
print(f'Total weight size: {total_gb:.2f} GB', flush=True)
for f in safetensors:
    print(f'  {f.name}: {f.stat().st_size/1e9:.2f} GB', flush=True)

# \u68c0\u67e5 config
print('\n=== Config load test ===', flush=True)
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    print(f'OK arch: {cfg.architectures}', flush=True)
    if hasattr(cfg, 'text_config'):
        tc = cfg.text_config
        print(f'Text hidden={tc.hidden_size} layers={tc.num_hidden_layers}', flush=True)
    elif hasattr(cfg, 'hidden_size'):
        print(f'hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}', flush=True)
except Exception as e:
    print(f'[ERR] config load: {e}', flush=True)
    sys.exit(1)

# \u68c0\u67e5 safetensors \u53ef\u6253\u5f00
print('\n=== Safetensors header test ===', flush=True)
try:
    from safetensors import safe_open
    for f in safetensors:
        with safe_open(str(f), framework='pt') as h:
            keys = list(h.keys())
        print(f'  {f.name}: {len(keys)} tensors OK', flush=True)
except Exception as e:
    print(f'[ERR] safetensors: {e}', flush=True)
    sys.exit(2)

print('\n\u2705 Qwen3-VL-4B verified OK', flush=True)
