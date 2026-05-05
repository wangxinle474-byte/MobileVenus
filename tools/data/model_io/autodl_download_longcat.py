"""AutoDL 上下载 LongCat-Image-Edit-Turbo (走 ModelScope)."""
from modelscope import snapshot_download

CACHE = '/root/autodl-tmp/cache/modelscope'
MODEL = 'meituan-longcat/LongCat-Image-Edit-Turbo'

print(f'[DOWNLOAD] {MODEL} -> {CACHE}', flush=True)
path = snapshot_download(
    model_id=MODEL,
    cache_dir=CACHE,
    allow_patterns=[
        '*.json', '*.txt', '*.safetensors',
        '*.model', 'tokenizer*/*', '*.py',
    ],
)
print(f'[DONE] {path}', flush=True)
