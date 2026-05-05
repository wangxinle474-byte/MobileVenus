"""最小测试: 只加载 LongCat pipeline 看是否成功. 用 -u 实时输出."""
import os
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('https_proxy', None)
os.environ.pop('http_proxy', None)
os.environ['HF_HOME'] = r'E:\cache\huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'E:\cache\huggingface\hub'

MODEL_PATH = r'E:\cache\modelscope\meituan-longcat\LongCat-Image-Edit-Turbo'

print(f'[1/6] importing torch...', flush=True)
import torch
print(f'      torch {torch.__version__}, cuda={torch.cuda.is_available()}', flush=True)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f'      GPU {torch.cuda.get_device_name(0)} '
          f'free={free/1e9:.1f}GB total={total/1e9:.1f}GB', flush=True)

print(f'[2/6] importing diffusers...', flush=True)
from diffusers import LongCatImageEditPipeline
import diffusers
print(f'      diffusers {diffusers.__version__}', flush=True)

print(f'[3/6] from_pretrained({MODEL_PATH})...', flush=True)
t0 = time.time()
pipe = LongCatImageEditPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
print(f'      from_pretrained OK ({time.time()-t0:.1f}s)', flush=True)

print(f'[4/6] enable_sequential_cpu_offload...', flush=True)
t0 = time.time()
pipe.enable_sequential_cpu_offload()
print(f'      offload OK ({time.time()-t0:.1f}s)', flush=True)

print(f'[5/6] vae enable_tiling/slicing...', flush=True)
try:
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print(f'      OK', flush=True)
except AttributeError as e:
    print(f'      skipped: {e}', flush=True)

print(f'[6/6] All loaded. Final GPU mem:', flush=True)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f'      free={free/1e9:.1f}GB used={(total-free)/1e9:.1f}GB', flush=True)

print(f'\n[DONE] Pipeline ready. Exit cleanly.', flush=True)
