"""
最终方案: 手动 bnb 4-bit 加载 AesExpert-HF 到 GPU.

策略:
  - vision_tower (CLIP, 0.6GB fp16) -> GPU
  - multi_modal_projector (~0.04GB fp16) -> GPU
  - language_model.embed_tokens (0.25GB) -> GPU
  - language_model.norm (small) -> GPU
  - language_model.layers.* Linear (q/k/v/o/gate/up/down) -> bnb.Linear4bit nf4 -> GPU (~3GB)
  - language_model.layers.* layernorms -> fp16 GPU (~256KB total)
  - lm_head (0.25GB) -> fp16 GPU

总 GPU: ~4.5GB, 8GB GPU 余 ~3.5GB 给激活+kv cache.

避开 mmap (Windows error 1455) + 避开 from_pretrained (segfault):
  - 用 open().seek().read() 逐 tensor 读, 一次只占一个 tensor 的 RAM
  - 用 init_empty_weights 构 meta model 后手动替换 Linear 为 bnb.Linear4bit
  - 用 bnb.nn.Params4bit(...).to('cuda:0') 自动量化
"""
import os
import sys
import time
import json
import re
import gc
import struct
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

MODEL_PATH = Path(r'e:/AesExpert_HF')

# 区分度测试: 3 张 AADB 最低 + 3 张 AADB 最高 (按 aug_full_aadb_scored.json)
TEST_IMAGES = [
    # 最低 AADB ~4.39
    'E:/dataset/fivek_jpeg/a3434-jmac_MG_5831.jpg',
    'E:/dataset/fivek_jpeg/a2252-jmac_MG_6404.jpg',
    'E:/dataset/fivek_jpeg/a0831-IMG_4991.jpg',  # 中等 4.75
    'E:/dataset/fivek_jpeg/a0973-WP_CRW_8501.jpg',  # 中等 4.75
    # 最高 AADB ~6.11
    'E:/dataset/fivek_jpeg/a3669-IMG_4072.jpg',
    'E:/dataset/fivek_jpeg/a4976-_DSC0005.jpg',
]

DTYPE_MAP = {
    'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
    'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16,
    'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool,
}

NP_DTYPE = {
    torch.int64: np.int64, torch.int32: np.int32,
    torch.int16: np.int16, torch.int8: np.int8,
    torch.uint8: np.uint8, torch.bool: np.bool_,
    torch.float32: np.float32, torch.float16: np.float16,
}


def parse_header(filepath: Path):
    with open(str(filepath), 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))
        header.pop('__metadata__', None)
    return header_len, header


def read_one_tensor(file_obj, info, data_offset):
    dtype = DTYPE_MAP[info['dtype']]
    shape = info['shape']
    beg, end = info['data_offsets']
    n_bytes = end - beg
    file_obj.seek(data_offset + beg)
    blob = file_obj.read(n_bytes)
    if dtype == torch.bfloat16:
        buf = np.frombuffer(blob, dtype=np.uint16).copy()
        t = torch.from_numpy(buf).view(torch.bfloat16).reshape(shape)
    elif dtype in NP_DTYPE:
        buf = np.frombuffer(blob, dtype=NP_DTYPE[dtype]).copy()
        t = torch.from_numpy(buf).reshape(shape)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    del blob, buf
    return t


def replace_linear_with_4bit(model, skip_paths):
    """递归把 nn.Linear 替换为 bnb.nn.Linear4bit (跳过 skip_paths 中的).
    构造时用 init_empty_weights 避免 CPU 上分配 fp32 临时权重 (会 segfault)."""
    import bitsandbytes as bnb
    from accelerate import init_empty_weights

    def is_skipped(full_name):
        return any(s in full_name for s in skip_paths)

    def _replace(module, prefix):
        for name, child in list(module.named_children()):
            full = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear) and not is_skipped(full):
                with init_empty_weights():
                    new = bnb.nn.Linear4bit(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type='nf4',
                        quant_storage=torch.uint8,
                    )
                setattr(module, name, new)
            else:
                _replace(child, full)

    _replace(model, '')


def get_module_param(model, key):
    """根据 key 返回 (module, param_name).  e.g. 'model.language_model.layers.0.self_attn.q_proj.weight' -> (q_proj_module, 'weight')."""
    *path, leaf = key.split('.')
    mod = model
    for p in path:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            return None, None
    return mod, leaf


def load_to_4bit_gpu(model, shards):
    """逐 shard 读, 把权重放到 GPU (Linear -> 4-bit, 其余 fp16)."""
    import bitsandbytes as bnb
    from accelerate.utils import set_module_tensor_to_device

    total_keys = 0
    for shard in shards:
        sz_gb = shard.stat().st_size / 1e9
        t0 = time.time()
        print(f'  Open {shard.name} ({sz_gb:.2f}GB)...', flush=True)
        header_len, header = parse_header(shard)
        data_offset = 8 + header_len
        n = len(header)
        print(f'    {n} tensors', flush=True)

        with open(str(shard), 'rb') as f:
            for i, (key, info) in enumerate(header.items()):
                t = read_one_tensor(f, info, data_offset)
                # 统一 fp16
                if t.dtype in (torch.bfloat16, torch.float32):
                    t = t.to(torch.float16)

                module, param_name = get_module_param(model, key)
                if module is None:
                    print(f'    SKIP unknown key: {key}', flush=True)
                    del t
                    continue

                if isinstance(module, bnb.nn.Linear4bit) and param_name == 'weight':
                    # 创建 Params4bit 并 .to('cuda') 自动量化
                    new_w = bnb.nn.Params4bit(
                        t.contiguous(),
                        requires_grad=False,
                        quant_type='nf4',
                    ).to('cuda:0')
                    module.weight = new_w
                    del t, new_w
                else:
                    # 用 accelerate 工具自动判断 Parameter / buffer
                    set_module_tensor_to_device(
                        model, key, 'cuda:0', value=t,
                    )
                    del t
                total_keys += 1

                if (i + 1) % 100 == 0:
                    gc.collect()
                    used = torch.cuda.memory_allocated() / 1e9
                    print(f'    {i+1}/{n} dispatched, GPU={used:.2f}GB',
                          flush=True)

        gc.collect()
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        print(f'  shard done in {time.time()-t0:.1f}s, GPU={used:.2f}GB',
              flush=True)

    print(f'\n  Total keys loaded: {total_keys}', flush=True)


def main():
    print(f'torch: {torch.__version__}', flush=True)
    if not torch.cuda.is_available():
        print('CUDA not available, abort.', flush=True)
        return

    free, total = torch.cuda.mem_get_info()
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'free={free/1e9:.2f}GB / total={total/1e9:.2f}GB', flush=True)

    print('\n[1] Loading config + processor...', flush=True)
    from transformers import LlavaConfig, LlavaForConditionalGeneration, AutoProcessor
    cfg = LlavaConfig.from_pretrained(str(MODEL_PATH))
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    # transformers 5.5+ LlavaProcessor 期望这两个属性, 但 AesExpert 的旧配置里没有
    if getattr(processor, 'patch_size', None) is None:
        processor.patch_size = cfg.vision_config.patch_size  # 14
    if getattr(processor, 'vision_feature_select_strategy', None) is None:
        processor.vision_feature_select_strategy = cfg.vision_feature_select_strategy
    # 'default' 策略 drop CLS, processor 内部 num_image_tokens 会 -1, 需 num_additional=1 保持 576
    processor.num_additional_image_tokens = 1  # 强制覆盖 (默认 0 导致 575)
    print(f'  patch_size={processor.patch_size}, '
          f'select={processor.vision_feature_select_strategy}, '
          f'num_additional={processor.num_additional_image_tokens}',
          flush=True)

    print('[2] Init empty model on meta...', flush=True)
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = LlavaForConditionalGeneration(cfg)

    print('[3] Replace Linear in language_model with bnb.Linear4bit...', flush=True)
    # 跳过 vision_tower (要保 fp16) 和 lm_head (输出层保 fp16 准确率)
    skip_paths = ['vision_tower', 'lm_head', 'multi_modal_projector']
    replace_linear_with_4bit(model, skip_paths=skip_paths)

    # 统计替换情况
    import bitsandbytes as bnb
    n_4bit = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f'  Linear4bit: {n_4bit}, Linear (fp16): {n_linear}', flush=True)

    print('[4] Stream load weights to GPU...', flush=True)
    shards = sorted(MODEL_PATH.glob('model-*.safetensors'))
    load_to_4bit_gpu(model, shards)

    model.tie_weights()
    model.eval()

    # 修复: embed_tokens 是 32000 行但 image_token_id=32000 越界
    # 把它扩到 32064 (LLaVA-1.5 标准, 多出的填零, 反正 image 位会被替换)
    img_id = cfg.image_token_index
    embed = model.model.language_model.embed_tokens
    if embed.weight.shape[0] <= img_id:
        new_n = max(img_id + 64, 32064)
        print(f'  Padding embed_tokens from {embed.weight.shape[0]} to {new_n}',
              flush=True)
        old_w = embed.weight.data
        new_w = torch.zeros(new_n, old_w.shape[1],
                            dtype=old_w.dtype, device=old_w.device)
        new_w[:old_w.shape[0]] = old_w
        embed.weight = nn.Parameter(new_w, requires_grad=False)
        embed.num_embeddings = new_n
        # lm_head 同步扩 (虽然 lm_head 输出 logits 不会预测 image_id, 但 shape 要一致)
        lmh = model.lm_head
        old_lh = lmh.weight.data
        if old_lh.shape[0] != new_n:
            new_lh = torch.zeros(new_n, old_lh.shape[1],
                                 dtype=old_lh.dtype, device=old_lh.device)
            new_lh[:old_lh.shape[0]] = old_lh
            lmh.weight = nn.Parameter(new_lh, requires_grad=False)
            lmh.out_features = new_n

    free, total = torch.cuda.mem_get_info()
    used = torch.cuda.memory_allocated() / 1e9
    print(f'\n  Final GPU: used={used:.2f}GB free={free/1e9:.2f}GB', flush=True)

    # 审计 dtype 一致性
    print('\n[4.5] Dtype audit:', flush=True)
    dtype_count = {}
    for n, p in model.named_parameters():
        d = str(p.dtype)
        dtype_count[d] = dtype_count.get(d, 0) + 1
        if 'bfloat16' in d or 'float32' in d:
            # 只打印异常的
            print(f'  NON-fp16: {n} {p.dtype} {p.shape}', flush=True)
    print(f'  Param dtype histogram: {dtype_count}', flush=True)

    # 强制把所有非 4bit 的 fp32/bf16 转 fp16 (修补任何遗漏)
    import bitsandbytes as bnb
    n_fixed = 0
    for n, p in model.named_parameters():
        if isinstance(p, bnb.nn.Params4bit):
            continue
        if p.dtype != torch.float16:
            with torch.no_grad():
                new_p = nn.Parameter(p.data.to(torch.float16),
                                      requires_grad=False)
            # 找父模块 + 名字, 替换
            *parents, leaf = n.split('.')
            mod = model
            for x in parents:
                mod = getattr(mod, x)
            setattr(mod, leaf, new_p)
            n_fixed += 1
    if n_fixed:
        print(f'  Fixed {n_fixed} non-fp16 params -> fp16', flush=True)

    print('\n[5] Sanity: vision_tower fwd on dummy image...', flush=True)
    try:
        dummy = torch.zeros(1, 3, 336, 336, dtype=torch.float16, device='cuda:0')
        with torch.no_grad():
            vt_out = model.model.vision_tower(dummy, output_hidden_states=True)
        print(f'  vision_tower OK, hidden[-2].shape={vt_out.hidden_states[-2].shape}',
              flush=True)
    except Exception as e:
        print(f'  vision_tower FAIL: {type(e).__name__}: {e}', flush=True)
        # 尝试关 cuDNN benchmark 模式
        torch.backends.cudnn.benchmark = False
        try:
            with torch.no_grad():
                vt_out = model.model.vision_tower(dummy, output_hidden_states=True)
            print(f'  vision_tower OK after disabling cudnn benchmark',
                  flush=True)
        except Exception as e2:
            print(f'  vision_tower still FAIL: {type(e2).__name__}: {e2}',
                  flush=True)
            torch.backends.cudnn.enabled = False

    print('\n[6] Inference on test images...', flush=True)
    SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")
    # 短直接 prompt: 只要数字 (不能给示例数字, 否则被抄)
    USER_MSG = (
        "What is the overall aesthetic quality score of this image on a scale "
        "from 1 to 10? Answer with only a single number."
    )

    results = []
    for img_path in TEST_IMAGES:
        if not Path(img_path).exists():
            print(f'  SKIP: {img_path}', flush=True)
            continue

        pil = Image.open(img_path).convert('RGB')
        prompt = f"{SYSTEM} USER: <image>\n{USER_MSG} ASSISTANT:"
        inputs = processor(text=prompt, images=pil, return_tensors='pt')
        # 强制 pixel_values 为 fp16 (CLIPImageProcessor 默认 fp32)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        inputs = {k: (v.to('cuda:0') if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}

        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id or 0,
                )
        except Exception as e:
            print(f'  generate FAIL on {Path(img_path).name}: '
                  f'{type(e).__name__}: {e}', flush=True)
            import traceback
            traceback.print_exc()
            continue
        gen_time = time.time() - t0

        new_tokens = out[0][inputs['input_ids'].shape[-1]:]
        response = processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        # 解析 Overall + 5 维分数
        scores_dim = {}
        for dim in ['composition', 'lighting', 'color', 'clarity', 'subject', 'overall']:
            m = re.search(rf'{dim}\s*[:\-=]\s*([\d.]+)', response, re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1).rstrip('.'))
                    scores_dim[dim] = max(0.0, min(10.0, v))
                except ValueError:
                    pass
        score = scores_dim.get('overall')
        if score is None:
            # 回退: 取任何 1-10 数字
            nums = re.findall(r'(\d+(?:\.\d+)?)', response)
            for n in nums:
                v = float(n)
                if 0 < v <= 10:
                    score = v
                    break

        print(f'\n  [{Path(img_path).name}] ({gen_time:.1f}s)', flush=True)
        print(f'    response: {response[:300]}', flush=True)
        print(f'    parsed: overall={score}, dims={scores_dim}', flush=True)
        results.append({'img': str(img_path), 'response': response,
                        'score': score, 'dims': scores_dim,
                        'time_s': gen_time})

    print('\n=== Summary ===', flush=True)
    if results:
        scores = [r['score'] for r in results if r['score'] is not None]
        if scores:
            print(f'  scored: {len(scores)}/{len(results)}', flush=True)
            print(f'  range: min={min(scores):.2f} max={max(scores):.2f} '
                  f'avg={sum(scores)/len(scores):.2f}', flush=True)
            n7 = sum(1 for s in scores if s >= 7.0)
            print(f'  >=7.0: {n7}/{len(scores)}', flush=True)
        avg_t = sum(r['time_s'] for r in results) / len(results)
        print(f'  avg gen time: {avg_t:.1f}s/image', flush=True)

        # 持久化结果
        out_path = Path('outputs/data/aesexpert_test_results.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'  saved: {out_path}', flush=True)

    print('\nDONE.', flush=True)


if __name__ == '__main__':
    main()
