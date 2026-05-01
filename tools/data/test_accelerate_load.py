"""
绕开 transformers from_pretrained, 直接用 accelerate API 加载 AesExpert-HF
对 8GB GPU + 16GB RAM 的低配 Windows 设置, fp16 + GPU/CPU 自动分配.
"""
import os
import sys
import time
import re
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch
from PIL import Image

MODEL_PATH = r'e:/AesExpert_HF'  # junction to bypass non-ASCII path issue
OFFLOAD_DIR = r'e:/aesexpert_offload'

TEST_IMAGES = [
    'E:/dataset/fivek_jpeg/a0001-jmac_DSC1459.jpg',
    'E:/dataset/fivek_jpeg/a0002-dgw_005.jpg',
    'E:/dataset/fivek_jpeg/a0003-NKIM_MG_8178.jpg',
    'E:/dataset/fivek_jpeg/a0004-jmac_MG_1384.jpg',
    'E:/dataset/fivek_jpeg/a0005-jn_2007_05_10__564.jpg',
]


def main():
    Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)

    print(f'torch: {torch.__version__}', flush=True)
    print(f'cuda: {torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'  GPU: {torch.cuda.get_device_name(0)} '
              f'free={free/1e9:.2f}GB total={total/1e9:.2f}GB', flush=True)

    print('\n[1] Loading config...', flush=True)
    from transformers import LlavaConfig
    cfg = LlavaConfig.from_pretrained(MODEL_PATH)
    print(f'  OK: vision={cfg.vision_config.model_type}, '
          f'text={cfg.text_config.model_type}', flush=True)

    print('\n[2] Initializing empty model on meta device...', flush=True)
    from accelerate import init_empty_weights
    from transformers import LlavaForConditionalGeneration

    with init_empty_weights():
        model = LlavaForConditionalGeneration(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Empty model OK: {n_params/1e9:.2f}B params', flush=True)

    print('\n[3] Dispatching weights via accelerate...', flush=True)
    from accelerate import load_checkpoint_and_dispatch

    t0 = time.time()
    try:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=MODEL_PATH,
            device_map='auto',
            max_memory={0: '6GiB', 'cpu': '13GiB'},
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
            dtype=torch.float16,
            no_split_module_classes=[
                'LlamaDecoderLayer',
                'CLIPEncoderLayer',
            ],
        )
    except Exception as e:
        print(f'  FAIL: {type(e).__name__}: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return

    print(f'  Loaded in {time.time()-t0:.1f}s', flush=True)
    if torch.cuda.is_available():
        print(f'  GPU mem allocated: '
              f'{torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

    # 显示 device map
    if hasattr(model, 'hf_device_map'):
        print(f'  device_map keys: {len(model.hf_device_map)}', flush=True)
        cpu_n = sum(1 for v in model.hf_device_map.values() if v == 'cpu')
        gpu_n = sum(1 for v in model.hf_device_map.values() if v != 'cpu')
        print(f'  layers on GPU: {gpu_n}, on CPU: {cpu_n}', flush=True)

    model.eval()

    print('\n[4] Loading processor...', flush=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f'  OK: {type(processor).__name__}', flush=True)

    print('\n[5] Running inference on test images...', flush=True)
    SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")
    USER_MSG = ("Please rate the aesthetic quality of this image on a scale of 1 to 10. "
                "Output the score in the format 'Score: X' where X is a decimal number.")

    results = []
    for img_path in TEST_IMAGES:
        if not Path(img_path).exists():
            print(f'  SKIP (not found): {img_path}', flush=True)
            continue

        pil = Image.open(img_path).convert('RGB')
        prompt = f"{SYSTEM} USER: <image>\n{USER_MSG} ASSISTANT:"

        try:
            inputs = processor(text=prompt, images=pil, return_tensors='pt')
            # 把输入放到模型第一参数所在的设备上 (通常是 cuda:0 或 cpu)
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) if torch.is_tensor(v) else v
                      for k, v in inputs.items()}
        except Exception as e:
            print(f'  process FAIL: {e}', flush=True)
            continue

        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id or 0,
                )
        except Exception as e:
            print(f'  generate FAIL: {type(e).__name__}: {e}', flush=True)
            import traceback
            traceback.print_exc()
            continue
        gen_time = time.time() - t0

        new_tokens = out[0][inputs['input_ids'].shape[-1]:]
        response = processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        nums = re.findall(r'(\d+(?:\.\d+)?)', response)
        score = None
        for n in nums:
            v = float(n)
            if 0 < v <= 10:
                score = v
                break

        print(f'\n  [{Path(img_path).name}] ({gen_time:.1f}s)', flush=True)
        print(f'    response: {response[:200]}', flush=True)
        print(f'    parsed score: {score}', flush=True)
        results.append({'img': str(img_path), 'response': response, 'score': score, 'time_s': gen_time})

    # 总结
    print('\n=== Summary ===', flush=True)
    if results:
        scores = [r['score'] for r in results if r['score'] is not None]
        if scores:
            print(f'  scored: {len(scores)}/{len(results)}', flush=True)
            print(f'  min={min(scores):.2f} max={max(scores):.2f} avg={sum(scores)/len(scores):.2f}', flush=True)
            n7plus = sum(1 for s in scores if s >= 7.0)
            print(f'  >=7.0: {n7plus}/{len(scores)}', flush=True)
        avg_time = sum(r['time_s'] for r in results) / len(results)
        print(f'  avg gen time: {avg_time:.1f}s', flush=True)

    print('\nDONE.', flush=True)


if __name__ == '__main__':
    main()
