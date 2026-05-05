"""\u68c0\u67e5 transformers \u4e2d Qwen3-VL \u76f8\u5173\u7c7b\u7684\u53ef\u7528\u6027\u3002"""
import transformers
print(f'transformers: {transformers.__version__}')

for cls in [
    'Qwen3VLForConditionalGeneration',
    'Qwen3VLProcessor',
    'Qwen2_5_VLForConditionalGeneration',   # fallback
    'Qwen2VLForConditionalGeneration',      # fallback
    'AutoModelForCausalLM',
    'AutoProcessor',
]:
    try:
        getattr(transformers, cls)
        print(f'  [OK]   {cls}')
    except AttributeError:
        print(f'  [MISS] {cls}')
