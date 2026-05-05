"""\u68c0\u67e5 LongCat / QwenImageEdit pipeline \u5728\u5f53\u524d diffusers \u4e2d\u662f\u5426\u53ef\u7528."""
import diffusers
print(f'diffusers version: {diffusers.__version__}')

for cls in [
    'LongCatImageEditPipeline',
    'QwenImageEditPipeline',
    'QwenImageEditInpaintPipeline',
    'QwenImagePipeline',
    'QwenImageControlNetInpaintPipeline',
    'FireRedImageEditPipeline',
    'JoyAIImageEditPipeline',
]:
    try:
        obj = getattr(diffusers, cls)
        print(f'  [OK]   {cls}')
    except AttributeError:
        print(f'  [MISS] {cls}')
