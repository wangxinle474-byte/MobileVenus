"""\u67e5 FireRed Image Edit + Qwen-Image \u76f8\u5173\u4ed3\u5e93\u6587\u4ef6\u5217\u8868\u4e0e\u603b\u5927\u5c0f."""
from modelscope.hub.api import HubApi


def list_repo_files(api, model_id, patterns):
    """\u5217\u51fa\u5339\u914d patterns \u7684\u6587\u4ef6 (size in MB)."""
    print(f'\n=== {model_id} ===')
    try:
        files = api.get_model_files(model_id, recursive=True)
    except Exception as e:
        print(f'  ERROR: {e}')
        return 0

    total = 0
    for f in files:
        path = f.get('Path', '')
        size = f.get('Size', 0)
        if any(path.startswith(p.rstrip('*')) or
               (p.endswith('*') and path.startswith(p[:-1])) or
               p in path for p in patterns):
            print(f'  {size/1e6:>8.1f} MB  {path}')
            total += size
    print(f'  ----- TOTAL ({len(patterns)} pattern(s)): {total/1e9:.2f} GB')
    return total


api = HubApi()

# 1.0 + Lightning LoRA
total_10 = list_repo_files(api, 'FireRedTeam/FireRed-Image-Edit-1.0',
                           ['transformer/'])
total_10_light = list_repo_files(api, 'FireRedTeam/FireRed-Image-Edit-1.0-Lightning',
                                 ['', '*.safetensors', '*.json'])

# 1.1 (\u53ef\u80fd\u6709\u5185\u7f6e Lightning, \u53ef\u80fd fp8)
total_11 = list_repo_files(api, 'FireRedTeam/FireRed-Image-Edit-1.1',
                           ['transformer/'])

# Qwen-Image (text_encoder + vae)
total_qwen = list_repo_files(api, 'Qwen/Qwen-Image',
                             ['text_encoder/', 'vae/'])

# Qwen-Image-Edit (processor)
total_qwen_edit = list_repo_files(api, 'Qwen/Qwen-Image-Edit',
                                  ['processor/'])

print('\n========== \u603b\u8ba1 ==========')
print(f'  1.0 + Lightning LoRA + Qwen base: '
      f'{(total_10 + total_10_light + total_qwen + total_qwen_edit)/1e9:.2f} GB')
print(f'  1.1 + Qwen base                 : '
      f'{(total_11 + total_qwen + total_qwen_edit)/1e9:.2f} GB')
