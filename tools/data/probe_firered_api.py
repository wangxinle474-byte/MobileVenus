"""\u63a2\u67e5 FireRed Space \u7684 Gradio API \u7aef\u70b9, \u4f9b\u540e\u7eed\u8c03\u7528."""
import os
import sys

# \u5982\u679c\u672c\u5730\u5728\u4e2d\u56fd, HF \u53ef\u80fd\u9700\u8981\u955c\u50cf
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from gradio_client import Client

CANDIDATES = [
    ('HF Space (\u5b98\u65b9 1.1)', 'FireRedTeam/FireRed-Image-Edit-1.1', 'hf'),
    ('HF Space (1.0-Fast Lightning)', 'prithivMLmods/FireRed-Image-Edit-1.0-Fast', 'hf'),
    ('ModelScope Studio (\u5b98\u65b9 1.0)', 'FireRedTeam/FireRed-Image-Edit-1.0', 'modelscope'),
]


def probe(label, repo, host_type):
    print(f'\n========== {label} ==========')
    print(f'  repo = {repo}')
    try:
        if host_type == 'hf':
            client = Client(repo)
        else:  # modelscope
            client = Client(f'https://modelscope.cn/api/v1/studio/{repo}/gradio/')
        print(f'  [OK] connected')

        # \u5217\u51fa API \u7aef\u70b9
        api_info = client.view_api(return_format='dict')
        for ep, info in api_info.get('named_endpoints', {}).items():
            params = info.get('parameters', [])
            print(f'  endpoint: {ep}')
            for p in params[:8]:
                print(f'    - {p.get("parameter_name", "?")}: {p.get("type", "?")}  '
                      f'({p.get("python_type", {}).get("type", "?")})')
            ret = info.get('returns', [])
            for r in ret[:3]:
                print(f'    -> returns: {r.get("type", "?")}')
        return client
    except Exception as e:
        print(f'  FAIL: {type(e).__name__}: {e}')
        return None


for label, repo, host_type in CANDIDATES:
    c = probe(label, repo, host_type)
    if c is not None:
        print(f'\n[\u2713 USE THIS] {label}')
        sys.exit(0)

print('\n[ALL FAILED] No accessible endpoint')
sys.exit(1)
