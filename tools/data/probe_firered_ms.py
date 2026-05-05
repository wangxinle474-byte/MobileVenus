"""\u76f4\u63a5\u8fde\u63a5 ModelScope \u7684\u72ec\u7acb gradio \u5730\u5740."""
from gradio_client import Client

URL = 'https://fireredteam-firered-image-edit-1-1.ms.show'

print(f'connecting: {URL}')
client = Client(URL, verbose=True)
print('\n=== API INFO ===')
info = client.view_api(return_format='dict')
for ep, d in info.get('named_endpoints', {}).items():
    print(f'\nendpoint: {ep}')
    for p in d.get('parameters', []):
        pt = p.get('python_type', {}).get('type', '?')
        print(f'  param: {p.get("parameter_name")!r}  label={p.get("label")!r}  type={pt}')
    for r in d.get('returns', []):
        pt = r.get('python_type', {}).get('type', '?')
        print(f'  returns: {r.get("label")!r}  type={pt}')
