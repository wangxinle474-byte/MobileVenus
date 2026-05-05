"""检查 VLM 原始输出, 诊断为什么 parse 失败."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/aug_ip2p_with_vlm_params.json'
d = json.load(open(path, encoding='utf-8'))
print(f"Total: {d['num_processed']}  parsed: {d['num_vlm_parsed']}")

for i, r in enumerate(d['results'][:3]):
    print('=' * 70)
    print(f"[{r['idx']}] {r['image']}  delta=+{r['delta']:.1f}")
    print('--- prompt fragment (venus) ---')
    print(r['edit_prompt'][:150])
    print('--- VLM raw output ---')
    print(r['vlm_output_full'])
    print()
    print(f"parsed params: {r['vlm_params']}")
    print()
