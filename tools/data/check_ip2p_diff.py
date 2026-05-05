from PIL import Image
import numpy as np
import os

d = r'outputs/ip2p_test'
for i in range(3):
    o = Image.open(os.path.join(d, f'{i:04d}_orig.png')).convert('RGB')
    e = Image.open(os.path.join(d, f'{i:04d}_edit.png')).convert('RGB')
    if e.size != o.size:
        e_res = e.resize(o.size)
    else:
        e_res = e
    a = np.array(o, dtype=np.float32) / 255
    b = np.array(e_res, dtype=np.float32) / 255
    diff = np.abs(a - b).mean()
    psnr = 10 * np.log10(1.0 / max(((a-b)**2).mean(), 1e-10))
    da_mean = a.mean(axis=(0,1))
    db_mean = b.mean(axis=(0,1))
    delta_rgb = db_mean - da_mean
    print(f'[{i}] diff={diff:.4f} PSNR={psnr:.1f}dB  '
          f'RGB shift: R{delta_rgb[0]:+.3f} G{delta_rgb[1]:+.3f} B{delta_rgb[2]:+.3f}')
    print(f'    orig {o.size} -> edit {e.size}')
