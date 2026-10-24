"""Check ExportCodeFormer (eager and traced) against the original CodeFormer.forward.

python verify_wrapper.py <CodeFormer repo root> <aligned 512x512 face image> [weights]
"""
import sys

import numpy as np
import torch
from PIL import Image

from cf_loader import load_codeformer
from export_model import ExportCodeFormer


def to_input(path):
    img = np.asarray(Image.open(path).convert('RGB').resize((512, 512)), dtype=np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1


def main():
    src, face = sys.argv[1], sys.argv[2]
    net = load_codeformer(src, sys.argv[3] if len(sys.argv) > 3 else None)
    wrapper = ExportCodeFormer(net).eval()
    x = to_input(face)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x, torch.tensor([0.5])))
        worst = 0.0
        for w in (0.0, 0.5, 0.7, 1.0):
            ref = (net(x, w=w, adain=True)[0].clamp(-1, 1) + 1) * 127.5
            eager = wrapper(x, torch.tensor([w]))
            trace = traced(x, torch.tensor([w]))
            d1 = (ref - eager).abs().max().item()
            d2 = (ref - trace).abs().max().item()
            worst = max(worst, d1, d2)
            print(f'w={w:.1f}  max|orig-wrapper|={d1:.4f}  max|orig-traced|={d2:.4f}  (0-255 scale)')
    ok = worst < 0.5
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
