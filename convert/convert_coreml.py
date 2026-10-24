"""Convert CodeFormer to Core ML (macOS only: prediction-based verification needs Core ML).

python convert_coreml.py <CodeFormer repo root> <output dir> [aligned test face]

Produces <output dir>/CodeFormer.mlpackage with
  input  image    : 512x512 RGB image (aligned face, FFHQ 5-point template)
  input  fidelity : float32[1], 0 = best quality, 1 = closest to the input identity
  output restored : 512x512 RGB image
"""
import os
import sys

import coremltools as ct
import numpy as np
import torch
from PIL import Image

from cf_loader import load_codeformer
from export_model import ExportCodeFormer


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    test_face = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(out_dir, exist_ok=True)

    wrapper = ExportCodeFormer(load_codeformer(src)).eval()
    example = torch.rand(1, 3, 512, 512) * 2 - 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example, torch.tensor([0.5])))

    model = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name='image', shape=(1, 3, 512, 512), scale=1 / 127.5, bias=[-1, -1, -1],
                         color_layout=ct.colorlayout.RGB),
            ct.TensorType(name='fidelity', shape=(1,), dtype=np.float32),
        ],
        outputs=[ct.ImageType(name='restored', color_layout=ct.colorlayout.RGB)],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    model.short_description = 'CodeFormer blind face restoration (aligned 512x512 face).'
    model.license = 'S-Lab License 1.0 (non-commercial). https://github.com/sczhou/CodeFormer'
    model.input_description['image'] = 'Aligned 512x512 RGB face'
    model.input_description['fidelity'] = 'w in [0, 1]: 0 = quality, 1 = fidelity'
    model.output_description['restored'] = 'Restored 512x512 RGB face'
    path = os.path.join(out_dir, 'CodeFormer.mlpackage')
    model.save(path)
    print('saved', path)

    if test_face:
        face = Image.open(test_face).convert('RGB').resize((512, 512))
        x = torch.from_numpy(np.asarray(face, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        worst = 99.0
        for w in (0.0, 0.5, 1.0):
            with torch.no_grad():
                ref = wrapper(x, torch.tensor([w]))[0].permute(1, 2, 0).round().numpy()
            pred = model.predict({'image': face, 'fidelity': np.array([w], dtype=np.float32)})['restored']
            got = np.asarray(pred.convert('RGB'), dtype=np.float32)
            p = psnr(ref, got)
            worst = min(worst, p)
            print(f'w={w:.1f}  PSNR(torch vs Core ML fp16) = {p:.2f} dB')
            Image.fromarray(got.astype(np.uint8)).save(os.path.join(out_dir, f'coreml_w{w:.1f}.png'))
        print('PASS' if worst >= 30 else 'FAIL (PSNR < 30 dB)')
        sys.exit(0 if worst >= 30 else 1)


if __name__ == '__main__':
    main()
