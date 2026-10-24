"""Convert CodeFormer to Core ML (macOS only: prediction-based verification needs Core ML).

python convert_coreml.py <CodeFormer repo root> <output dir> <aligned test face>

Produces <output dir>/CodeFormer.mlpackage with
  input  image    : 512x512 RGB image (aligned face, FFHQ 5-point template)
  input  fidelity : float32[1], 0 = best quality, 1 = closest to the input identity
  output restored : 512x512 RGB image

Late generator activations reach ~3e4, so statistics in GroupNorm overflow pure fp16.
Precision variants are tried from smallest/fastest to safest; the first whose Core ML
output matches PyTorch (PSNR >= 30 dB) is kept. Results are printed as GitHub notices.
"""
import os
import shutil
import sys

import coremltools as ct
import numpy as np
import torch
from PIL import Image

from cf_loader import load_codeformer
from export_model import ExportCodeFormer

MIN_PSNR = 30.0
# Ops computing or applying normalisation statistics; kept in fp32 in the mixed variant.
FP32_OPS = {'reduce_mean', 'reduce_sum', 'reduce_l2_norm', 'square', 'pow', 'sqrt', 'rsqrt',
            'real_div', 'sub', 'mul', 'add', 'instance_norm', 'layer_norm', 'batch_norm'}

VARIANTS = [
    ('fp16', ct.precision.FLOAT16),
    ('mixed (norm ops fp32)', ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in FP32_OPS)),
    ('fp32', ct.precision.FLOAT32),
]


def notice(msg, level='notice'):
    print(f'::{level} title=Core ML conversion::{msg}', flush=True)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def convert(traced, precision):
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
        compute_precision=precision,
    )
    model.short_description = 'CodeFormer blind face restoration (aligned 512x512 face).'
    model.license = 'S-Lab License 1.0 (non-commercial). https://github.com/sczhou/CodeFormer'
    model.input_description['image'] = 'Aligned 512x512 RGB face'
    model.input_description['fidelity'] = 'w in [0, 1]: 0 = quality, 1 = fidelity'
    model.output_description['restored'] = 'Restored 512x512 RGB face'
    return model


def evaluate(model, wrapper, face):
    x = torch.from_numpy(np.asarray(face, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    scores, outputs = [], {}
    for w in (0.0, 0.5, 1.0):
        with torch.no_grad():
            ref = wrapper(x, torch.tensor([w]))[0].permute(1, 2, 0).round().numpy()
        pred = model.predict({'image': face, 'fidelity': np.array([w], dtype=np.float32)})['restored']
        got = np.asarray(pred.convert('RGB'), dtype=np.float32)
        scores.append(psnr(ref, got))
        outputs[w] = (got, float(got.std()))
    return scores, outputs


def main():
    src, out_dir, test_face = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    face = Image.open(test_face).convert('RGB').resize((512, 512))

    wrapper = ExportCodeFormer(load_codeformer(src)).eval()
    example = torch.rand(1, 3, 512, 512) * 2 - 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example, torch.tensor([0.5])))

    path = os.path.join(out_dir, 'CodeFormer.mlpackage')
    for name, precision in VARIANTS:
        model = convert(traced, precision)
        scores, outputs = evaluate(model, wrapper, face)
        detail = ', '.join(f'w={w:.1f}: {p:.1f} dB (pixel std {outputs[w][1]:.1f})'
                           for w, p in zip(outputs, scores))
        ok = min(scores) >= MIN_PSNR
        notice(f'{name}: {"PASS" if ok else "FAIL"} - {detail}')
        if ok:
            if os.path.exists(path):
                shutil.rmtree(path)
            model.save(path)
            for w, (img, _) in outputs.items():
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f'coreml_w{w:.1f}.png'))
            notice(f'saved {name} model')
            return 0
    notice('no precision variant matched PyTorch', level='error')
    return 1


if __name__ == '__main__':
    sys.exit(main())
