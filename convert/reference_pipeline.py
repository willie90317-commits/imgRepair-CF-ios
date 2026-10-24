"""Python reference of the on-device pipeline implemented in Swift (FaceRestorer.swift).

Uses facelib (from the desktop project) only to obtain 5 landmarks, standing in for
iOS Vision; everything after that mirrors the Swift code:
  least-squares similarity -> 512 crop -> CodeFormer -> square soft-mask paste-back.

python reference_pipeline.py <desktop app dir> <image> <out dir>
"""
import os
import sys

import cv2
import numpy as np
import torch

from export_model import ExportCodeFormer

TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float64)
BORDER = (135, 133, 132)


def similarity(src, dst):
    """Least-squares similarity (rotation + uniform scale + translation), 2x3 matrix."""
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    var = (s ** 2).sum()
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum() / var
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / var
    tx = dm[0] - (a * sm[0] - b * sm[1])
    ty = dm[1] - (b * sm[0] + a * sm[1])
    return np.array([[a, -b, tx], [b, a, ty]])


def paste_back(base, face, inv):
    h, w = base.shape[:2]
    inv_face = cv2.warpAffine(face, inv, (w, h))
    mask = cv2.warpAffine(np.ones((512, 512), np.float32), inv, (w, h))
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
    edge = int(mask.sum() ** 0.5) // 20
    center = cv2.erode(mask, np.ones((edge * 2, edge * 2), np.uint8))
    soft = cv2.GaussianBlur(center, (edge * 2 + 1, edge * 2 + 1), 0)[:, :, None]
    return soft * (mask[:, :, None] * inv_face) + (1 - soft) * base


def main():
    app_dir, image_path, out_dir = sys.argv[1:4]
    sys.path.insert(0, app_dir)
    import cf_engine as eng
    os.makedirs(out_dir, exist_ok=True)

    engine = eng.CodeFormerEngine('auto', log=lambda m: None)
    img = eng.imread(image_path)
    helper = engine._get_helper('retinaface_resnet50', 1)
    helper.clean_all()
    helper.read_image(img)
    helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
    base = helper.input_img.astype(np.float32)   # facelib may upscale tiny inputs; use the same base
    landmarks = [np.asarray(l, np.float64) for l in helper.all_landmarks_5]
    print('faces:', len(landmarks))

    wrapper = ExportCodeFormer(engine._net('restoration')).to(engine.device).eval()
    result = base.copy()
    for i, lm in enumerate(landmarks):
        m = similarity(lm, TEMPLATE)
        m_cv = cv2.estimateAffinePartial2D(lm, TEMPLATE, method=cv2.LMEDS)[0]
        print(f'face {i}: max |LSQ - cv2 LMEDS| matrix diff = {np.abs(m - m_cv).max():.4f}')
        crop = cv2.warpAffine(base, m, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER)
        rgb = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(engine.device) / 127.5 - 1
        with torch.no_grad():
            out = wrapper(rgb, torch.tensor([0.7], device=engine.device))[0]
        face = out.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].round()
        result = paste_back(result, face, cv2.invertAffineTransform(m))

    ref = engine.restore(img, eng.RestoreOptions(fidelity=0.7, upscale=1, bg_upsample=False))[0].astype(np.float32)
    mse = np.mean((result - ref) ** 2)
    print(f'PSNR(iOS pipeline vs official, whole image) = {10 * np.log10(255 ** 2 / mse):.2f} dB')
    eng.imwrite(os.path.join(out_dir, 'ios_reference.png'), result.clip(0, 255).astype(np.uint8))
    eng.imwrite(os.path.join(out_dir, 'official.png'), ref.astype(np.uint8))


if __name__ == '__main__':
    main()
