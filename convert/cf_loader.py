"""Load the CodeFormer network from the official source tree using only PyTorch.

Importing `basicsr` normally pulls in its whole training stack, so this registers
minimal stand-ins for the two helpers the arch files need and loads just
vqgan_arch.py and codeformer_arch.py.
"""
import importlib.util
import logging
import os
import sys
import types

import torch

WEIGHTS_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'


class _Registry:
    def __init__(self):
        self._items = {}

    def register(self, obj=None):
        def deco(o):
            self._items[o.__name__] = o
            return o
        return deco if obj is None else deco(obj)

    def get(self, name):
        return self._items[name]


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_codeformer(src_dir, weights_path=None):
    """src_dir: CodeFormer repo root (contains basicsr/archs). Returns the eval-mode net."""
    archs = os.path.join(src_dir, 'basicsr', 'archs')
    registry = _Registry()
    _module('basicsr', __path__=[])
    _module('basicsr.utils', __path__=[], get_root_logger=lambda *a, **k: logging.getLogger('basicsr'))
    _module('basicsr.utils.registry', ARCH_REGISTRY=registry)
    _module('basicsr.archs', __path__=[])
    _load_file('basicsr.archs.vqgan_arch', os.path.join(archs, 'vqgan_arch.py'))
    _load_file('basicsr.archs.codeformer_arch', os.path.join(archs, 'codeformer_arch.py'))

    net = registry.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                     connect_list=['32', '64', '128', '256'])
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'codeformer.pth')
        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.hub.download_url_to_file(WEIGHTS_URL, weights_path)
    net.load_state_dict(torch.load(weights_path, map_location='cpu')['params_ema'])
    return net.eval()
