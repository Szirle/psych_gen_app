"""Lazy Metal backend, independent of NVIDIA's CUDA plugin loader."""

from functools import lru_cache
from pathlib import Path
import sys

import torch


@lru_cache(None)
def get_plugin():
    if sys.platform != 'darwin' or not torch.backends.mps.is_available():
        raise RuntimeError('StyleGAN Metal ops require an available PyTorch MPS device')
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    plugin = load(
        name='stylegan_mps_ops', sources=[str(root / 'stylegan_ops.mm')],
        extra_cflags=['-O3', '-fno-objc-arc', '-Wno-deprecated-declarations'],
        extra_ldflags=['-framework', 'Metal', '-framework', 'Foundation'],
        with_cuda=False, verbose=False,
    )
    # Compile once through the system Metal runtime, without xcrun or a bundled
    # metallib tied to the build machine's macOS version.
    source = (root / 'stylegan_ops.h').read_text() + '\n' + (root / 'stylegan_ops.metal').read_text()
    plugin.initialize(source)
    return plugin
