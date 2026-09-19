"""Load the same StyleGAN2 / early-output checkpoints that app.py uses.

Visualizer rendering still expects a G_ema-like object with mapping + synthesis.
This module resolves paths from config.yaml and wraps the project loaders so
those objects come from early_output_backend and gan_backend, not NVIDIA pickles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


def repo_root() -> Path:
    return _REPO_ROOT


def load_app_config() -> dict:
    config_path = _REPO_ROOT / 'config.yaml'
    if not config_path.is_file():
        return {}
    import yaml
    with open(config_path, 'r') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader) or {}


def _abs_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    return str(resolved.resolve())


def configured_model_paths() -> Dict[str, Optional[str]]:
    """Return the checkpoint paths app.py would consult."""
    config = load_app_config()
    models_path = _abs_path(config.get('models_path', 'models')) or str(_REPO_ROOT / 'models')
    resolution = os.environ.get(
        'STYLEGAN_EARLY_OUTPUT_RESOLUTION',
        config.get('stylegan_early_output_resolution'),
    )
    explicit_early = os.environ.get('STYLEGAN_EARLY_OUTPUT_PATH')
    early_256 = _abs_path(
        config.get('stylegan_early_output_256_path', os.path.join(models_path, 'sg256_pointwise_style32.pt'))
    )
    early_128 = _abs_path(
        config.get('stylegan_early_output_128_path', os.path.join(models_path, 'sg128_pointwise_style32.pt'))
    )
    if explicit_early:
        selected_early = _abs_path(explicit_early)
    elif resolution in (None, '', 'none', 'off', False):
        selected_early = None
    else:
        try:
            resolution = int(resolution)
        except (TypeError, ValueError):
            resolution = None
        if resolution == 256:
            selected_early = early_256
        elif resolution == 128:
            selected_early = early_128
        else:
            selected_early = None

    return {
        'models_path': models_path,
        'stylegan2': _abs_path(config.get('stylegan_path', os.path.join(models_path, 'stylegan2-ffhq-1024x1024.pkl'))),
        'early_output_128': early_128,
        'early_output_256': early_256,
        'early_output_default': selected_early,
        'distilled': _abs_path(config.get('stylegan_distilled_path')),
        'gan_device': os.environ.get('GAN_DEVICE', config.get('gan_device', 'auto')),
        'early_output_precision': os.environ.get(
            'GAN_PRECISION', config.get('early_output_precision', 'auto')
        ),
    }


def existing_local_models() -> List[str]:
    """Local checkpoints to populate the visualizer Recent list, preferred first."""
    paths = configured_model_paths()
    ordered = []
    for key in ('early_output_default', 'early_output_256', 'early_output_128', 'stylegan2'):
        path = paths.get(key)
        if path and os.path.isfile(path) and path not in ordered:
            ordered.append(path)
    return ordered


def models_browse_dir() -> Optional[str]:
    path = configured_model_paths().get('models_path')
    if path and os.path.isdir(path):
        return path
    return None


def is_early_output_checkpoint(path: str) -> bool:
    name = os.path.basename(path).lower()
    return path.lower().endswith('.pt') and ('pointwise_style32' in name or name.startswith('sg128') or name.startswith('sg256'))


def is_stylegan2_checkpoint(path: str) -> bool:
    return path.lower().endswith(('.pkl', '.pt')) and not is_early_output_checkpoint(path)


def load_network_dict(path: str, device) -> Dict[str, Any]:
    """Load a checkpoint into the {'G_ema': module} dict the visualizer expects."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if is_early_output_checkpoint(path):
        return {'G_ema': _load_early_output(path, device)}
    if path.lower().endswith('.pkl') or is_stylegan2_checkpoint(path):
        return {'G_ema': _load_stylegan2(path, device)}
    raise ValueError(f'Unrecognized local checkpoint: {path}')


def _load_early_output(path: str, device):
    from early_output_backend import EarlyOutputStyleGAN

    cfg = configured_model_paths()
    backend = EarlyOutputStyleGAN(
        path,
        device=device,
        precision=cfg.get('early_output_precision', 'auto'),
        compile=False,
        noise_mode='const',
    )
    G = backend.generator
    G.eval().requires_grad_(False)
    print(f'early-output {backend.describe()}')
    return G


def _load_stylegan2(path: str, device):
    # Build_model in gan_backend.py is the app.py loader, but that module currently
    # imports an unused distilled-head helper. Load with the same primitive it uses.
    errors = []
    try:
        from gan_backend import Build_model
        backend = Build_model(
            SimpleNamespace(network_pkl=path, distilled_network_pkl=None),
            device=str(device),
        )
        G = backend.G
        G.eval().requires_grad_(False)
        return G
    except Exception as error:
        errors.append(('gan_backend.Build_model', error))

    try:
        from utils import load_generator
        G, _dev = load_generator(path, device=str(device))
        G.eval().requires_grad_(False)
        print('Loaded StyleGAN2 via utils.load_generator')
        return G
    except Exception as error:
        errors.append(('utils.load_generator', error))

    try:
        from StyleGAN2_mps.early_output_model import _load_full_source_generator
        G = _load_full_source_generator(path).to(device)
        G.eval().requires_grad_(False)
        print('Loaded StyleGAN2 via StyleGAN2_mps.early_output_model')
        return G
    except Exception as error:
        errors.append(('StyleGAN2_mps.early_output_model', error))

    details = '; '.join(f'{name}: {err}' for name, err in errors)
    raise RuntimeError(f'Could not load StyleGAN2 checkpoint {path}: {details}')
