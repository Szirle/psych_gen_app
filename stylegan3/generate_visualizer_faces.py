"""Visualizer seed faces and W-space seed matching.

generate: JPEG faces from the StyleGAN3 visualizer seed → Z → W path.
match: mapping-only search of seeds against psychGAN photo_to_coords W vectors.

The Latent widget shows an integer index:

    seed = round(x) + round(y) * 100          # viz/latent_widget.py

Typing 0, 1, 2, … sets (x, y) = (seed, 0). At those integer positions:

    rnd = numpy.random.RandomState(seed)
    z = rnd.randn(G.z_dim)

Examples:

    /opt/anaconda3/envs/manip311/bin/python generate_visualizer_faces.py generate \\
        --outdir ../visualizer_faces --seeds 0-1003

    /opt/anaconda3/envs/manip311/bin/python generate_visualizer_faces.py match \\
        --batch-size 2000 --batches 5000000
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import click
import numpy as np
import PIL.Image
import torch

_STYLEGAN3_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _STYLEGAN3_DIR.parent
if str(_STYLEGAN3_DIR) not in sys.path:
    sys.path.insert(0, str(_STYLEGAN3_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from torch_utils.device import get_device
from viz.local_models import load_network_dict

# LatentWidget.step_y — vertical grid stride used to pack (x, y) into a seed.
STEP_Y = 100
DEFAULT_NETWORK = str(_REPO_ROOT / 'models' / 'stylegan2-ffhq-1024x1024.pkl')
DEFAULT_DATA_PATH = str(_REPO_ROOT / 'data')
DATASET_PHOTO_COUNT = 1004
CALIBRATE_SEED = 1
CUTOFF_MULTIPLIER = 3.0
MATCH_TRUNC_PSI = 0.7
DEFAULT_SOURCE_IMAGES = Path('/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images')
DEFAULT_VALIDATE_DIR = str(_REPO_ROOT / 'visualizer_w_validate')


# ---------------------------------------------------------------------------
# Exact copies of the visualizer seed / Z / W construction.
# ---------------------------------------------------------------------------

def displayed_seed(x: float, y: float, step_y: int = STEP_Y) -> int:
    """Integer shown in the Latent widget: round(x) + round(y) * step_y."""
    return int(round(x) + round(y) * step_y)


def w0_seeds_from_xy(x: float, y: float, step_y: int = STEP_Y) -> List[List[float]]:
    """Bilinear mix of the four grid-corner seeds, copied from LatentWidget.

    For an integer UI index N typed into the seed box, (x, y) = (N, 0) and
    this returns ``[[N, 1.0]]``.
    """
    w0_seeds: List[List[float]] = []
    for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        seed_x = np.floor(x) + ofs_x
        seed_y = np.floor(y) + ofs_y
        seed = (int(seed_x) + int(seed_y) * step_y) & ((1 << 32) - 1)
        weight = (1 - abs(x - seed_x)) * (1 - abs(y - seed_y))
        if weight > 0:
            w0_seeds.append([seed, weight])
    return w0_seeds


def z_from_seed(seed: int, z_dim: int) -> np.ndarray:
    """Visualizer Z for one integer seed: RandomState(seed).randn(z_dim)."""
    rnd = np.random.RandomState(int(seed))
    return rnd.randn(z_dim).astype(np.float32)


def zs_from_seeds(seeds: Sequence[int], z_dim: int) -> np.ndarray:
    """Stack visualizer Z vectors for ``seeds`` in that order."""
    all_zs = np.zeros([len(seeds), z_dim], dtype=np.float32)
    for idx, seed in enumerate(seeds):
        all_zs[idx] = z_from_seed(seed, z_dim)
    return all_zs


def w0_seeds_for_ui_index(index: int) -> List[List[float]]:
    """Seeds/weights for Latent widget index ``index`` with default y=0."""
    return w0_seeds_from_xy(float(index), 0.0)


def mapping_latents(
    G,
    w0_seeds: Sequence[Sequence[float]],
    *,
    trunc_psi: float = 1.0,
    trunc_cutoff: Optional[int] = None,
    stylemix_seed: int = 0,
    stylemix_idx: Optional[Sequence[int]] = None,
    device: torch.device,
) -> torch.Tensor:
    """Reproduce renderer._render_impl mapping: Z → mixed W.

    Copied from ``viz/renderer.py`` (Generate random latents / Run mapping /
    Calculate final W). Stylemix is off in the default GUI (no layer
    checkboxes), which is the ``stylemix_idx=[]`` path.
    """
    stylemix_idx = list(stylemix_idx or [])
    all_seeds = [int(seed) for seed, _weight in w0_seeds] + [int(stylemix_seed)]
    all_seeds = list(set(all_seeds))
    z_dim = int(G.z_dim)
    c_dim = int(getattr(G, 'c_dim', 0))
    all_zs = np.zeros([len(all_seeds), z_dim], dtype=np.float32)
    all_cs = np.zeros([len(all_seeds), c_dim], dtype=np.float32)
    for idx, seed in enumerate(all_seeds):
        rnd = np.random.RandomState(seed)
        all_zs[idx] = rnd.randn(z_dim)
        if c_dim > 0:
            all_cs[idx, rnd.randint(c_dim)] = 1

    cutoff = G.num_ws if trunc_cutoff is None else trunc_cutoff
    w_avg = G.mapping.w_avg
    all_zs_t = torch.from_numpy(all_zs).to(device)
    all_cs_t = torch.from_numpy(all_cs).to(device)
    all_ws = G.mapping(
        all_zs_t, all_cs_t, truncation_psi=trunc_psi, truncation_cutoff=cutoff
    ) - w_avg
    all_ws = dict(zip(all_seeds, all_ws))

    w = torch.stack(
        [all_ws[int(seed)] * float(weight) for seed, weight in w0_seeds]
    ).sum(dim=0, keepdim=True)
    stylemix_idx = [idx for idx in stylemix_idx if 0 <= idx < G.num_ws]
    if len(stylemix_idx) > 0:
        w[:, stylemix_idx] = all_ws[int(stylemix_seed)][np.newaxis, stylemix_idx]
    w = w + w_avg
    return w


def image_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Visualizer RGB conversion: (img * 127.5 + 128).clamp(0, 255)."""
    img = img.to(torch.float32)
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
    return img.detach().cpu().numpy()


def synthesize(
    G,
    w: torch.Tensor,
    *,
    noise_mode: str = 'const',
    random_seed: int = 0,
    force_fp32: bool = True,
) -> torch.Tensor:
    """Run G.synthesis the way the visualizer does (const noise, seeded RNG)."""
    torch.manual_seed(int(random_seed))
    kwargs = dict(noise_mode=noise_mode, force_fp32=force_fp32)
    try:
        return G.synthesis(w, **kwargs)
    except TypeError:
        kwargs.pop('force_fp32', None)
        return G.synthesis(w, **kwargs)


def first_w512(w) -> torch.Tensor:
    """Take the first 512-d W vector from a stored coord (handles [512] or [18, 512])."""
    t = torch.as_tensor(w, dtype=torch.float32).reshape(-1)
    if t.numel() == 0:
        raise ValueError('empty latent')
    if t.numel() >= 18 * 512 and t.numel() % 512 == 0:
        return t.view(-1, 512)[0].contiguous()
    if t.numel() < 512:
        raise ValueError(f'latent has {t.numel()} values, need 512')
    return t[:512].contiguous()


def stack_dataset_w(photo_to_coords, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    """Stack ``1.jpg`` … ``1004.jpg`` as [1004, 512] on ``device``."""
    names = [f'{i}.jpg' for i in range(1, DATASET_PHOTO_COUNT + 1)]
    missing = [name for name in names if name not in photo_to_coords]
    if missing:
        raise click.ClickException(
            f'missing {len(missing)} photo_to_coords keys, e.g. {missing[:5]}'
        )
    rows = [first_w512(photo_to_coords[name]) for name in names]
    return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32), names


def map_w0_batch(
    G,
    seeds: Sequence[int],
    *,
    trunc_psi: float,
    trunc_cutoff: int,
    device: torch.device,
) -> torch.Tensor:
    """Map integer visualizer seeds to the first W vector, batched, no grad."""
    z = torch.from_numpy(zs_from_seeds(seeds, int(G.z_dim))).to(device)
    c_dim = int(getattr(G, 'c_dim', 0))
    c = torch.zeros([len(seeds), c_dim], device=device, dtype=z.dtype)
    ws = G.mapping(z, c, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
    if ws.ndim == 3:
        return ws[:, 0, :]
    return ws


def pairwise_l2(queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L2 distances [B, N] via one GEMM: ||q-t||^2 = ||q||^2 + ||t||^2 - 2 q t^T."""
    q2 = queries.pow(2).sum(dim=1, keepdim=True)
    t2 = targets.pow(2).sum(dim=1)
    dist2 = q2 + t2.unsqueeze(0) - 2.0 * (queries @ targets.transpose(0, 1))
    return dist2.clamp_min_(0).sqrt()


def w_plus_from_w512(
    w512: torch.Tensor,
    num_ws: int,
    *,
    w_avg: Optional[torch.Tensor] = None,
    trunc_psi: float = 1.0,
    trunc_cutoff: Optional[int] = None,
) -> torch.Tensor:
    """Broadcast [N, 512] to [N, num_ws, 512], optionally truncating toward w_avg."""
    if w512.ndim != 2:
        raise ValueError(f'expected [N, 512], got {tuple(w512.shape)}')
    w = w512.unsqueeze(1).repeat(1, int(num_ws), 1)
    if trunc_psi == 1:
        return w
    if w_avg is None:
        raise ValueError('w_avg is required when trunc_psi != 1')
    cutoff = int(num_ws if trunc_cutoff is None else trunc_cutoff)
    avg = w_avg.to(device=w.device, dtype=w.dtype).reshape(1, 1, -1)
    w = w.clone()
    w[:, :cutoff] = avg.lerp(w[:, :cutoff], trunc_psi)
    return w


def _load_source_rgb(path: Path, size: Tuple[int, int]) -> np.ndarray:
    image = PIL.Image.open(path).convert('RGB')
    if image.size != size:
        image = image.resize(size, PIL.Image.Resampling.LANCZOS)
    return np.asarray(image)


def _concat_triplet(left: np.ndarray, mid: np.ndarray, right: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(np.concatenate([left, mid, right], axis=1), 'RGB')


def _save_webp(image: PIL.Image.Image, path: Path, quality: int) -> None:
    image.save(path, format='WEBP', quality=int(quality), method=6)


def visual_validation(
    G,
    *,
    photo_names: Sequence[str],
    dict_w: torch.Tensor,
    best_seeds: Sequence[int],
    source_dir: Path,
    out_dir: Path,
    trunc_psi: float,
    trunc_cutoff: int,
    device: torch.device,
    synth_batch: int = 4,
    noise_mode: str = 'const',
    webp_quality: int = 80,
) -> int:
    """Write `{photo}_{seed}.webp` triplets: OMI original | dict W | matched seed.

    Each batch is synthesized and written before the next, so interrupting still
    leaves completed WebPs on disk.
    """
    if synth_batch < 1:
        raise ValueError('synth_batch must be >= 1')
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    force_fp32 = device.type != 'cuda'
    num_ws = int(G.num_ws)
    w_avg = G.mapping.w_avg

    rows = []
    for idx, (name, seed) in enumerate(zip(photo_names, best_seeds)):
        seed = int(seed)
        if seed < 0:
            continue
        photo_i = int(Path(name).stem)
        source_path = source_dir / name
        if not source_path.is_file():
            raise click.ClickException(f'missing source image: {source_path}')
        rows.append((idx, name, photo_i, seed, source_path))
    if not rows:
        print('visual validation: no best matches to render', flush=True)
        return 0

    print(
        f'visual validation: {len(rows)} triplets → {out_dir}  synth_batch={synth_batch}',
        flush=True,
    )
    written = 0
    with torch.inference_mode():
        for start in range(0, len(rows), synth_batch):
            chunk = rows[start : start + synth_batch]
            w512 = torch.stack([dict_w[idx] for idx, *_rest in chunk], dim=0)
            dict_plus = w_plus_from_w512(
                w512, num_ws, w_avg=w_avg, trunc_psi=trunc_psi, trunc_cutoff=trunc_cutoff,
            )
            dict_images = synthesize(
                G, dict_plus, noise_mode=noise_mode, force_fp32=force_fp32,
            )
            match_seeds = [seed for _idx, _name, _photo_i, seed, _path in chunk]
            match_w = map_w0_batch(
                G, match_seeds, trunc_psi=trunc_psi, trunc_cutoff=trunc_cutoff, device=device,
            )
            match_plus = match_w.unsqueeze(1).repeat(1, num_ws, 1)
            match_images = synthesize(
                G, match_plus, noise_mode=noise_mode, force_fp32=force_fp32,
            )
            size = None
            for (_idx, _name, photo_i, seed, source_path), dict_img, match_img in zip(
                chunk, dict_images, match_images,
            ):
                dict_uint8 = image_to_uint8(dict_img)
                match_uint8 = image_to_uint8(match_img)
                if size is None:
                    size = (dict_uint8.shape[1], dict_uint8.shape[0])
                original = _load_source_rgb(source_path, size)
                triplet = _concat_triplet(original, dict_uint8, match_uint8)
                out_path = out_dir / f'{photo_i}_{seed}.webp'
                _save_webp(triplet, out_path, webp_quality)
                written += 1
            print(
                f'visual validation wrote {written}/{len(rows)}: '
                f'{chunk[0][2]}_{chunk[0][3]}.webp … {chunk[-1][2]}_{chunk[-1][3]}.webp',
                flush=True,
            )
    print(f'Saved {written} validation WebP(s) to {out_dir}', flush=True)
    return written


def _allocated_bytes(device: torch.device) -> int:
    if device.type == 'cuda':
        return int(torch.cuda.memory_allocated(device))
    if device.type == 'mps':
        return int(torch.mps.current_allocated_memory())
    return 0


def _recommended_bytes(device: torch.device) -> int:
    if device.type == 'cuda':
        return int(torch.cuda.get_device_properties(device).total_memory)
    if device.type == 'mps' and hasattr(torch.mps, 'recommended_max_memory'):
        return int(torch.mps.recommended_max_memory())
    return 0


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    """Parse '1,2,5-10' into a list of ints (same helper as gen_images.py)."""
    if isinstance(s, list):
        return s
    ranges: List[int] = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for part in s.split(','):
        match = range_re.match(part)
        if match:
            ranges.extend(range(int(match.group(1)), int(match.group(2)) + 1))
        else:
            ranges.append(int(part))
    return ranges


def _save_jpeg(array: np.ndarray, path: Path, quality: int) -> None:
    PIL.Image.fromarray(array, 'RGB').save(path, format='JPEG', quality=quality)


@click.group()
def cli():
    """Generate visualizer faces or match psychGAN W vectors to seeds."""


@cli.command('generate')
@click.option(
    '--network',
    'network_pkl',
    default=DEFAULT_NETWORK,
    show_default=True,
    help='StyleGAN2 pickle (same path the visualizer loads for FFHQ 1024).',
)
@click.option(
    '--seeds',
    type=parse_range,
    required=True,
    help="Visualizer Latent indices, e.g. '0-31' or '0,1,2'.",
)
@click.option('--outdir', required=True, metavar='DIR', help='Folder for JPEG faces.')
@click.option('--trunc', 'trunc_psi', type=float, default=.7, show_default=True,
              help='Visualizer Truncate Psi (GUI default is 1).')
@click.option('--trunc-cutoff', 'trunc_cutoff', type=int, default=None,
              help='Visualizer cutoff. Default: G.num_ws after the first GUI frame.')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']),
              default='const', show_default=True)
@click.option('--noise-seed', type=int, default=0, show_default=True,
              help='Visualizer noise seed (0 → const noise).')
@click.option('--batch', type=int, default=1, show_default=True,
              help='Synthesis batch size. 1 is safest for 1024² on MPS.')
@click.option('--quality', type=int, default=95, show_default=True, help='JPEG quality.')
@click.option('--name-format', default='{seed:05d}.jpg', show_default=True,
              help='Filename format; `{seed}` is the UI index.')
@click.option('--device', 'device_name', default=None,
              help='cuda / mps / cpu. Default: auto (same as the visualizer).')
def generate(
    network_pkl: str,
    seeds: List[int],
    outdir: str,
    trunc_psi: float,
    trunc_cutoff: Optional[int],
    noise_mode: str,
    noise_seed: int,
    batch: int,
    quality: int,
    name_format: str,
    device_name: Optional[str],
) -> None:
    """Write JPEG faces for visualizer Latent indices 0, 1, 2, …"""
    if not seeds:
        raise click.ClickException('No seeds given.')
    if batch < 1:
        raise click.ClickException('--batch must be >= 1')

    device = get_device(device_name)
    force_fp32 = device.type != 'cuda'
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading networks from "{network_pkl}"...', flush=True)
    print(f'Using device: {device}', flush=True)
    G = load_network_dict(os.path.abspath(network_pkl), device)['G_ema']
    G.eval().requires_grad_(False)
    G = G.to(device)
    cutoff = G.num_ws if trunc_cutoff is None else trunc_cutoff
    print(
        f'z_dim={G.z_dim}  num_ws={G.num_ws}  resolution={G.img_resolution}  '
        f'trunc_psi={trunc_psi}  trunc_cutoff={cutoff}  noise_mode={noise_mode}',
        flush=True,
    )

    # Integer UI indices use a single seed with weight 1, so we can map/synthesize
    # in batches while still drawing each Z with RandomState(index).
    total = len(seeds)
    for start in range(0, total, batch):
        chunk = seeds[start : start + batch]
        ws = []
        for seed in chunk:
            w = mapping_latents(
                G,
                w0_seeds_for_ui_index(seed),
                trunc_psi=trunc_psi,
                trunc_cutoff=cutoff,
                device=device,
            )
            ws.append(w)
        w_batch = torch.cat(ws, dim=0)
        torch.manual_seed(int(noise_seed))
        try:
            images = G.synthesis(w_batch, noise_mode=noise_mode, force_fp32=force_fp32)
        except TypeError:
            images = G.synthesis(w_batch, noise_mode=noise_mode)

        for seed, img in zip(chunk, images):
            array = image_to_uint8(img)
            path = out_dir / name_format.format(seed=seed)
            _save_jpeg(array, path, quality)
        done = start + len(chunk)
        print(f'Wrote {done}/{total}: {chunk[0]} … {chunk[-1]}', flush=True)

    print(f'Saved {total} JPEG(s) to {out_dir}', flush=True)


@cli.command('match')
@click.option('--batch-size', type=int, default=100, show_default=True,
              help='Seeds mapped and compared per batch.')
@click.option('--batches', type=int, required=True,
              help='How many batches to run, starting at --start-seed.')
@click.option('--start-seed', type=int, default=0, show_default=True)
@click.option('--n-seeds', type=int, default=None,
              help='Optional total-seed override. Default: --batches times --batch-size.')
@click.option('--trunc', 'trunc_psi', type=float, default=MATCH_TRUNC_PSI, show_default=True)
@click.option('--trunc-cutoff', 'trunc_cutoff', type=int, default=None)
@click.option('--calibrate-seed', type=int, default=CALIBRATE_SEED, show_default=True,
              help='Known visual match used to set the distance cutoff.')
@click.option('--cutoff-mult', type=float, default=CUTOFF_MULTIPLIER, show_default=True,
              help='Match cutoff = this times the calibrate-seed nearest-neighbour distance.')
@click.option('--network', 'network_pkl', default=DEFAULT_NETWORK, show_default=True)
@click.option('--data-path', default=DEFAULT_DATA_PATH, show_default=True,
              help='Directory with photo_to_coords.pkl (load_psychGAN_data).')
@click.option('--device', 'device_name', default=None)
@click.option('--out', 'json_out', default=None, type=click.Path(),
              help='Optional JSON path for match results.')
@click.option('--validate-dir', default=DEFAULT_VALIDATE_DIR, show_default=True,
              type=click.Path(), help='Folder for validation WebPs written after matching.')
@click.option('--source-images', default=str(DEFAULT_SOURCE_IMAGES), show_default=True,
              type=click.Path(), help='OMI originals named like 1.jpg.')
@click.option('--validate-batch', type=int, default=4, show_default=True,
              help='Synthesis batch size for the end-of-run validation faces (1024²).')
@click.option('--webp-quality', type=int, default=80, show_default=True)
def match(
    batch_size: int,
    batches: int,
    start_seed: int,
    n_seeds: Optional[int],
    trunc_psi: float,
    trunc_cutoff: Optional[int],
    calibrate_seed: int,
    cutoff_mult: float,
    network_pkl: str,
    data_path: str,
    device_name: Optional[str],
    json_out: Optional[str],
    validate_dir: str,
    source_images: str,
    validate_batch: int,
    webp_quality: int,
) -> None:
    """Map seeds at trunc 0.7 and match them to stacked psychGAN W vectors."""
    if batch_size < 1 or batches < 1:
        raise click.ClickException('--batch-size and --batches must be >= 1')
    if n_seeds is not None and n_seeds < 1:
        raise click.ClickException('--n-seeds must be >= 1')
    if cutoff_mult <= 0:
        raise click.ClickException('--cutoff-mult must be > 0')

    n_eval = batches * batch_size if n_seeds is None else n_seeds
    n_batches = (n_eval + batch_size - 1) // batch_size
    last_seed = start_seed + n_eval - 1
    device = get_device(device_name)

    from utils import load_psychGAN_data

    print(f'Using device: {device}', flush=True)
    print(f'Loading psychGAN data from {data_path}', flush=True)
    photo_to_coords, _ratings = load_psychGAN_data(data_path)
    targets, photo_names = stack_dataset_w(photo_to_coords, device)
    n_photos = targets.shape[0]
    print(f'Stacked dataset W: {tuple(targets.shape)}', flush=True)

    print(f'Loading mapping network from "{network_pkl}"...', flush=True)
    G = load_network_dict(os.path.abspath(network_pkl), device)['G_ema']
    G.eval().requires_grad_(False)
    G = G.to(device)
    cutoff = int(G.num_ws if trunc_cutoff is None else trunc_cutoff)
    print(
        f'z_dim={G.z_dim}  num_ws={G.num_ws}  trunc_psi={trunc_psi}  '
        f'trunc_cutoff={cutoff}  seeds={start_seed}…{last_seed} ({n_eval})  '
        f'batches={n_batches} × {batch_size}',
        flush=True,
    )

    with torch.inference_mode():
        calib_w = map_w0_batch(
            G, [calibrate_seed], trunc_psi=trunc_psi, trunc_cutoff=cutoff, device=device,
        )
        calib_dist = pairwise_l2(calib_w, targets)[0]
        nn_dist, nn_idx = calib_dist.min(dim=0)
        nn_dist = float(nn_dist)
        nn_idx = int(nn_idx)
        topk = min(5, n_photos)
        top_d, top_i = torch.topk(calib_dist, k=topk, largest=False)

    match_cutoff = cutoff_mult * nn_dist
    calib_photo = f'{calibrate_seed}.jpg'
    calib_self = None
    if calib_photo in photo_names:
        calib_self = float(calib_dist[photo_names.index(calib_photo)])
    print(
        f'Calibration seed {calibrate_seed} nearest {photo_names[nn_idx]}  '
        f'L2={nn_dist:.6f}  cutoff={cutoff_mult:g}× → {match_cutoff:.6f}',
        flush=True,
    )
    print(
        f'  seed-{calibrate_seed} vs dataset: mean={float(calib_dist.mean()):.4f}  '
        f'p5={float(calib_dist.kthvalue(max(n_photos // 20, 1)).values):.4f}  '
        f'max={float(calib_dist.max()):.4f}'
        + (f'  vs {calib_photo} L2={calib_self:.6f}' if calib_self is not None else ''),
        flush=True,
    )
    print('  top-5:', flush=True)
    for dist, idx in zip(top_d.tolist(), top_i.tolist()):
        print(f'    {photo_names[int(idx)]:12s}  L2={float(dist):.6f}', flush=True)

    best_dist = torch.full((n_photos,), float('inf'), device=device)
    best_seed = torch.full((n_photos,), -1, device=device, dtype=torch.long)
    bytes_before = _allocated_bytes(device)
    first_ms = None
    first_batch_bytes = None

    with torch.inference_mode():
        for batch_i in range(n_batches):
            origin = start_seed + batch_i * batch_size
            chunk_end = min(origin + batch_size, start_seed + n_eval)
            chunk = list(range(origin, chunk_end))
            _synchronize(device)
            t0 = time.perf_counter()
            mapped = map_w0_batch(
                G, chunk, trunc_psi=trunc_psi, trunc_cutoff=cutoff, device=device,
            )
            dists = pairwise_l2(mapped, targets)
            batch_best, batch_arg = dists.min(dim=0)
            improved = batch_best < best_dist
            best_dist = torch.where(improved, batch_best, best_dist)
            seed_tensor = torch.tensor(chunk, device=device, dtype=torch.long)
            best_seed = torch.where(improved, seed_tensor[batch_arg], best_seed)
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            if batch_i == 0:
                first_ms = elapsed_ms
                first_batch_bytes = max(_allocated_bytes(device) - bytes_before, 0)
            print(
                f'batch {batch_i + 1}/{n_batches}  seeds {chunk[0]}–{chunk[-1]}  '
                f'{elapsed_ms:.1f} ms  ({len(chunk) / (elapsed_ms / 1e3):.0f} seeds/s)',
                flush=True,
            )

    rec = _recommended_bytes(device)
    used = _allocated_bytes(device)
    per_seed = (first_batch_bytes or 0) / max(batch_size, 1)
    if rec > 0 and per_seed > 0:
        # Keep ~50% of recommended memory as headroom after current allocation.
        headroom = max(rec * 0.5 - used, 0.0)
        est_max = max(batch_size, int(headroom / per_seed))
    elif per_seed > 0:
        est_max = max(batch_size, int((2 * 1024**3) / per_seed))
    else:
        est_max = batch_size * 50
    est_max = max(batch_size, min(est_max, 50_000))

    print(
        f'Memory: allocated={used / 1024**2:.1f} MiB  '
        f'first-batch delta={ (first_batch_bytes or 0) / 1024**2:.2f} MiB  '
        f'recommended={rec / 1024**2:.0f} MiB  '
        f'est. max batch≈{est_max}  first-batch {first_ms:.1f} ms',
        flush=True,
    )

    best_dist_cpu = best_dist.detach().cpu()
    best_seed_cpu = best_seed.detach().cpu()
    matched = []
    unmatched = []
    seed_to_photos = {}
    for i, name in enumerate(photo_names):
        dist = float(best_dist_cpu[i])
        seed = int(best_seed_cpu[i])
        row = {'photo': name, 'seed': seed, 'l2': dist}
        if seed >= 0 and dist <= match_cutoff:
            matched.append(row)
            seed_to_photos.setdefault(seed, []).append(name)
        else:
            unmatched.append(row)
    collisions = {seed: photos for seed, photos in seed_to_photos.items() if len(photos) > 1}

    best_vals = best_dist_cpu[torch.isfinite(best_dist_cpu)]
    same_index = 0
    index_minus_one = 0
    for i, name in enumerate(photo_names):
        seed = int(best_seed_cpu[i])
        photo_i = int(name.split('.', 1)[0])
        if seed == photo_i:
            same_index += 1
        elif seed == photo_i - 1:
            index_minus_one += 1
    print(
        f'best-dist over photos: min={float(best_vals.min()):.4f}  '
        f'median={float(best_vals.median()):.4f}  max={float(best_vals.max()):.4f}',
        flush=True,
    )
    print(
        f'index pairing: i.jpg→seed i: {same_index}  i.jpg→seed i-1: {index_minus_one}',
        flush=True,
    )
    print(
        f'Matches: {len(matched)}/{n_photos} under cutoff {match_cutoff:.6f}  '
        f'unmatched={len(unmatched)}  colliding_seeds={len(collisions)}',
        flush=True,
    )
    show = matched[:25]
    if show:
        print('  first matches:', flush=True)
        for row in show:
            print(f"    {row['photo']:12s}  seed={row['seed']:<6d}  L2={row['l2']:.6f}", flush=True)
        if len(matched) > len(show):
            print(f'    … {len(matched) - len(show)} more', flush=True)

    payload = {
        'trunc_psi': trunc_psi,
        'trunc_cutoff': cutoff,
        'calibrate_seed': calibrate_seed,
        'calibrate_photo': photo_names[nn_idx],
        'calibrate_l2': nn_dist,
        'calibrate_self_l2': calib_self,
        'calibrate_mean_l2': float(calib_dist.mean()),
        'cutoff_mult': cutoff_mult,
        'cutoff': match_cutoff,
        'seeds': {'start': start_seed, 'end': last_seed, 'count': n_eval},
        'batch_size': batch_size,
        'batches_run': n_batches,
        'est_max_batch': est_max,
        'first_batch_ms': first_ms,
        'matched': matched,
        'unmatched_count': len(unmatched),
        'collisions': collisions,
        'index_pairing': {'same': same_index, 'minus_one': index_minus_one},
        'best_dist': {
            'min': float(best_vals.min()),
            'median': float(best_vals.median()),
            'max': float(best_vals.max()),
        },
    }
    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f'Wrote {out_path}', flush=True)

    if validate_batch < 1:
        raise click.ClickException('--validate-batch must be >= 1')
    print('Matching done. Running visual validation.', flush=True)
    visual_validation(
        G,
        photo_names=photo_names,
        dict_w=targets,
        best_seeds=[int(seed) for seed in best_seed_cpu.tolist()],
        source_dir=Path(source_images),
        out_dir=Path(validate_dir),
        trunc_psi=trunc_psi,
        trunc_cutoff=cutoff,
        device=device,
        synth_batch=validate_batch,
        webp_quality=webp_quality,
    )


if __name__ == '__main__':
    cli()

