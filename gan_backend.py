"""
PyTorch port of the original (TF/dnnlib) helpers for StyleGAN.
- Removes TensorFlow/tflib/dnnlib/pretrained_networks dependencies.
- Uses a PyTorch StyleGAN2-ADA generator loaded from a .pkl (expects 'G_ema').
- Keeps function/class names as close as possible to the originals for drop-in use.

Notes
-----
* For best compatibility, the Build_model class exposes a minimal wrapper that emulates
  the old `Gs.components.mapping.run`, `Gs.components.synthesis.run`, and `Gs.get_var('dlatent_avg')` APIs.
* `randomize_noise=False` from TF is equivalent to `noise_mode='const'` here.
* Images returned from generation methods are NumPy arrays in NHWC uint8, matching the TF helpers.
* If you already have a loader (e.g. `setup_stylegan`) you can plug it in by passing a path
  to the same pickled generator or modifying `load_generator` accordingly.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple, Union
import sys
# Ensure local StyleGAN3 utilities (torch_utils, dnnlib, legacy, etc.) are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STYLEGAN3_DIR = os.path.join(PROJECT_ROOT, "content", "psychGAN", "stylegan3")
if STYLEGAN3_DIR not in sys.path:
    sys.path.append(STYLEGAN3_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "stylegan3"))
import PIL.Image
from PIL import Image

import torch
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------

def _get_device(pref: Optional[str] = None) -> torch.device:
    if pref is not None:
        try:
            return torch.device(pref)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_nhwc_uint8(img_t: torch.Tensor) -> np.ndarray:
    """Convert synthesized images from NCHW float32 in [-1,1] to NHWC uint8 [0,255]."""
    if img_t.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(img_t.shape)}")
    img = (img_t.clamp(-1, 1) + 1) * 0.5  # [0,1]
    img = (img * 255).round().clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1).contiguous()  # NHWC
    return img.cpu().numpy()


def _ensure_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    raise TypeError(f"Unsupported type: {type(x)}")


# -----------------------------
# Face alignment (unchanged, TF-free)
# -----------------------------

def Align_face_image(src_file: str, output_size: int = 1024, transform_size: int = 4096,
                     enable_padding: bool = True) -> None:
    """Align an in-the-wild face image using dlib 68-landmark predictor.
    Saves back to `src_file` (in-place), identical behavior to original helper.
    """
    import dlib
    import scipy.ndimage  # type: ignore

    print('aligning image...')
    img_ = dlib.load_rgb_image(src_file)
    print("Image Shape :", img_.shape)

    frontal_face = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    shape_ = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = frontal_face(img_, 1)
    for i, d in enumerate(dets):
        print(f"Detection {i}: Left: {d.rect.left()} Top: {d.rect.top()} Right: {d.rect.right()} Bottom: {d.rect.bottom()} Confidence: {d.confidence}")
        shape = shape_(img_, d.rect)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0).x, shape.part(1)))

        # Landmarks
        lm_eye_left  = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)])
        lm_eye_right = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)])
        lm_mouth_outer = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 60)])

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = Image.open(src_file)

        # Shrink
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img_np = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img_np.shape
            yv, xv, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(xv) / pad[0], np.float32(w - 1 - xv) / pad[2]),
                              1.0 - np.minimum(np.float32(yv) / pad[1], np.float32(h - 1 - yv) / pad[3]))
            blur = qsize * 0.02
            img_np += (scipy.ndimage.gaussian_filter(img_np, [blur, blur, 0]) - img_np) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img_np += (np.median(img_np, axis=(0, 1)) - img_np) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img_np), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), Image.LANCZOS)
        img.save(src_file)


# -----------------------------
# Style loss helpers (PyTorch)
# -----------------------------

def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for a tensor.

    Accepts CHW or NCHW. Returns CxC Gram for each batch (N,C,C) or single (C,C).
    """
    if input_tensor.ndim == 3:  # CHW or HWC
        if input_tensor.shape[0] in (1, 3):  # assume CHW
            c, h, w = input_tensor.shape
            x = input_tensor.view(c, h * w)
        else:  # HWC
            h, w, c = input_tensor.shape
            x = input_tensor.permute(2, 0, 1).contiguous().view(c, h * w)
        gram = x @ x.t() / (h * w)
        return gram
    elif input_tensor.ndim == 4:  # NCHW or NHWC
        if input_tensor.shape[1] <= 4:  # likely NCHW
            n, c, h, w = input_tensor.shape
            x = input_tensor.view(n, c, h * w)
        else:  # NHWC
            n, h, w, c = input_tensor.shape
            x = input_tensor.permute(0, 3, 1, 2).contiguous().view(n, c, h * w)
        gram = torch.matmul(x, x.transpose(1, 2)) / (h * w)
        return gram
    else:
        raise ValueError("Expected 3D or 4D tensor for gram_matrix")


def get_style_loss(base_style: torch.Tensor, gram_target: torch.Tensor) -> torch.Tensor:
    """Classic style loss between Gram(base_style) and precomputed gram_target.

    Inputs can be CHW/NCHW or HWC/NHWC; function handles layout.
    """
    g = gram_matrix(base_style)
    # Broadcast gram_target if needed
    if gram_target.ndim == 2 and g.ndim == 3:
        gram_target = gram_target.unsqueeze(0).expand_as(g)
    return F.mse_loss(g, gram_target)


# -----------------------------
# Generator loading & image synthesis (PyTorch)
# -----------------------------

def load_generator(network_pkl: Optional[str] = None, device: Optional[str] = None):
    """Load a PyTorch StyleGAN2-ADA generator (expects pickle with a dict containing 'G_ema').

    If `network_pkl` is None or missing, raises FileNotFoundError.
    """
    dev = _get_device(device)
    if network_pkl is None or not os.path.exists(network_pkl):
        raise FileNotFoundError(
            f"Couldn't find network pickle at {network_pkl!r}. Provide a local .pkl with 'G_ema'.")
    with open(network_pkl, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'G_ema' in obj:
        G = obj['G_ema'].to(dev)
    else:
        # Some pickles store the generator directly
        try:
            G = obj.to(dev)
        except Exception as e:
            raise RuntimeError("Unsupported pickle format; expected dict with 'G_ema' or a torch.nn.Module") from e
    G.eval()
    return G, dev


def _mapping(G, z: torch.Tensor, truncation_psi: float = 1.0) -> torch.Tensor:
    # G.mapping returns [N, num_ws, w_dim]
    return G.mapping(z, None, truncation_psi=truncation_psi)


def _synthesis(G, w: torch.Tensor, noise_mode: str = 'const') -> torch.Tensor:
    # Expects w of shape [N, num_ws, w_dim]
    return G.synthesis(w, noise_mode=noise_mode)


def _shape_num_ws(G) -> int:
    # best-effort to discover num_ws
    try:
        return G.synthesis.num_ws  # stylegan2-ada
    except Exception:
        return 18  # sensible default for 1024x1024 ffhq


# -----------------------------
# Standalone functions mirroring the TF helpers
# -----------------------------

def generate_im_official(network_pkl: str, seeds: Iterable[int] = (22,), truncation_psi: float = 0.5,
                         noise_mode: str = 'const') -> List[np.ndarray]:
    """Generate images for a list of seeds and save as seedXXXX.png in CWD.

    Returns a list of images as NHWC uint8 numpy arrays.
    """
    print(f'Loading generator from "{network_pkl}"...')
    G, device = load_generator(network_pkl)
    z_dim = getattr(G.mapping, 'z_dim', 512)
    num_ws = _shape_num_ws(G)

    results = []
    for idx, seed in enumerate(seeds):
        print(f'Generating image for seed {seed} ({idx+1}/{len(list(seeds))}) ...')
        gen = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn([1, z_dim], generator=gen, device=device)
        w = _mapping(G, z, truncation_psi=truncation_psi)  # [1, num_ws, 512]
        img_t = _synthesis(G, w, noise_mode=noise_mode)      # [1, 3, H, W] in [-1,1]
        img = _to_nhwc_uint8(img_t)[0]
        Image.fromarray(img, 'RGB').save(f'seed{seed:04d}.png')
        results.append(img)
    return results


# -----------------------------
# Class keeping API-compat with the original Build_model
# -----------------------------
class Build_model:
    def __init__(self, opt: Optional[SimpleNamespace] = None, device: Optional[str] = None):
        """`opt` may include `network_pkl`.

        If `/usr/app/stylegan/stylegan2-ffhq-config-f.pkl` exists, it will be used by default.
        """
        self.opt = opt or SimpleNamespace(network_pkl=None)
        network_pkl = self.opt.network_pkl
        if not network_pkl:
            raise FileNotFoundError("Please provide a local StyleGAN2-ADA PyTorch pickle via opt.network_pkl")

        print(f'Loading generator from "{network_pkl}"...')
        self.G, self.device = load_generator(network_pkl, device)
        # compile the generator
        self.noise_mode = 'const'  # mirrors randomize_noise=False
        self.z_dim = getattr(self.G.mapping, 'z_dim', 512)
        self.num_ws = _shape_num_ws(self.G)

        # Minimal wrappers to emulate TF API used downstream
        class _MappingWrapper:
            def __init__(self, parent: 'Build_model'):
                self.parent = parent
            def run(self, z: Union[np.ndarray, torch.Tensor], c=None, truncation_psi: float = 1.0):
                z_t = _ensure_tensor(z, self.parent.device)
                w = _mapping(self.parent.G, z_t, truncation_psi=truncation_psi)  # [N, num_ws, wdim]
                return w.detach().cpu().numpy()

        class _SynthesisWrapper:
            def __init__(self, parent: 'Build_model'):
                self.parent = parent
            def run(self, w: Union[np.ndarray, torch.Tensor]):
                w_t = _ensure_tensor(w, self.parent.device)
                if w_t.ndim == 2:  # [N, 512] -> broadcast to [N, num_ws, 512]
                    w_t = w_t.unsqueeze(1).repeat(1, self.parent.num_ws, 1)
                img_t = _synthesis(self.parent.G, w_t, noise_mode=self.parent.noise_mode)
                return _to_nhwc_uint8(img_t)

        class _Components:
            def __init__(self, parent: 'Build_model'):
                self.mapping = _MappingWrapper(parent)
                self.synthesis = _SynthesisWrapper(parent)

        class _GsShim:
            def __init__(self, parent: 'Build_model'):
                self.parent = parent
                self.components = _Components(parent)
                # Backwards-compat convenience
                self.input_shape = (None, parent.z_dim)
            def get_var(self, name: str):
                if name in ('dlatent_avg', 'w_avg'):
                    return self.parent.G.mapping.w_avg.detach().cpu().numpy()
                raise KeyError(name)

        # Expose a shim to reduce downstream changes
        self.Gs = _GsShim(self)

    # --- Generation helpers (numpy in, numpy out) ---
    def generate_im_from_random_seed(self, seed: int = 22, truncation_psi: float = 0.5):
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        z = torch.randn([1, self.z_dim], generator=gen, device=self.device)
        w = _mapping(self.G, z, truncation_psi=truncation_psi)
        img_t = _synthesis(self.G, w, noise_mode=self.noise_mode)
        return _to_nhwc_uint8(img_t)

    def generate_im_from_z_space(self, z: Union[np.ndarray, torch.Tensor], truncation_psi: float = 0.5):
        z_t = _ensure_tensor(z, self.device)
        w = _mapping(self.G, z_t, truncation_psi=truncation_psi)
        img_t = _synthesis(self.G, w, noise_mode=self.noise_mode)
        return _to_nhwc_uint8(img_t)

    def generate_im_from_w_space(self, w: Union[np.ndarray, torch.Tensor], resolution: int = 1024):
        w_t = _ensure_tensor(w, self.device)
        if w_t.ndim == 2:  # [N, 512] -> [N, num_ws, 512]
            w_t = w_t.unsqueeze(1).repeat(1, self.num_ws, 1)
        img_t = _synthesis(self.G, w_t, noise_mode=self.noise_mode)
        if resolution != 1024:
            # select stride pixels from img_t
            img_t = img_t[:, :, ::1024//resolution, ::1024//resolution].contiguous()
        return _to_nhwc_uint8(img_t)


# -----------------------------
# (Optional) Quick smoke-test when running as a script
# -----------------------------
if __name__ == "__main__":
    # Example usage:
    #   python stylegan_pytorch_port.py /path/to/stylegan2-ffhq-1024x1024.pkl 10
    import sys
    pkl = sys.argv[1] if len(sys.argv) > 1 else None
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if not pkl:
        raise SystemExit("Please pass a local StyleGAN2-ADA PyTorch pickle path as the first argument.")
    bm = Build_model(SimpleNamespace(network_pkl=pkl))
    img = bm.generate_im_from_random_seed(seed=seed, truncation_psi=0.5)[0]
    Image.fromarray(img).save(f"seed{seed:04d}.png")
    print("Wrote:", f"seed{seed:04d}.png")
