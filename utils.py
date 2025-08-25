from sklearn.linear_model import Ridge, LinearRegression
import torch
import pickle
import numpy as np
import sys
import os
from typing import Optional
import torch.nn as nn
from stylgan_distilled import *
# Ensure local StyleGAN3 utilities (torch_utils, dnnlib, legacy, etc.) are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STYLEGAN3_DIR = os.path.join(PROJECT_ROOT, "content", "psychGAN", "stylegan3")
if STYLEGAN3_DIR not in sys.path:
    sys.path.append(STYLEGAN3_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "stylegan3"))

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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


@torch.no_grad()
def load_distilled_generator(pkl_path: str, device: Optional[torch.device] = None) -> nn.Module:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    G_tapped = obj["G_ema"].eval().to(device)
    return G_tapped, device


def load_psychGAN_data(data_path):
    if not data_path.endswith("/"):
        data_path += "/"
    with open(data_path+"photo_to_coords.pkl", "rb") as f:
        photo,coords = pickle.load(f)
    photo_coords = {k: v for k, v in zip(photo, coords)}
    with open(data_path+"dim_to_photo_to_ratings.pkl", "rb") as f: 
        dim_to_photo_to_ratings = pickle.load(f)

    return photo_coords, dim_to_photo_to_ratings


########################################################


def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5

def _grid_preview(pred: torch.Tensor, target: torch.Tensor, k: int = 4) -> torch.Tensor:
    from torchvision.utils import make_grid
    k = min(k, pred.shape[0], target.shape[0])
    p = _to01(pred[:k]).cpu()
    t = _to01(target[:k]).cpu()
    return make_grid(torch.cat([p, t], dim=0), nrow=k)  # CHW

def _device_synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try: torch.mps.synchronize()
        except Exception: pass

def _reset_peak_mem(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try: torch.mps.empty_cache()
        except Exception: pass

def _get_mem_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        try: return torch.mps.current_allocated_memory()
        except Exception: return 0
    else:
        try:
            import psutil
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return 0
















########################################################



def pad_list(l, max_len):
    return l[:max_len] + [np.nan] * (max_len - len(l[:max_len]))
    
def prepare_data(config: dict, verbose=False, return_imgs=False, backend=None):
    photo_to_coords, dim_to_photo_to_ratings, device, dtype = backend.photo_to_coords, backend.dim_to_photo_to_ratings, backend.device, backend.dtype
    xs, ys = photo_to_coords, dim_to_photo_to_ratings[config['data']['attribute_dim']]
    _imgs = set(ys.keys()).intersection(set(xs.keys()))#-set('638.jpg')
    if config['data'].get('imgs', None) is not None:
        imgs = _imgs.intersection(set(config['data']['imgs']))
        if len(imgs)<len(config['data']['imgs']):
            print(f"Warning: {len(config['data']['imgs'])-len(imgs)} images not found in psychGAN dataset")
    else:
        imgs = sorted(_imgs)

    X = torch.stack([xs[k] for k in imgs]).to(device, dtype)
    max_len = int(max(len(ys[k]) for k in imgs) * config['data'].get('rating_oversampling_factor', 1.0))
    y = torch.tensor([pad_list(ys[k], max_len) for k in imgs], device=device, dtype=dtype)
    n = y.shape[1] - y.isnan().sum(dim=1)
    sample_size_weights = (n.reshape(-1)).to(device, dtype)
    sample_size_weights = sample_size_weights / sample_size_weights.mean()
    if return_imgs:
        return X, y, sample_size_weights, imgs
    else:
        return X, y, sample_size_weights

def cpu_numpy(*args):
    if len(args)==1 and isinstance(args[0], (tuple, list,set)):
        args = tuple(args[0])
    output = tuple([a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in args])
    return output if len(output)>1 else output[0]
def tensor_dd(*args):
    if len(args)==1 and isinstance(args[0], (tuple, list,set)):
        args = tuple(args[0])
    output = tuple([a.to(dtype=torch.float32,device=device) if isinstance(a, torch.Tensor) else (torch.tensor(a, dtype=torch.float32).to(device) if isinstance(a, np.ndarray) else a) for a in args])
    return output if len(output)>1 else output[0]

def get_data(dim, train=True, logit=True, backend=None):
    X, y, weights, imgs = prepare_data({'data':{'attribute_dim':dim}},return_imgs=True, backend=backend)
    train_imgs = [f"{i}.jpg" for i in range(1,1005)]+[i for i in imgs if "0_our" in i or "_flow_level_0" in i]
    X, y, weights, imgs = prepare_data({'data':{'attribute_dim':dim,"imgs":(train_imgs if train else list(set(imgs)-set(train_imgs)))}},return_imgs=True, backend=backend) #list(set(imgs)-set(train_imgs))
    
    weights[[len(i)>8 for i in imgs]] = weights[[len(i)>8 for i in imgs]]*1.5
    y = y.nanmean(dim=1)
    
    return X, (torch.logit if logit else lambda x: x)(y), weights, imgs

import numpy as np
import torch
from sklearn.linear_model import Ridge

def ridge_coefs(dim: str, alpha: float, *, fit_intercept: bool = True, backend=None) -> torch.Tensor:
    """
    Fit Ridge on the full dataset for `dim` and return the coefficient vector as a torch tensor.
    
    Trains on -y (as in your search loop), then returns -coef_ so that X @ returned_coefs (+ intercept)
    predicts the original y.

    Args:
        dim: attribute dimension name, e.g. "attractive"
        alpha: L2 penalty for Ridge
        device: torch.device to place the returned tensor on (defaults to CUDA if available, else CPU)
        fit_intercept: whether to fit an intercept in Ridge (does not affect returned coef shape)

    Returns:
        torch.Tensor of shape (n_features,) with dtype float32 on `device`
    """
    # Full set for this dimension (get_data already returns y.nanmean(dim=1))
    X, y, _, _ = get_data(dim, train=True, backend=backend)

    # To NumPy (fast path using your helpers)
    X_np, y_np = cpu_numpy(X, y)

    # Fit Ridge on -y (to match your correlation code’s convention)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X_np, y_np)

    # Coefficients for predicting original y (negate back), as float32 tensor on desired device
    coefs_np = (model.coef_).astype(np.float32, copy=False)
    coefs_t = torch.from_numpy(coefs_np).to(device)
    # print(f"Pearson correlation: {pearsonr(y_np, X_np @ coefs_np)[0]}")
    return coefs_t


if __name__ == "__main__":
    # Example:
    coefs = ridge_coefs("attractive", alpha=100)