import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Dict, List
import copy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STYLEGAN3_DIR = os.path.join(PROJECT_ROOT, "content", "psychGAN", "stylegan3")
if STYLEGAN3_DIR not in sys.path:
    sys.path.append(STYLEGAN3_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "stylegan3"))

from stylegan3.training.networks_stylegan2 import ToRGBLayer, SynthesisNetwork, Generator


# -----------------------------
# Trainable wrapper: new ToRGB at start_res
# -----------------------------
class ToRGBWrapper(nn.Module):
    """A simple wrapper around ToRGBLayer that expects (x, w_rgb) and outputs RGB in [-1,1]."""
    def __init__(self, in_channels: int, w_dim: int, out_channels: int = 3,
                 kernel_size: int = 1, conv_clamp: Optional[float] = None, channels_last: bool = False):
        super().__init__()
        self.torgb = ToRGBLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            w_dim=w_dim,
            kernel_size=kernel_size,
            conv_clamp=conv_clamp,
            channels_last=channels_last
        )

    def forward(self, x: torch.Tensor, w_rgb: torch.Tensor):
        # x: [N, C, R, R], w_rgb: [N, w_dim]
        return self.torgb(x, w_rgb)  # [N, 3, R, R] in StyleGAN's [-1,1] space

# -----------------------------
# Style extractor for the new ToRGB
# -----------------------------
def _style_for_block_torgb(cur_ws: torch.Tensor, num_conv: int) -> torch.Tensor:
    """Choose a style vector to feed the new ToRGBLayer.
       Here we take the mean of the block's conv styles as a stable choice.
       cur_ws: [N, num_conv + num_torgb (0/1), w_dim]
    """
    if num_conv <= 0:
        raise ValueError("Block has no convs to draw style from.")
    w_conv = cur_ws[:, :num_conv, :]               # [N, num_conv, w_dim]
    w_rgb  = w_conv.mean(dim=1, keepdim=False)     # [N, w_dim]
    return w_rgb
# -----------------------------
# Training loop (MPS + TensorBoard)
# -----------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- FiLM from w: produce per-channel gamma/beta ----
class StyleAffine(nn.Module):
    def __init__(self, w_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(w_dim, 2 * channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias[:channels], 1.0)  # gamma init = 1
        nn.init.constant_(self.fc.bias[channels:], 0.0)  # beta init = 0

    def forward(self, w: torch.Tensor):
        # w: [B, w_dim] -> gamma,beta: [B, C, 1, 1]
        gb = self.fc(w)  # [B, 2C]
        B, _ = gb.shape
        C = gb.shape[1] // 2
        gamma, beta = gb[:, :C], gb[:, C:]
        return gamma.view(B, C, 1, 1), beta.view(B, C, 1, 1)

# ---- Depthwise-separable conv block with FiLM ----
class StyleResBlock(nn.Module):
    def __init__(self, channels: int, w_dim: int, drop: float = 0.0):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.affine1 = StyleAffine(w_dim, channels)
        self.affine2 = StyleAffine(w_dim, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x, w):
        # Block 1
        y = self.dw1(x)
        y = self.pw1(y)
        g, b = self.affine1(w)
        y = self.act(y * g + b)
        y = self.drop(y)
        # Block 2
        y = self.dw2(y)
        y = self.pw2(y)
        g2, b2 = self.affine2(w)
        y = y * g2 + b2
        return self.act(x + y)  # residual

# ---- Windowed self-attention (lightweight, no half precision, MPS-friendly) ----
class WindowSelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, window: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.h = num_heads
        self.d = channels // num_heads
        self.window = window
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        w = self.window
        Hp = math.ceil(H / w) * w
        Wp = math.ceil(W / w) * w
        x = F.pad(x, (0, Wp - W, 0, Hp - H))  # pad right/bottom
        B, C, Hp, Wp = x.shape
        nh, nw = Hp // w, Wp // w

        # Split into windows -> [B*nh*nw, T, C]
        xw = x.view(B, C, nh, w, nw, w).permute(0, 2, 4, 3, 5, 1).reshape(B * nh * nw, w * w, C)

        qkv = self.qkv(xw)                        # [Bn, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)            # [Bn, T, C] each
        # split heads: [Bn, T, h, d] -> [Bn, h, T, d]
        def split_heads(t):
            return t.view(t.shape[0], t.shape[1], self.h, self.d).permute(0, 2, 1, 3)
        q = split_heads(q); k = split_heads(k); v = split_heads(v)

        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)  # [Bn, h, T, T]
        out = attn @ v                                                                # [Bn, h, T, d]
        out = out.permute(0, 2, 1, 3).contiguous().view(xw.shape[0], xw.shape[1], C)  # [Bn, T, C]
        out = self.proj(out)                                                          # [Bn, T, C]

        # Merge windows back to [B, C, Hp, Wp]
        out = out.view(B, nh, nw, w, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        return out[:, :, :H, :W]  # crop
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChromaticAberrationReducer(nn.Module):
    """
    Edge-aware chromatic aberration reducer.
    Works in YUV: suppresses high-frequency color fringing in U/V guided by luminance edges.
    """
    def __init__(self, strength: float = 1.0, blend: float = 1.0):
        super().__init__()
        self.strength = strength
        self.blend = blend  # final residual blend into RGB

        # depthwise Gaussian blur for U/V (init as gentle 3x3)
        self.blur_uv = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        with torch.no_grad():
            g = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32) / 16.0
            k = g.view(1, 1, 3, 3).repeat(2, 1, 1, 1)  # groups=2
            self.blur_uv.weight.copy_(k)

        # 1x1 gate from [U, V, edge, Y] -> [gate_U, gate_V] in [0,1]
        self.gate = nn.Conv2d(4, 2, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)  # start as identity (no correction)

        # fixed Sobel for edge(Y)
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", ky.view(1, 1, 3, 3))

    @staticmethod
    def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B,3,H,W] in [-1,1]
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        U = -0.14713 * r - 0.28886 * g + 0.436 * b
        V = 0.615 * r - 0.51499 * g - 0.10001 * b
        return torch.cat([Y, U, V], dim=1)

    @staticmethod
    def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
        Y, U, V = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U
        return torch.cat([R, G, B], dim=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb in [-1,1], shape [B,3,H,W]
        B, C, H, W = rgb.shape
        yuv = self.rgb_to_yuv(rgb)
        Y, U, V = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]

        # edge map from luminance
        gx = F.conv2d(Y, self.sobel_x, padding=1)
        gy = F.conv2d(Y, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)

        # predict per-pixel gates in [0,1] for U and V
        gate_in = torch.cat([U, V, edge, Y], dim=1)           # [B,4,H,W]
        gate_uv = torch.sigmoid(self.gate(gate_in)) * self.strength  # [B,2,H,W]

        # attenuate high-frequency in U/V guided by gate
        UV = torch.cat([U, V], dim=1)                         # [B,2,H,W]
        UV_blur = self.blur_uv(UV)
        UV_hp = UV - UV_blur
        UV_corr = UV - gate_uv * UV_hp

        yuv_corr = torch.cat([Y, UV_corr[:, 0:1], UV_corr[:, 1:2]], dim=1)
        rgb_corr = self.yuv_to_rgb(yuv_corr)

        # residual blend and clamp to [-1,1]
        out = torch.tanh(rgb + self.blend * (rgb_corr - rgb))
        return out
# ---- Super-res ToRGB head: 64x64 -> 128x128 ----
class SRToRGBHead(nn.Module):
    """
    Lightweight head that consumes 64x64 feature maps (C channels) + style w,
    adds local self-attention, upsamples to 128x128, and outputs RGB via ToRGBLayer.
    Optionally adds a 64->RGB skip path (upsampled) for sharper color guidance.
    """
    def __init__(self, in_ch: int, w_dim: int, mid_ch: int = 256,
                 num_style_blocks: int = 2, num_heads: int = 4, window: int = 8,
                 use_skip64: bool = True, ToRGBLayer_cls=None,
                 use_ca: bool = True, ca_strength: float = 1.0, ca_blend: float = 1.0):
        super().__init__()
        assert ToRGBLayer_cls is not None, "Pass your StyleGAN ToRGBLayer class via ToRGBLayer_cls"
        self.stem = nn.Conv2d(in_ch, mid_ch, 1)
        self.blocks = nn.ModuleList([StyleResBlock(mid_ch, w_dim) for _ in range(num_style_blocks)])
        self.attn = WindowSelfAttention2d(mid_ch, num_heads=num_heads, window=window)
        # upsample x2 with pixelshuffle
        self.ups_conv = nn.Conv2d(mid_ch, mid_ch * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        # ToRGB at 128
        self.torgb128 = ToRGBLayer_cls(in_channels=mid_ch, out_channels=3, w_dim=w_dim)
        # Optional 64->RGB skip
        self.use_skip64 = use_skip64
        if use_skip64:
            self.torgb64 = ToRGBLayer_cls(in_channels=in_ch, out_channels=3, w_dim=w_dim)

        self.use_ca = use_ca
        if use_ca:
            self.ca = ChromaticAberrationReducer(strength=ca_strength, blend=ca_blend)

    def forward(self, x64: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        x64: [B, C_in, 64, 64], w: [B, w_dim]
        returns: RGB128 in [-1,1], [B, 3, 128, 128]
        """
        y = self.stem(x64)
        for blk in self.blocks:
            y = blk(y, w)
        y = self.attn(y)                           # [B, mid_ch, 64, 64]
        y = self.pixel_shuffle(self.ups_conv(y))   # [B, mid_ch, 128, 128]
        rgb128 = self.torgb128(y, w)               # [-1,1]

        if self.use_skip64:
            rgb64 = self.torgb64(x64, w)           # [-1,1], [B,3,64,64]
            rgb64_up = F.interpolate(rgb64, scale_factor=2, mode="bilinear", align_corners=False)
            rgb128 = torch.tanh(rgb128 + 0.5 * rgb64_up)  # small residual blend

        # --- chromatic aberration correction ---
        if self.use_ca:
            rgb128 = self.ca(rgb128)

        return rgb128



# ---------- tiny utility: probe feature shape at tap (start_res) ----------
@torch.no_grad()
def _probe_feature_shape(G: nn.Module,
                         start_res: int,
                         truncation_psi: float = 1.0,
                         device: Optional[torch.device] = None) -> Tuple[int, int]:
    """Return (C, R) at the chosen synthesis tap (start_res)."""
    if device is None:
        device = next(G.parameters()).device
    z_dim = getattr(G, "z_dim", 512)
    z = torch.randn(2, z_dim, device=device)
    ws = G.mapping(z, None, truncation_psi=truncation_psi)
    x, _, _, _ = run_until_resolution(G, ws, start_res, noise_mode="const")
    C, R = int(x.shape[1]), int(x.shape[2])
    return C, R

# ---------- head builders ----------
def build_torgb_head(G: nn.Module,
                     start_res: int,
                     *,
                     use_sr_head: bool = False,
                     ToRGBLayer_cls: Any = ToRGBLayer,
                     sr_mid_ch: int = 256,
                     sr_num_style_blocks: int = 2,
                     sr_num_heads: int = 4,
                     sr_window: int = 8,
                     sr_use_skip64: bool = True,
                     truncation_psi: float = 1.0,
                     device: Optional[torch.device] = None) -> nn.Module:
    """
    Construct a head compatible with G at `start_res`, without loading weights.
    - use_sr_head=False -> ToRGBWrapper at R x R
    - use_sr_head=True  -> SRToRGBHead (R x R -> 2R x 2R)
    """
    if device is None:
        device = next(G.parameters()).device
    C, _R = _probe_feature_shape(G, start_res, truncation_psi, device)
    if use_sr_head:
        head = SRToRGBHead(
            in_ch=C, w_dim=G.w_dim,
            mid_ch=sr_mid_ch,
            num_style_blocks=sr_num_style_blocks,
            num_heads=sr_num_heads,
            window=sr_window,
            use_skip64=sr_use_skip64,
            ToRGBLayer_cls=ToRGBLayer_cls,
        ).to(device)
    else:
        head = ToRGBWrapper(in_channels=C, w_dim=G.w_dim, out_channels=G.img_channels).to(device)
    return head



# ========= target size helper (uses your probe) =========
@torch.no_grad()
def _target_size(G: torch.nn.Module, start_res: int, use_sr_head: bool, truncation_psi: float, device: torch.device):
    from typing import Tuple
    C, R = _probe_feature_shape(G, start_res, truncation_psi, device)  # from your loader module
    return (2*R, 2*R) if use_sr_head else (R, R)


# Assumes these are available from your StyleGAN codebase:
# - ToRGBLayer (you provided)
# - bias_act, modulated_conv2d, etc. imported by ToRGBLayer
# - Generator / SynthesisNetwork definitions


# ---------- checkpoint helpers ----------
def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Accepts several common checkpoint layouts and returns a state_dict."""
    if "head" in ckpt and isinstance(ckpt["head"], dict):
        return ckpt["head"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # Some tools save the raw state dict directly
    # Heuristic: a mapping of str -> Tensor
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # looks like a plain state_dict
    raise ValueError("Unsupported checkpoint format: cannot find model weights.")




# ---------- main loader ----------
def load_torgb_head(G: nn.Module,
                    start_res: int,
                    *,
                    checkpoint: Optional[str] = None,
                    use_sr_head: bool = False,
                    ToRGBLayer_cls: Any = ToRGBLayer,
                    sr_mid_ch: int = 256,
                    sr_num_style_blocks: int = 2,
                    sr_num_heads: int = 4,
                    sr_window: int = 8,
                    sr_use_skip64: bool = True,
                    truncation_psi: float = 1.0,
                    device: Optional[torch.device] = None,
                    eval_mode: bool = True,
                    compile_mode: Optional[str] = None,   # e.g. "reduce-overhead" | "max-autotune"
                    strict: bool = False) -> nn.Module:
    """
    Minimal, app-friendly loader:
      - Builds the correct head for G at `start_res`.
      - (Optional) loads weights from `checkpoint`.
      - (Optional) sets memory format and compiles the head.
      - Returns a ready-to-use nn.Module.
    """
    if device is None:
        device = next(G.parameters()).device

    # 1) Build
    head = build_torgb_head(
        G, start_res,
        use_sr_head=use_sr_head,
        ToRGBLayer_cls=ToRGBLayer_cls,
        sr_mid_ch=sr_mid_ch,
        sr_num_style_blocks=sr_num_style_blocks,
        sr_num_heads=sr_num_heads,
        sr_window=sr_window,
        sr_use_skip64=sr_use_skip64,
        truncation_psi=truncation_psi,
        device=device,
    )

    # 2) Load weights (if provided)
    if checkpoint is not None:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        state = _extract_state_dict(ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = head.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            print(f"[load_torgb_head] non-strict load: missing={missing}, unexpected={unexpected}")

    # 3) Optional memory format & compile
    if compile_mode:
        try:
            torch_compile = getattr(torch, "compile", None)
            if torch_compile is not None:
                head = torch_compile(head, mode=compile_mode, fullgraph=False)
            else:
                print("[load_torgb_head] torch.compile not available; skipping compilation.")
        except Exception as e:
            print(f"[load_torgb_head] compile failed ({e}); continuing without compilation.")

    if eval_mode:
        head.eval()

    return head



# -----------------------------
# Utilities to walk Synthesis
# -----------------------------
@torch.no_grad()
def run_until_resolution(G, ws: torch.Tensor, stop_res: int, noise_mode: str = "const", **block_kwargs):
    """Run synthesis up to and including `stop_res`, return:
       x_stop, img_stop, last_block_cur_ws, next_w_idx
    """
    S = G.synthesis
    x = img = None
    w_idx = 0
    for res in S.block_resolutions:
        block = getattr(S, f"b{res}")
        cur_len = block.num_conv + block.num_torgb
        cur_ws  = ws.narrow(1, w_idx, cur_len)
        x, img  = block(x, img, cur_ws, noise_mode=noise_mode, **block_kwargs)
        w_idx  += block.num_conv
        if res == stop_res:
            return x, img, cur_ws, w_idx  # w_idx now points to NEXT block's conv styles
    raise ValueError(f"stop_res {stop_res} not found in block_resolutions {S.block_resolutions}")

@torch.no_grad()
def resume_from_next_block(G, x, img, ws: torch.Tensor, next_w_idx: int, start_res: int,
                           noise_mode: str = "const", **block_kwargs):
    """Continue synthesis from the block AFTER `start_res` to the end (frozen)."""
    S = G.synthesis
    begin = False
    w_idx = next_w_idx
    for res in S.block_resolutions:
        block = getattr(S, f"b{res}")
        cur_len = block.num_conv + block.num_torgb
        if not begin:
            if res == start_res:
                begin = True
            continue  # skip the block where we stopped (already executed)
        cur_ws = ws.narrow(1, w_idx, cur_len)
        x, img = block(x, img, cur_ws, noise_mode=noise_mode, **block_kwargs)
        w_idx += block.num_conv
    return img  # [N, 3, H, W] in [-1,1]



class TapSynthesis(nn.Module):
    """
    A minimal synthesis network that:
      - runs original StyleGAN2-ADA blocks up to `stop_res`,
      - extracts the ToRGB style for that block,
      - calls a provided head (e.g., SRToRGBHead) to produce final RGB in [-1,1].

    It keeps a compatible API: forward(ws, noise_mode='const', **block_kwargs) -> NCHW in [-1,1]
    """
    def __init__(self, original_synth: nn.Module, head: nn.Module, stop_res: int):
        super().__init__()
        self.stop_res = int(stop_res)
        self.head = head

        # Copy only needed blocks {b4, b8, ..., b{stop_res}} with their weights.
        self.block_resolutions: List[int] = []
        self.blocks = nn.ModuleDict()
        for res in getattr(original_synth, "block_resolutions"):
            self.block_resolutions.append(int(res))
            blk = getattr(original_synth, f"b{res}")
            self.blocks[f"b{res}"] = copy.deepcopy(blk)
            if int(res) == self.stop_res:
                break

        # Keep metadata for compatibility with mapping and other code
        # We expose num_ws SAME as original synthesis (safe: mapping often reads this).
        try:
            self.num_ws = int(getattr(original_synth, "num_ws"))
        except Exception:
            # conservative fallback if unknown
            self.num_ws = sum(getattr(getattr(original_synth, f"b{r}"), "num_conv", 2)
                              + getattr(getattr(original_synth, f"b{r}"), "num_torgb", 1)
                              for r in getattr(original_synth, "block_resolutions"))

        # Other helpful attrs often accessed in downstream code
        self.w_dim = int(getattr(original_synth, "w_dim", getattr(head, "w_dim", 512)))
        self.img_channels = int(getattr(original_synth, "img_channels", 3))

    def forward(self, ws: torch.Tensor, noise_mode: str = "const", **block_kwargs) -> torch.Tensor:
        """
        ws: [N, num_ws, w_dim] from the original Mapping.
        Returns NCHW RGB in [-1,1] (head output).
        """
        x = None
        img = None
        w_idx = 0
        last_cur_ws = None
        last_block = None

        for res in self.block_resolutions:
            block = self.blocks[f"b{res}"]
            # cur_ws for this block: conv + torgb
            cur_len = block.num_conv + block.num_torgb
            cur_ws = ws.narrow(1, w_idx, cur_len)          # [N, cur_len, w_dim]
            # Run the block
            x, img = block(x, img, cur_ws, noise_mode=noise_mode, **block_kwargs)
            # Track for head
            last_cur_ws = cur_ws
            last_block = block
            # Increment like your custom runner (keeps parity with your code)
            w_idx += block.num_conv
            if int(res) == self.stop_res:
                break

        assert last_cur_ws is not None and last_block is not None, "TapSynthesis ran no blocks."
        # Extract ToRGB style for the tap block
        w_rgb = _style_for_block_torgb_safe(last_cur_ws, last_block.num_conv)  # [N, w_dim]
        # Head expects features at tap resolution and style
        pred = self.head(x, w_rgb)  # [N, 3, H_out, W_out] in [-1,1]
        return pred


class TappedGenerator(nn.Module):
    """
    Container that mirrors the public API of the original G:
      - mapping: original mapping (deep-copied)
      - synthesis: TapSynthesis (blocks up to stop_res + head)
      - w_dim / z_dim / img_channels exposed for compatibility
    """
    def __init__(self, mapping: nn.Module, tap_synthesis: TapSynthesis,
                 w_dim: int, z_dim: int, img_channels: int):
        super().__init__()
        self.mapping = mapping
        self.synthesis = tap_synthesis
        self.w_dim = int(w_dim)
        self.z_dim = int(z_dim)
        self.img_channels = int(img_channels)

    def forward(self, z: torch.Tensor, truncation_psi: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Optional convenience forward: z -> ws -> synthesis
        """
        ws = self.mapping(z, None, truncation_psi=truncation_psi)
        return self.synthesis(ws, **kwargs)