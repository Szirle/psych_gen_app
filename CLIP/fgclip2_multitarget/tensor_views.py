"""Batched device-side FFHQ transforms on the native encoder input raster.

Decode/resize/patchify each original once using the existing official processor.
Subsequent views only gather tensors, unpatch, augment, and repatch on device.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

from .assets import image_inputs


def random_tensor(shape, segments, device, *, normal=False, stream=0):
    """One RNG launch per cache shard, never per image; packing-independent RNG."""
    # CPU generators support the optional MPS development path.
    rng_device = device if device.type == 'cuda' else torch.device('cpu')
    output = torch.empty((sum(count for count, _ in segments), *shape), device=rng_device)
    start = 0
    for count, seed in segments:
        generator = torch.Generator(device=rng_device).manual_seed((seed+stream) % (2**63-1))
        chunk = output[start:start+count]
        if normal:
            chunk.normal_(generator=generator)
        else:
            chunk.uniform_(generator=generator)
        start += count
    return output.to(device)


def transform(images, args, segments, enabled=None):
    """BCHW [0,1], independent per-image parameters, fused affine geometry."""
    if args.augmentation == 'none':
        return images
    batch = len(images)
    r = random_tensor((12,), segments, images.device)
    use = torch.ones(batch, device=images.device, dtype=torch.bool) if enabled is None else enabled
    flip = torch.where((r[:, 0] < .5) & use, -1., 1.)
    if args.augmentation == 'flip':
        return torch.where((flip < 0)[:, None, None, None], images.flip(-1), images)
    scale = 1+(2*r[:, 1]-1)*args.augmentation_scale*use
    theta = images.new_zeros(batch, 2, 3)
    theta[:, 0, 0], theta[:, 1, 1] = flip/scale, 1/scale
    theta[:, :, 2] = (2*r[:, 2:4]-1)*(-2*args.augmentation_translate)*use[:, None]/scale[:, None]
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    output = F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    if args.augmentation_color:
        active = ((r[:, 4] < .2) & use)[:, None, None, None]
        brightness = 1+(2*r[:, 5]-1)[:, None, None, None]*args.augmentation_color*active
        contrast = 1+(2*r[:, 6]-1)[:, None, None, None]*args.augmentation_color*active
        output = output*brightness
        mean = output.mean((1, 2, 3), keepdim=True)
        output = (output-mean)*contrast+mean
    if args.augmentation_blur:
        # One optimized convolution for the batch; independent blend per image.
        radius = math.ceil(3*args.augmentation_blur)
        blurred = gaussian_blur(output, [2*radius+1]*2, [args.augmentation_blur]*2)
        blend = (r[:, 8]*(r[:, 7] < .2)*use)[:, None, None, None]
        output = torch.lerp(output, blurred, blend)
    if args.augmentation_noise:
        strength = (r[:, 10]*(r[:, 9] < .35)*use*args.augmentation_noise)[:, None, None, None]
        noise = random_tensor(images.shape[1:], segments, images.device, normal=True, stream=101)
        output = output+noise*strength
    return torch.where(use[:, None, None, None], output.clamp(0, 1), images)


def feature_noise(features, strength, segments, enabled=None):
    if not strength:
        return features
    result = []
    for stream, x in enumerate(features[:2], 201):
        noise = random_tensor(x.shape[1:], segments, x.device, normal=True, stream=stream)
        if enabled is not None:
            noise *= enabled.reshape(-1, *([1]*(x.ndim-1)))
        result.append(F.normalize(x.float()+noise*(strength/math.sqrt(x.shape[-1])), dim=-1))
    return (*result, *features[2:])


class PreparedImages:
    """Process-wide CPU tensor bank shared by fits, with no transformed PIL path."""
    def __init__(self, assets):
        self.assets = assets
        self.items = {}

    @torch.no_grad()
    def ensure(self, rows):
        missing = [int(row) for row in np.unique(rows) if int(row) not in self.items]
        assets = self.assets
        if missing:
            print(f'Preparing {len(missing)} original image tensors once for GPU augmentation.', flush=True)
        for start in range(0, len(missing), assets.args.encode_batch):
            chunk = missing[start:start+assets.args.encode_batch]
            inputs = image_inputs(assets.engine, [assets.data.paths[row] for row in chunk], assets.args.patches)
            cpu = {key: value.detach().cpu() for key, value in inputs.items()}
            for index, row in enumerate(chunk):
                self.items[row] = {key: value[index] for key, value in cpu.items()}

    def batch(self, rows, args, segments, enabled=None):
        self.ensure(rows)
        engine = self.assets.engine
        cpu = {key: torch.stack([self.items[int(row)][key] for row in rows]) for key in self.items[int(rows[0])]}
        shapes = cpu['spatial_shapes'].numpy()
        inputs = {key: (value.pin_memory().to(engine.device, non_blocking=True)
                        if engine.device.type == 'cuda' else value.to(engine.device)) for key, value in cpu.items()}
        if args is None:
            return inputs
        pixels = inputs['pixel_values']
        patch = int(engine.image_processor.patch_size)
        mean = torch.as_tensor(engine.image_processor.image_mean, device=engine.device).reshape(1, 3, 1, 1)
        std = torch.as_tensor(engine.image_processor.image_std, device=engine.device).reshape(1, 3, 1, 1)
        # Aligned FFHQ has one native shape; do not silently distort other
        # aspect ratios to make them fit a square augmentation batch.
        unique = np.unique(shapes, axis=0)
        if len(unique) != 1:
            raise ValueError('Tensor augmentation expects one native raster shape per batch (aligned FFHQ). '
                             'Use a dataset with consistent image dimensions/aspect ratio.')
        h, w = map(int, unique[0])
        count = h*w
        images = pixels[:, :count].float().reshape(-1, h, w, patch, patch, 3)
        images = images.permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, h*patch, w*patch)
        images = transform(images*std+mean, args, segments, enabled)
        images = (images-mean)/std
        transformed = images.reshape(-1, 3, h, patch, w, patch).permute(0, 2, 4, 3, 5, 1).reshape(len(rows), count, -1)
        output = pixels.clone()
        output[:, :count] = transformed.to(pixels.dtype)
        if enabled is not None:
            output = torch.where(enabled[:, None, None], output, pixels)
        inputs['pixel_values'] = output
        return inputs
