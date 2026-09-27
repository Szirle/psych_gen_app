"""Text-only FGCLIP2 loading for phrase studies with cached image features.

Reuses the repository's tokenizer, normalization, walk modes and checkpoint
resolver. The vision tower is absent; no training setup or cache is modified.
"""
import json
import math
import torch
from ..fgclip2_core import FGCLIP2


class FrozenTextEncoder(FGCLIP2):
    def _load_streamed(self, common, reserve_gb):
        import psutil
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(str(self.model_path), **common)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=common['trust_remote_code'],
                                                    dtype=self.dtype, attn_implementation='sdpa')
        # These named components are verified against the pinned FGCLIP2 source.
        if not all(hasattr(model, name) for name in ('vision_model', 'dense_feature_head', 'text_model', 'boxtext_head', 'longtext_head')):
            raise ValueError('Text-only loader requires the FGCLIP2 component layout.')
        model.vision_model = None
        model.dense_feature_head = None
        expected = dict(model.named_parameters())
        index = self.model_path/'model.safetensors.index.json'
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        needed = sum(p.numel() for p in expected.values())*torch.empty((), dtype=self.dtype).element_size()
        available = psutil.virtual_memory().available
        if self.device.type == 'cuda':
            available = min(available, torch.cuda.mem_get_info(self.device)[0])
        if available < needed+reserve_gb*1024**3:
            raise MemoryError(f'Text-only weights plus reserve need {(needed/1024**3+reserve_gb):.2f} GiB; {available/1024**3:.2f} GiB available.')
        loaded = set()
        with torch.no_grad():
            for shard in shards:
                with safe_open(self.model_path/shard, framework='pt', device='cpu') as archive:
                    for name in archive.keys():
                        if name not in expected:
                            continue
                        source = archive.get_slice(name)
                        shape = source.get_shape()
                        if tuple(shape) != tuple(expected[name].shape):
                            raise ValueError(f'Shape mismatch: {name}')
                        value = torch.empty(shape, dtype=self.dtype, device=self.device)
                        if shape:
                            rows = max(1, 8*1024**2//(max(1, math.prod(shape[1:]))*4))
                            for start in range(0, shape[0], rows):
                                value[start:start+rows].copy_(source[start:start+rows])
                        else:
                            value.copy_(archive.get_tensor(name))
                        owner, _, leaf = name.rpartition('.')
                        module = model.get_submodule(owner) if owner else model
                        setattr(module, leaf, torch.nn.Parameter(value, requires_grad=False))
                        loaded.add(name)
            if loaded != set(expected):
                raise ValueError(f'Missing text weights: {set(expected)-loaded}')
            for module in model.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    setattr(module, name, buffer.to(self.device))
        self.loaded_weight_bytes = needed
        return model

    def encode_image_preprocessed(self, *args, **kwargs):
        raise RuntimeError('Text-only encoder: use original cached image features.')
