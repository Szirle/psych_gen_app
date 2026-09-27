"""Conservative FFHQ views, shared by frozen-feature and live-image training."""
import math
from pathlib import Path
import time
import fcntl
import shutil
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

from .models import visual_features
from .assets import atomic_save
from ..fgclip2_face_impressions import array_hash
from .tensor_views import PreparedImages, feature_noise


@contextmanager
def cache_owner(cache):
    """One exclusive owner for the disposable augmentation subtree."""
    root = Path(cache)/'augmentation-v2'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.fit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another fit owns the augmentation cache. Use a separate --cache for concurrent training.')
        try:
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def remove_banks(root):
    removed = 0
    for path in root.iterdir():
        if path.name == '.fit.lock':
            continue  # Never unlink the ownership lock's inode.
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()  # Do not follow symlinks outside the augmentation tree.
        removed += 1
    print(f'Augmentation cleanup: removed {removed} entries from {root}; original caches retained.', flush=True)


def cleanup_cache(cache):
    """Startup cleanup also covers kernel-only and already-completed runs."""
    with cache_owner(cache) as root:
        remove_banks(root)


def add_arguments(parser):
    parser.add_argument('--augmentation', choices=('none', 'flip', 'ffhq'), default='ffhq')
    for name, value in (('views', 16), ('refresh', 20), ('encode-batch', 1024), ('bank-views', 32)):
        help_text = {'encode-batch': 'Maximum frozen augmentation encoding batch, packed in 32-image shards; CUDA OOM backs off.',
                     'bank-views': 'Persistent views per image; 0 grows indefinitely, otherwise rotate after filling the bank.'}.get(name)
        parser.add_argument('--augmentation-'+name, type=int, default=value, help=help_text)
    for name, value in (('clean-prob', .1), ('translate', .015), ('scale', .02),
                        ('noise', .01), ('blur', .5), ('color', 0.), ('feature-noise', 0.),
                        ('cache-gib', 48.), ('reuse-fraction', 15/16)):
        parser.add_argument('--augmentation-'+name, type=float, default=value)
    parser.add_argument('--augmentation-cache-device', choices=('auto', 'cpu', 'cuda'), default='cuda',
                        help='auto uses CUDA when the cache fits its GiB budget and 25%% of free VRAM; otherwise host RAM.')


def validate(args):
    for name in ('views', 'refresh', 'encode_batch'):
        if getattr(args, 'augmentation_'+name) < 1:
            raise ValueError(f'augmentation-{name} must be positive')
    if args.augmentation_encode_batch < 32:
        raise ValueError('augmentation-encode-batch must be at least the 32-image cache shard size')
    if args.augmentation_bank_views < args.augmentation_views:
        raise ValueError('augmentation-bank-views must be at least augmentation-views; unlimited banks are no longer supported')
    bounds = dict(reuse_fraction=1., clean_prob=1., translate=.05, scale=.1, noise=.05, blur=2., color=.1, feature_noise=.1)
    for name, maximum in bounds.items():
        value = getattr(args, 'augmentation_'+name)
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f'augmentation-{name} must be in [0, {maximum}]')
    if not math.isfinite(args.augmentation_cache_gib) or args.augmentation_cache_gib <= 0:
        raise ValueError('augmentation-cache-gib must be positive')


class TrainingViews:
    """Acquire ownership on construction; the fit must call close in finally."""
    SHARD = 32

    def __init__(self, original, rows, args, *, scores_only=False):
        self.original, self.rows, self.args = original, np.asarray(rows), args
        self.scores_only = scores_only
        self.lookup = np.full(len(original.data.paths), -1, dtype=np.int64)
        self.lookup[self.rows] = np.arange(len(self.rows))
        self.cache, self.generation = None, None
        self.rng = np.random.default_rng(args.seed)
        if not hasattr(original, '_prepared_images'):
            original._prepared_images = PreparedImages(original)
        self.prepared = original._prepared_images
        # Banks never survive a fit, so identity-indexed directories offer no
        # reuse and make stale accumulation harder to observe. One slot only.
        self.disk = Path(args.cache)/'augmentation-v2'/'active'
        self.score_key = array_hash(np.asarray(original.bank['phrases']))
        self.encode_batch = max(self.SHARD, args.augmentation_encode_batch)
        self.lock = None
        self._open()

    def _open(self):
        if self.lock is not None:
            return self
        owner = cache_owner(self.args.cache)
        root = owner.__enter__()
        self.lock = owner
        try:
            # Also reclaim banks abandoned by older runs or killed processes.
            # This subtree contains augmentations only, never clean visual .pt.
            remove_banks(root)
            return self
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        # Optional compatibility convenience, not the acquisition mechanism.
        return self._open()

    def close(self):
        self.cache = None
        if self.lock is not None:
            try:
                remove_banks(self.disk.parent)
            finally:
                self.lock.__exit__(None, None, None)
                self.lock = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getattr__(self, name):
        return getattr(self.original, name)

    @property
    def enabled(self):
        return self.args.augmentation != 'none' and self.args.augmentation_clean_prob < 1

    def set_epoch(self, epoch, *, live):
        self.rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, epoch, 9187]))
        generation = epoch//self.args.augmentation_refresh
        if self.enabled and not live and generation != self.generation:
            self.cache = None  # Release the old generation before allocating.
            self._encode(generation)
            self.generation = generation

    def _view_ids(self, generation):
        k, bank = self.args.augmentation_views, self.args.augmentation_bank_views
        fresh = math.ceil(k*(1-self.args.augmentation_reuse_fraction))
        end = bank+generation*fresh
        new = list(range(end-fresh, end))
        previous = np.arange(end-bank, end-fresh)
        rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, generation, 517]))
        return new+list(rng.choice(previous, k-fresh, replace=False))

    def _bank_ids(self, generation):
        fresh = math.ceil(self.args.augmentation_views*(1-self.args.augmentation_reuse_fraction))
        start = generation*fresh
        return range(start, start+self.args.augmentation_bank_views)

    def _prune(self, retained):
        if not self.disk.exists():
            return
        for kind in self.disk.iterdir():
            if not kind.is_dir():
                continue
            if kind.name not in ('dense', 'scores-'+self.score_key):
                shutil.rmtree(kind)
                continue
            for view in kind.iterdir():
                if view.is_dir() and (not view.name.isdecimal() or int(view.name) not in retained):
                    shutil.rmtree(view)

    def _paths(self, view, start):
        name = f'{view:05d}/{start:06d}.pt'
        return self.disk/'dense'/name, self.disk/('scores-'+self.score_key)/name

    def _scores(self, features):
        z, dense, mask, _ = (v.to(self.engine.device) for v in features)
        # Dense values are quantized before deriving scores, exactly as on disk.
        z, dense = F.normalize(z.float(), dim=-1), F.normalize(dense.float(), dim=-1)
        text = tuple(t.to(self.engine.device) for t in self.text)
        local = []
        for start in range(0, len(text[1]), 256):
            alignment = dense@text[1][start:start+256].T
            local.append(alignment.masked_fill(~mask[:, :, None], -torch.inf).amax(1))
        return z.half(), torch.cat((z@text[0].T, *local), -1)

    def _put(self, slot, start, values):
        if slot is None:  # Replay-bank prefill stays on disk, not in VRAM/RAM.
            return
        for destination, value in zip(self.cache, values):
            destination[slot, start:start+len(value)].copy_(value)

    @staticmethod
    def _save(values, path):
        # Reuse the existing atomic writer, with a unique staging filename so
        # concurrent experiments can safely populate the same label-free bank.
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as handle:
            staging = Path(handle.name)
        try:
            atomic_save(values, staging)
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)
            staging.with_suffix(staging.suffix+'.tmp').unlink(missing_ok=True)

    def _save_batch(self, tasks, dense, scores):
        position = 0
        for _, view, start, rows in tasks:
            end = position+len(rows)
            dense_path, score_path = self._paths(view, start)
            self._save(tuple(t[position:end].clone() for t in dense), dense_path)
            if scores is not None:
                self._save(tuple(t[position:end].clone() for t in scores), score_path)
            position = end

    def _seed(self, view, start):
        return int(np.random.SeedSequence([self.args.seed, view, start, 8301]).generate_state(1)[0])

    def _generate(self, tasks):
        rows = np.concatenate([task[3] for task in tasks])
        segments = [(len(r), self._seed(view, start)) for _, view, start, r in tasks]
        inputs = self.prepared.batch(rows, self.args, segments)
        features = visual_features(self.engine, inputs)
        features = feature_noise(features, self.args.augmentation_feature_noise, segments)
        return tuple(t if t.dtype == torch.bool else t.half() for t in features)

    def _try_generate(self, tasks):
        # Catch outside _generate so the failed frame releases its CUDA tensors.
        try:
            return self._generate(tasks)
        except torch.cuda.OutOfMemoryError:
            return None

    @torch.no_grad()
    def _encode(self, generation):
        if self.lock is None:
            raise RuntimeError('Cannot encode augmentations after TrainingViews.close(); create a new fit-owned bank.')
        args, device = self.args, self.engine.device
        started = time.monotonic()
        k, n = args.augmentation_views, len(self.rows)
        templates = (self.features[0], torch.empty(1, self.scores.shape[1], dtype=torch.float32)) if self.scores_only else self.features
        dtypes = (torch.float16, torch.float32) if self.scores_only else (torch.float16, torch.float16, torch.bool, torch.float16)
        shapes = [(k, n, *t.shape[1:]) for t in templates]
        size = sum(math.prod(shape)*torch.empty((), dtype=dtype).element_size() for shape, dtype in zip(shapes, dtypes))
        storage = 'cpu'
        if args.augmentation_cache_device != 'cpu' and device.type == 'cuda':
            available = torch.cuda.mem_get_info(device)[0]
            if size <= min(args.augmentation_cache_gib*1024**3, available*.5):
                storage = device
            elif args.augmentation_cache_device == 'cuda':
                raise ValueError('Augmentation cache exceeds its VRAM budget; reduce views/patches or choose cpu/auto.')
        elif args.augmentation_cache_device == 'cuda':
            raise ValueError('CUDA augmentation cache requested without a CUDA encoder')
        self.cache = tuple(torch.empty(shape, dtype=dtype, device=storage) for shape, dtype in zip(shapes, dtypes))
        views, pending, reused = self._view_ids(generation), [], 0
        bank = list(self._bank_ids(generation))
        # Evict before generating replacements: on-disk view IDs never exceed
        # bank_views, including when regenerating after a resumed epoch.
        self._prune(set(bank))
        slots = {view: slot for slot, view in enumerate(views)}
        # print(f'Augmentation generation {generation}: views {views}; {k} × {n} images; '
        #       f'{size/1024**3:.2f} GiB on {storage}; encoding batch up to {self.encode_batch}.', flush=True)
        dense_size = sum(n*math.prod(t.shape[1:])*(1 if t.dtype == torch.bool else 2) for t in self.features)
        if generation == 0:
            bound = f'{dense_size*args.augmentation_bank_views/1024**3:.2f} GiB dense bank'
            # print(f'  Disk budget for this training split: {bound}, plus optional score sidecars.', flush=True)
        if self.generation is None:
            print(f'  Prefilling {args.augmentation_bank_views-k} reserve + {k} other views before training.', flush=True)
        was_training = self.engine.model.training
        self.engine.model.eval()
        writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='augmentation-cache')
        writes = []
        try:
            for view in bank:
                slot = slots.get(view)
                for start in range(0, n, self.SHARD):
                    dense_path, score_path = self._paths(view, start)
                    path = score_path if self.scores_only and score_path.exists() else dense_path
                    if path.exists():
                        if slot is None:
                            reused += min(self.SHARD, n-start)
                            continue
                        values = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                        if self.scores_only and path == dense_path:
                            values = self._scores(values)
                            self._save(tuple(t.cpu() for t in values), score_path)
                        self._put(slot, start, values)
                        reused += len(values[0])
                        del values
                    else:
                        pending.append((slot, view, start, self.rows[start:start+self.SHARD]))
            offset, encoded = 0, 0
            while offset < len(pending):
                count = max(1, self.encode_batch//self.SHARD)
                tasks = pending[offset:offset+count]
                features = self._try_generate(tasks)
                if features is None:
                    torch.cuda.empty_cache()
                    if len(tasks) == 1:
                        raise RuntimeError('CUDA OOM encoding one 32-image augmentation shard. '
                                           'Reduce patch count or resident cache budget.')
                    self.encode_batch = max(self.SHARD, (len(tasks)//2)*self.SHARD)
                    print(f'Augmentation encoder OOM: retrying at {self.encode_batch} images.', flush=True)
                    continue
                # One transfer per batched feature tensor, then views into CPU
                # storage. Clone shard slices to avoid serializing the full batch.
                cpu = tuple(t.cpu() for t in features)
                score_values = self._scores(features) if self.scores_only else None
                score_cpu = tuple(t.cpu() for t in score_values) if score_values is not None else None
                # Bound pending host buffers while overlapping disk writes with
                # the next GPU forward. Future.result propagates I/O failures.
                if len(writes) >= 2:
                    writes.pop(0).result()
                writes.append(writer.submit(self._save_batch, tasks, cpu, score_cpu))
                position = 0
                for slot, view, start, rows in tasks:
                    end = position+len(rows)
                    self._put(slot, start, tuple(t[position:end] for t in (score_values or features)))
                    position = end
                encoded += position
                offset += len(tasks)
                del features, cpu, score_values, score_cpu
                # print(f'  augmented encodings: {encoded}/{len(bank)*n-reused} new, {reused} reused.', flush=True)
            for future in writes:
                future.result()
        finally:
            writer.shutdown(wait=True)
            self.engine.model.train(was_training)
        # print(f'Augmentation ready in {time.monotonic()-started:.1f}s; disk bank: {self.disk}', flush=True)

    def _choices(self, rows):
        indices = self.lookup[rows]
        if np.any(indices < 0):
            raise ValueError('Augmentation requested for a row outside this fit training split')
        use = self.rng.random(len(rows)) >= self.args.augmentation_clean_prob
        views = self.rng.integers(self.args.augmentation_views, size=len(rows))
        return indices, views, use

    def _mix(self, clean, rows):
        indices, views, use = self._choices(rows)
        if self.cache is None:
            raise RuntimeError('Call set_epoch before requesting augmented cached features')
        result = []
        for original, cached in zip(clean, self.cache):
            output = original.clone()
            ix = torch.as_tensor(indices[use], device=cached.device)
            v = torch.as_tensor(views[use], device=cached.device)
            output[torch.as_tensor(use, device=output.device)] = cached[v, ix].to(output)
            result.append(output)
        return tuple(result)

    def score_batch(self, rows):
        z = self.features[0][rows].to(self.engine.device)
        scores = torch.as_tensor(self.scores[rows], device=self.engine.device)
        if self.enabled:
            z, scores = self._mix((z, scores), rows)
        return F.normalize(z.float(), dim=-1), scores.float()

    def batch(self, rows, live=False, augment=False):
        if not augment or not self.enabled:
            if live:
                return visual_features(self.engine, self.prepared.batch(rows, None, None))
            return self.original.batch(rows, live, False)
        if live:
            _, _, use = self._choices(rows)
            segments = [(len(rows), int(self.rng.integers(0, 2**32)))]
            enabled = torch.as_tensor(use, device=self.engine.device)
            inputs = self.prepared.batch(rows, self.args, segments, enabled)
            return feature_noise(visual_features(self.engine, inputs), self.args.augmentation_feature_noise, segments, enabled)
        features = self.original.batch(rows, False)
        z, dense, mask, coordinates = self._mix(features, rows)
        return F.normalize(z.float(), dim=-1), F.normalize(dense.float(), dim=-1), mask, coordinates.float()
