"""Compare the old CPU MI path with batched MI; no model or checkpoints."""
import argparse
import json
from time import perf_counter

import numpy as np
from sklearn.feature_selection import mutual_info_regression
import torch
from threadpoolctl import threadpool_limits

from .data import batched_mutual_information


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=642, help='80%% of an 803-image training split.')
    parser.add_argument('--features', type=int, default=1500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memory-gib', type=float, default=4.)
    parser.add_argument('--cpu-threads', type=int, default=8)
    args = parser.parse_args()
    if args.samples < 4 or args.features < 4 or args.cpu_threads < 1:
        parser.error('Need >=4 samples/features and positive cpu-threads')
    if not np.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error('memory-gib must be finite and positive')
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    rng = np.random.default_rng(20260923)
    x = rng.normal(size=(args.samples, args.features)).astype(np.float32)
    y = (.5+.2*np.tanh(x[:, 0]+rng.normal(size=args.samples)*.3)).astype(np.float32)
    # Exercise constant features, exact duplicates and quantized similarities.
    x[:, 1], x[:, 2], x[:, 3] = 1., x[:, 0], np.round(x[:, 3], 2)
    with threadpool_limits(limits=args.cpu_threads):
        batched_mutual_information(x[:32, :4], y[:32], seed=7, device=device, memory_gib=args.memory_gib)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        actual = batched_mutual_information(x, y, seed=7, device=device, memory_gib=args.memory_gib)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        batched_seconds = perf_counter()-start
        start = perf_counter()
        reference = mutual_info_regression(x, y, random_state=7, n_neighbors=3, n_jobs=1)
        sklearn_seconds = perf_counter()-start
    error = float(np.max(np.abs(actual-reference)))
    print(json.dumps(dict(samples=args.samples, features=args.features, device=str(device),
        batched_seconds=batched_seconds, sklearn_seconds=sklearn_seconds,
        speedup=sklearn_seconds/batched_seconds, max_absolute_error=error,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None,
        note='One synthetic target/repeat; not an end-to-end training benchmark.'), indent=2), flush=True)
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=0)


if __name__ == '__main__':
    main()
