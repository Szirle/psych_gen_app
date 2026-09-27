"""Future prompt-optimizer building block, not a trainer; methodology 12.2.

Torch is imported only on invocation. Run locally outside sandbox for MPS.
"""


def kernel_predict(support, targets, query, *, alpha=.1, gamma=.1, family='rbf',
                   detach_support=False, detach_dual=False, scale_floor=1e-6):
    """Centered KRR differentiable through scores, standardization and solve.

    alpha is the additive Gram penalty, not n*lambda. Missing labels must be
    handled as separate observed-row groups by the caller. Float32/64 only.
    detach_dual=True changes the gradient objective; not equivalent to refitting.
    """
    import torch
    if alpha <= 0 or gamma <= 0 or scale_floor <= 0 or family not in ('linear', 'poly2', 'rbf'):
        raise ValueError('Positive alpha, gamma and scale_floor plus a supported family required.')
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1] or support.shape[1] == 0 or len(support) < 2:
        raise ValueError('Aligned nonempty support/query matrices required.')
    if support.dtype not in (torch.float32, torch.float64) or targets.shape[0] != len(support) or targets.ndim not in (1, 2):
        raise ValueError('Float32/64 scores and aligned targets required; avoid half-precision solves.')
    if not all(torch.isfinite(t).all() for t in (support, targets, query)):
        raise ValueError('Scores and target labels must be finite.')
    support = support.detach() if detach_support else support
    mean = support.mean(0)
    scale = support.std(0, correction=0).clamp_min(scale_floor)
    z, q = (support-mean)/scale, (query-mean)/scale
    def kernel(a, b):
        dot = a@b.T/a.shape[1]
        if family == 'linear':
            return dot
        if family == 'poly2':
            return (1+dot).square()
        distance = a.square().mean(1)[:, None]+b.square().mean(1)[None, :]-2*dot
        return (-gamma*distance.clamp_min(0)).exp()
    gram = kernel(z, z)
    km, grand = gram.mean(0), gram.mean()
    centered = gram-km[:, None]-km[None, :]+grand
    y = targets.to(device=support.device, dtype=support.dtype)
    ymean = y.mean(0)
    dual = torch.linalg.solve(centered+alpha*torch.eye(len(z), dtype=z.dtype, device=z.device), y-ymean)
    if detach_dual:
        dual = dual.detach()
    cross = kernel(q, z)
    cross = cross-cross.mean(1, keepdim=True)-km[None, :]+grand
    return cross@dual+ymean
