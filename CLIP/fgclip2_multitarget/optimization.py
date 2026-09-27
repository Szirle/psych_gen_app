"""Joint optimization and preservation of pretrained phrase evidence."""
import math

POLICY = 'joint-v4'


class StepSchedule:
    """One multiplier for every group's base LR, counted in optimizer updates."""
    def __init__(self, total_steps, warmup_fraction, minimum_ratio):
        self.total_steps = total_steps
        self.warmup_steps = min(total_steps-1, math.ceil(total_steps*warmup_fraction))
        self.minimum_ratio = minimum_ratio

    def factor(self, step):
        if self.total_steps == 1:
            return 1.
        step = min(step, self.total_steps-1)
        if step < self.warmup_steps:
            return (step+1)/self.warmup_steps
        remaining = self.total_steps-self.warmup_steps
        progress = (step-self.warmup_steps)/(remaining-1) if remaining > 1 else 1.
        return self.minimum_ratio+(1-self.minimum_ratio)*(1+math.cos(math.pi*progress))/2

    def apply(self, optimizer, step):
        scale = self.factor(step)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*scale


def optimizer_groups(network, backbone, args, kind):
    """Decay mixing matrices, not biases/norms/ensemble scales/token identities."""
    import torch
    groups, components = [], {}
    modules = dict(network.named_modules())

    def add(component, named, lr, decay):
        named = [(name, p) for name, p in named if p.requires_grad]
        if not named:
            return
        if any(p.dtype != torch.float32 for _, p in named):
            raise ValueError(f'{component} optimizer parameters must be FP32')
        components[component] = [p for _, p in named]
        buckets = {False: [], True: []}
        for name, p in named:
            parent, _, leaf = name.rpartition('.')
            module_type = type(modules.get(parent)).__name__
            no_decay = (not decay or p.ndim < 2 or leaf in ('bias', 'r', 's', 'feature_bias', 'queries')
                        or module_type in ('ElementwiseAffine', 'LayerNorm', 'Embedding'))
            buckets[no_decay].append(p)
        for no_decay, parameters in buckets.items():
            if parameters:
                groups.append(dict(params=parameters, lr=lr, base_lr=lr, component=component,
                                   weight_decay=0. if no_decay else decay))

    add('head', [(n, p) for n, p in network.named_parameters() if not n.startswith(('prompt.', 'pool.'))],
        args.head_lr, args.weight_decay)
    add('pool', network.pool.named_parameters(), args.pool_lr, 0.)
    add('prompt', network.prompt.named_parameters(), args.prompt_lr, 0.)
    # Adapters are already a low-rank constraint; avoid decaying their factors.
    add('vision', backbone, args.partial_lr if kind == 'partial' else args.encoder_lr,
        args.vision_weight_decay if kind == 'partial' else 0.)
    return groups, components


def joint_epochs(args):
    if args.smoke:
        return 1
    if args.joint_epochs is not None:
        return args.joint_epochs
    return args.epochs


def phrase_preservation(features, text, original, temperature):
    """KL(original || learned) over phrases on the SAME frozen image features.

    Separate global/local-max distributions preserve phrase agreement, not
    class probabilities or regression predictions. Pooling/vision are outside
    this prompt-only constraint. Text features retain their task gradients.
    """
    import torch
    from torch.nn import functional as F
    z, dense, mask = (v.detach() for v in features[:3])

    def scores(embeddings):
        local = (dense.float() @ embeddings[1].T).masked_fill(~mask[:, :, None], -torch.inf).amax(1)
        return torch.stack((z.float() @ embeddings[0].T, local), 1)/temperature

    with torch.no_grad():
        teacher = scores(original).log_softmax(-1)
    student = scores(text).log_softmax(-1)
    return F.kl_div(student, teacher, log_target=True, reduction='none').sum(-1).mean().clamp_min(0)


def initialize_pool(pool):
    """Zero learned logits, retaining the phrase-alignment softmax prior.

    Leave the key projection random: zeroing both factors would kill gradients.
    """
    import torch
    with torch.no_grad():
        pool.query.zero_()
        pool.spatial.zero_()
