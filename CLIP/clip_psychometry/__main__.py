"""CLI over portable scores: import, audit, compare, cards, annotations, views."""
import argparse
import json
from pathlib import Path

import numpy as np

from .adapters import from_legacy_features
from .cards import annotation_sheet, extreme_cards, residual_cards
from .diagnostics import detector_summary
from .evaluation import ComparisonConfig, compare
from .features import Representation
from .robustness import augmentation_audit
from .schema import Dataset, write_json


def parse_plan(data, plan):
    variants = {}
    all_columns = set(range(data.x.shape[1]))
    for name, item in plan['variants'].items():
        if sum(k in item for k in ('columns', 'keep_roles', 'drop_roles')) > 1:
            raise ValueError('Choose columns, keep_roles, or drop_roles per variant.')
        if 'columns' in item:
            columns = tuple(item['columns'])
        elif 'keep_roles' in item:
            columns = data.registry.columns(item['keep_roles'], closure=item.get('closure', False))
        elif 'drop_roles' in item:
            columns = tuple(sorted(all_columns-set(data.registry.columns(item['drop_roles'], closure=item.get('closure', True)))))
        else:
            columns = tuple(sorted(all_columns))
        variants[name] = Representation(columns, item.get('transform', 'raw'), item.get('quantile', .5),
                                         item.get('balance_blocks', False),
                                         tuple(item['blocks']) if 'blocks' in item else None)
    cfg = dict(plan.get('config', {}))
    if 'alphas' in cfg:
        cfg['alphas'] = tuple(cfg['alphas'])
    if 'kernels' in cfg:
        cfg['kernels'] = tuple(tuple(k) for k in cfg['kernels'])
    return variants, ComparisonConfig(**cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import-legacy')
    p.add_argument('--features', nargs='+', required=True)
    p.add_argument('--channels', nargs='+', required=True)
    p.add_argument('--phrase-groups')
    p.add_argument('--identity-groups', help='JSON mapping image filename to identity/latent family')
    p.add_argument('--protocol', help='Existing JSON with development_indices and face_names')
    p.add_argument('--exposure', choices=('development', 'discovery', 'previously_exposed', 'undeclared'), default='undeclared')
    p.add_argument('--output', required=True)
    for name in ('audit', 'compare', 'cards', 'annotations', 'views'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True)
        p.add_argument('--output', required=True)
        if name == 'compare':
            p.add_argument('--plan', required=True)
        if name == 'cards':
            p.add_argument('--columns', nargs='+', type=int)
            p.add_argument('--predictions', help='predictions.npz from compare; enables residual cards')
            p.add_argument('--variant', default='full')
        if name == 'annotations':
            p.add_argument('--column', required=True, type=int)
            p.add_argument('--per-stratum', type=int, default=10)
            p.add_argument('--seed', type=int, default=20260924)
        if name == 'views':
            p.add_argument('--view-scores', required=True, help='NPZ with views[n,v,p] and ids in data order')
            p.add_argument('--training-reference', required=True, help='Portable training-only dataset for scales')
    args = parser.parse_args()
    if args.command == 'import-legacy':
        groups = json.loads(Path(args.identity_groups).read_text()) if args.identity_groups else None
        data = from_legacy_features(args.features, args.channels, phrase_groups=args.phrase_groups, identity_groups=groups)
        if args.protocol:
            protocol = json.loads(Path(args.protocol).read_text())
            if tuple(protocol['face_names']) != data.ids:
                raise ValueError('Protocol image order does not match the score archives.')
            data = data.subset(protocol['development_indices'], exposure='development')
            data.metadata['development_protocol'] = str(Path(args.protocol).resolve())
        else:
            data.metadata['exposure'] = args.exposure
        data.save(args.output)
    else:
        data = Dataset.load(args.data)
        if args.command == 'audit':
            write_json(args.output, detector_summary(data))
        elif args.command == 'compare':
            plan = json.loads(Path(args.plan).read_text())
            variants, config = parse_plan(data, plan)
            compare(data, variants, config=config, contrasts=plan.get('contrasts'), output=args.output)
        elif args.command == 'cards':
            extreme_cards(data, args.output, columns=args.columns)
            if args.predictions:
                with np.load(args.predictions, allow_pickle=False) as d:
                    if tuple(d['ids'].tolist()) != data.ids or tuple(d['targets'].tolist()) != data.targets or not np.array_equal(d['y'], data.y, equal_nan=True):
                        raise ValueError('Prediction archive does not align with dataset.')
                    index = d['variants'].tolist().index(args.variant)
                    residual_cards(data, d['predictions'][:, index], Path(args.output)/'residuals')
        elif args.command == 'annotations':
            annotation_sheet(data, args.column, args.output, per_stratum=args.per_stratum, seed=args.seed)
        elif args.command == 'views':
            reference = Dataset.load(args.training_reference)
            if reference.registry.items != data.registry.items:
                raise ValueError('Training reference items differ.')
            with np.load(args.view_scores, allow_pickle=False) as d:
                if tuple(d['ids'].tolist()) != data.ids:
                    raise ValueError('View score image order differs.')
                report = augmentation_audit(reference.x, data.x, d['views'])
            write_json(args.output, report)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
