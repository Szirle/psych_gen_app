"""Focused real-data study, not a smoke test or a new headline benchmark.

Run from CLIP: python -m clip_psychometry.examples.development_study
Uses only the historical 400 development rows; no image/model encoding.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ..adapters import from_legacy_features
from ..cards import boundary_card, conjunction_card, contrast_card, extreme_cards, residual_cards, annotation_sheet
from ..diagnostics import detector_summary
from ..evaluation import ComparisonConfig, compare
from ..features import Representation
from ..metrics import complementarity, correlation
from ..schema import Registry, write_json


def main():
    repository = Path(__file__).resolve().parents[2]
    source, output = repository/'research/trustworthiness', repository/'research/clip_psychometry/development'
    data = from_legacy_features([source/f'p{p}/trustworthy-features.npz' for p in (128, 256)],
                                ('p128', 'p256'), phrase_groups=source/'phrase_groups.json')
    protocol = json.loads((source/'protocol.json').read_text())
    if tuple(protocol['face_names']) != data.ids:
        raise ValueError('Historical development indices refer to a different image order.')
    data = data.subset(protocol['development_indices'], exposure='development')
    # Explicit role closure covers the four gender-presentation conjunctions.
    items = tuple(replace(c, roles=(*c.roles, 'authenticity_conjunction'))
                  if c.cue == 'cue_combinations' and ('genuine smile' in c.text or 'insincere smile' in c.text)
                  else c for c in data.registry.items)
    data.registry = Registry(items, {'smile_authenticity': ['authenticity_conjunction']})
    data.metadata.update(development_protocol=str(source/'protocol.json'),
        purpose='Prespecified family-removal comparison and modifier-fidelity hypothesis generation.',
        evaluation_excluded=604, prior_exposure='Historical development images; no independent confirmation claim.')
    data.save(output/'data.npz')
    all_columns = set(range(data.x.shape[1]))
    direct = set(data.registry.columns('direct_impressions', closure=True))
    authenticity = set(data.registry.columns('smile_authenticity', closure=True))
    insincere = {i for i, c in enumerate(items) if c.text == 'an insincere smile'}
    variants = {name: Representation(tuple(sorted(columns))) for name, columns in (
        ('full', all_columns), ('without_direct', all_columns-direct),
        ('without_authenticity_role', all_columns-authenticity),
        ('without_both_roles', all_columns-direct-authenticity),
        ('without_insincere_phrase', all_columns-insincere))}
    config = ComparisonConfig()
    contrasts = dict(direct_utility=('without_direct', 'full'),
                     authenticity_role_utility=('without_authenticity_role', 'full'),
                     insincere_unique_utility=('without_insincere_phrase', 'full'))
    write_json(output/'question.json', dict(question=data.metadata['purpose'], contrasts=contrasts,
        semantic_limit='Nominal direct family has substitutes outside its group. Authenticity role includes four declared conjunctions; other implicit smile proxies remain.',
        hypotheses=['Positive marginal association of insincere need not imply absent conditional utility.',
                    'Direct adjective-family marginal weakness need not imply zero joint utility.',
                    'Phrase-wise removal and role-wise removal ask different questions.']))
    report, predictions = compare(data, variants, config=config, contrasts=contrasts, output=output)
    interaction = complementarity(data.y[:, 0], predictions['without_both_roles'][:, 0],
        predictions['without_authenticity_role'][:, 0], predictions['without_direct'][:, 0], predictions['full'][:, 0],
        groups=data.groups, seed=config.seed)
    write_json(output/'complementarity.json', interaction)
    write_json(output/'detectors.json', detector_summary(data))
    lookup = {(c.text, c.channel): i for i, c in enumerate(items)}
    phrases = ('a trustworthy face', 'a joyful face', 'a genuine smile', 'an insincere smile',
               'a face wearing clear eyeglasses', 'a face wearing dark sunglasses')
    chosen = [lookup[(p, ch)] for ch in ('p128', 'p256') for p in phrases]
    modifier = {}
    for channel in ('p128', 'p256'):
        a, b = (data.x[:, lookup[(p, channel)]] for p in ('a genuine smile', 'an insincere smile'))
        modifier[channel] = dict(genuine_insincere_r=correlation(a, b),
                                 genuine_target_r=correlation(a, data.y[:, 0]),
                                 insincere_target_r=correlation(b, data.y[:, 0]))
    write_json(output/'modifier_diagnostics.json', modifier)
    # Generate targeted qualitative evidence, not an exhaustive disk-heavy gallery.
    extreme_cards(data, output/'cards', columns=chosen)
    residual_cards(data, predictions['full'], output/'cards')
    boundary_card(data, lookup[('a face wearing clear eyeglasses', 'p128')], output/'cards/glasses-boundary.png')
    conjunction_card(data, lookup[('a happy face', 'p128')], lookup[('a face wearing glasses', 'p128')],
                     lookup[('a happy face wearing glasses', 'p128')], data.x, output/'cards/happy-glasses-conjunction.png')
    # This diagnostic was added after the original family comparison; it remains
    # exploratory even when a later reproduction computes it in the same run.
    modifier_contrast = {}
    for channel in ('p128', 'p256'):
        a, b = [lookup[(p, channel)] for p in ('a genuine smile', 'an insincere smile')]
        _, modifier_contrast[channel] = contrast_card(data, a, b, data.x, output/f'cards/{channel}-smile-contrast.png')
    write_json(output/'modifier_contrast.json', dict(status='Exploratory follow-up after the original family comparison.', channels=modifier_contrast))
    annotation_sheet(data, lookup[('a face wearing clear eyeglasses', 'p128')], output/'annotations', seed=config.seed)
    write_json(output/'artifact_status.json', dict(visual_inspection_performed=False,
        generated_cards=17, annotations_completed=False, new_encodings=0, augmented_views=0,
        reason='Only existing development scores were used. Cards await inspection; cue fidelity is unresolved.'))
    print(json.dumps(dict(metrics={k: v['trustworthy'] for k, v in report['metrics'].items()},
                          contrasts=report['contrasts'], complementarity=interaction, modifier=modifier), indent=2))


if __name__ == '__main__':
    main()
