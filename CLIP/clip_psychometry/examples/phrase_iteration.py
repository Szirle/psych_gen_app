"""Prespecified family interventions on the exposed development sample."""
from pathlib import Path
import json
from ..schema import Dataset, write_json
from ..features import Representation
from ..evaluation import ComparisonConfig, compare


def main():
    root = Path(__file__).resolve().parents[2]/'research/clip_psychometry/iteration_01'
    data = Dataset.load(root/'expanded.npz')
    original = set(range(512))
    families = ('eyewear_explicit', 'smile_mechanics', 'photo_conditions', 'presentation_geometry')
    # Exact text/channel duplicates would otherwise change metric multiplicity.
    seen = {(c.text, c.channel) for c in data.registry.items[:512]}
    additions = {}
    for family in families:
        additions[family] = {i for i in data.registry.columns(family)
                             if (data.registry.items[i].text, data.registry.items[i].channel) not in seen}
    banks = {'original': original}
    banks.update({f'add_{f}': original | additions[f] for f in families})
    concrete = set().union(*additions.values())
    banks['replace_direct'] = original-set(data.registry.columns('direct_impressions', closure=True)) | concrete
    banks['replace_authenticity'] = original-set(data.registry.columns('smile_authenticity', closure=True)) | additions['smile_mechanics']
    variants = {k: Representation(tuple(sorted(v))) for k, v in banks.items()}
    write_json(root/'study3/design.json', {'columns': {k: sorted(v) for k,v in banks.items()},
        'duplicates_excluded': True, 'interpretation': 'Exploratory development comparisons; no untouched test sample.'})
    report, _ = compare(data, variants, config=ComparisonConfig(),
        contrasts={k: ('original', k) for k in banks if k != 'original'}, output=root/'study3')
    print(json.dumps(report['metrics'], indent=2))

if __name__ == '__main__':
    main()
