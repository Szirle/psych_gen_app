"""Traceable image evidence for inspection, not automated visual verdicts; ch. 4."""
import csv
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .features import conjunction
from .metrics import conjunction_overlap
from .metrics import contrast_audit
from .schema import write_json


def _font(size):
    for name in ('DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _permit(data):
    if data.metadata.get('exposure') not in ('development', 'discovery', 'previously_exposed'):
        raise ValueError('Declare metadata.exposure before inspecting images: development, discovery, or previously_exposed. Inspection consumes a holdout.')


def _filename(text):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', text).strip('-')[:100]


def _independent_extremes(data, scores, count=4):
    order = np.argsort(scores, kind='stable')
    used, low, high = set(), [], []
    for indices, result in ((order, low), (order[::-1], high)):
        for i in indices:
            if not np.isfinite(scores[i]) or data.groups[i] in used:
                continue
            result.append(int(i))
            used.add(data.groups[i])
            if len(result) == count:
                break
    if len(low) != count or len(high) != count:
        raise ValueError('Need eight distinct image/identity groups for disjoint extreme panels.')
    return low, high


def _panel_card(data, title, panels, output, *, subtitle='', scores=None, extra=None):
    """Each labeled panel contains a 2x2 grid; up to four panels per card."""
    _permit(data)
    cell, gap, header = 256, 24, 160
    width = len(panels)*(2*cell+gap)+gap
    canvas = Image.new('RGB', (width, header+2*(cell+62)+gap), '#f6f3ec')
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((gap, 12), '\n'.join(textwrap.wrap(title, max(40, width//17))), font=_font(28), fill='#172c38', spacing=6)
    draw.text((gap, header-58), subtitle[:width//9], font=_font(16), fill='#435c67')
    records = []
    for panel, (label, rows) in enumerate(panels):
        x0 = gap+panel*(2*cell+gap)
        draw.text((x0, header-30), label, font=_font(19), fill='#172c38')
        for k, row in enumerate(rows[:4]):
            row = int(row)
            x, y = x0+(k % 2)*cell, header+(k//2)*(cell+62)
            with Image.open(data.paths[row]) as source:
                image = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (cell-8, cell-8))
            canvas.paste(image, (x+(cell-image.width)//2, y+(cell-image.height)//2))
            score = None if scores is None else float(scores[row])
            line = f'{data.ids[row]}' + ('' if score is None else f' | {score:.4f}')
            draw.text((x+4, y+cell), line[:38], font=_font(15), fill='#172c38')
            detail = '' if extra is None else str(extra[row])
            draw.text((x+4, y+cell+24), detail[:37], font=_font(13), fill='#435c67')
            records.append(dict(panel=label, rank=k+1, row=row, id=data.ids[row], path=data.paths[row],
                                group=str(data.groups[row]), score=score, detail=detail))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    write_json(output.with_suffix('.json'), dict(title=title, subtitle=subtitle, exposure=data.metadata['exposure'],
               data_sha256=data.fingerprint(), panels=records, status='Generated for inspection; no visual verdict inferred.'))
    return str(output)


def extreme_cards(data, output, *, columns=None):
    columns = range(data.x.shape[1]) if columns is None else columns
    paths = []
    for j in columns:
        item, values = data.registry.items[j], data.x[:, j]
        low, high = _independent_extremes(data, values)
        paths.append(_panel_card(data, item.text, [('LOWEST SCORE', low), ('HIGHEST SCORE', high)],
                    Path(output)/f'{j:04d}-{_filename(item.id)}.png', scores=values,
                    subtitle=f'{item.channel} | intended cue: {item.cue} | {data.metadata["exposure"]} | scores are not probabilities'))
    return paths


def boundary_card(data, column, output):
    order = np.argsort(data.x[:, column], kind='stable')
    rows = order[np.round(np.linspace(.05, .95, 8)*(len(order)-1)).astype(int)]
    return _panel_card(data, data.registry.items[column].text, [('LOW/MID QUANTILES', rows[:4]), ('MID/HIGH QUANTILES', rows[4:])],
                       output, scores=data.x[:, column], subtitle='Score-range audit; quantiles are not presence thresholds.')


def contrast_card(data, a, b, training_x, output, *, target=0):
    audit, _, contrast = contrast_audit(training_x[:, [a, b]], data.x[:, [a, b]], data.y[:, target])
    low, high = _independent_extremes(data, contrast)
    title = f'Contrast: {data.registry.items[a].text} MINUS {data.registry.items[b].text}'
    path = _panel_card(data, title, [('SECOND RELATIVELY STRONGER', low), ('FIRST RELATIVELY STRONGER', high)],
                       output, scores=contrast, subtitle='Training-standardized difference. Inspect what differs; do not assume modifier fidelity.')
    write_json(Path(output).with_suffix('.contrast.json'), audit)
    return path, audit


def residual_cards(data, prediction, output, *, targets=None):
    prediction = np.asarray(prediction, float)
    if prediction.shape != data.y.shape:
        raise ValueError('OOF predictions must align with dataset Y.')
    paths = []
    for t in (range(len(data.targets)) if targets is None else targets):
        error = prediction[:, t]-data.y[:, t]
        low, high = _independent_extremes(data, error)
        extra = [f'human={y:.3f}; prediction={p:.3f}' for y, p in zip(data.y[:, t], prediction[:, t])]
        paths.append(_panel_card(data, f'{data.targets[t]}: largest signed prediction errors',
            [('MOST NEGATIVE ERRORS', low), ('MOST POSITIVE ERRORS', high)],
            Path(output)/f'{_filename(data.targets[t])}-residuals.png', scores=error, extra=extra,
            subtitle='OOF discovery only. Error = prediction - human mean. Extremes can share the same sign.'))
    return paths


def conjunction_card(data, a, b, lexical, training_x, output, *, fraction=.1):
    component = conjunction(training_x[:, a], training_x[:, b], data.x[:, a], data.x[:, b])
    lexical_score = data.x[:, lexical]
    audit = conjunction_overlap(component, lexical_score, fraction=fraction)
    ranked = lambda rows, score: sorted(rows, key=lambda i: (-score[i], data.ids[i]))[:4]
    panels = [('COMPONENT TOP', np.argsort(component, kind='stable')[-4:][::-1]),
              ('LEXICAL TOP', np.argsort(lexical_score, kind='stable')[-4:][::-1]),
              ('COMPONENT ONLY', ranked(audit['component_only'], component)),
              ('LEXICAL ONLY', ranked(audit['lexical_only'], lexical_score))]
    extra = [f'AND rank={c:.3f}; lexical={l:.3f}' for c, l in zip(component, lexical_score)]
    path = _panel_card(data, data.registry.items[lexical].text, panels, output, extra=extra,
                       subtitle='Component rule: min(training ECDF(A), training ECDF(B)). Disagreement needs annotation.')
    write_json(Path(output).with_suffix('.overlap.json'), audit)
    return path, audit


def annotation_sheet(data, column, output, *, strata=5, per_stratum=10, seed=0):
    """Stratified probability sample: blinded sheet plus separate design file."""
    _permit(data)
    if strata < 1 or per_stratum < 1:
        raise ValueError('Positive strata/sample sizes required.')
    rng, selected = np.random.default_rng(seed), []
    for k, pool in enumerate(np.array_split(np.argsort(data.x[:, column], kind='stable'), strata)):
        if len(pool):
            count = min(per_stratum, len(pool))
            selected.extend((int(i), k, count/len(pool)) for i in rng.choice(pool, count, replace=False))
    rng.shuffle(selected)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'blind_annotations.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'path', 'present_0_or_1', 'unclear', 'observable_evidence', 'annotator'])
        for i, _, _ in selected:
            w.writerow([data.ids[i], data.paths[i], '', '', '', ''])
    write_json(output/'sampling_design.json', dict(column=column, item_id=data.registry.items[column].id,
        seed=seed, exposure=data.metadata['exposure'], data_sha256=data.fingerprint(),
        rows=[dict(id=data.ids[i], stratum=k, inclusion_probability=p, score=float(data.x[i, column])) for i, k, p in selected],
        instruction='Keep this design file hidden during annotation. Use an independent cue rubric; unclear is missing, not absent.'))
    return len(selected)


def blind_contact_sheets(data, rows, output, *, seed=0, page_size=24):
    """Random-order ID-only sheets for a specified sample; no scores or ranks.

    Caller retains sampling design separately and must declare what population
    the sample covers. This renders evidence for annotation, not visual QA.
    """
    _permit(data)
    rows = np.asarray(rows, int)
    if len(set(rows.tolist())) != len(rows) or page_size != 24:
        raise ValueError('Use unique rows and 24 images per page.')
    rows = np.random.default_rng(seed).permutation(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(rows), page_size)):
        batch = rows[start:start+page_size]
        sheet = Image.new('RGB', (6*220, 4*240+48), '#faf9f6')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), f'Observable cue annotation | page {page+1} | IDs only', font=_font(22), fill='#172c38')
        for k, row in enumerate(batch):
            x, y = (k % 6)*220, 48+(k//6)*240
            with Image.open(data.paths[row]) as original:
                face = ImageOps.contain(ImageOps.exif_transpose(original).convert('RGB'), (210, 210))
            sheet.paste(face, (x+(220-face.width)//2, y))
            draw.text((x+5, y+213), data.ids[row], font=_font(18), fill='#172c38')
        sheet.save(output/f'page-{page+1:02d}.png')
    write_json(output/'order.json', dict(seed=seed, pages=[data.ids[i] for i in rows],
               note='IDs only; sample membership may be informative. Score-blinding is not independent human validation.'))
    return [str(output/f'page-{i+1:02d}.png') for i in range((len(rows)+23)//24)]
