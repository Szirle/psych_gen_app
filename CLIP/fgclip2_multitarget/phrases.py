"""Predeclared language priors for subjective impressions of synthetic faces.

Opposite wordings are feature directions, not assumed slider endpoints. The
supervised readout learns their signs. No target is treated as a factual label.
"""
from ..race_perception.phrases import build_phrase_bank as race_bank


VISUAL = ('skin-color', 'hair-color', 'long-haired', 'outdoors', 'happy', 'alert', 'well-groomed')
BASIC = ('age', 'gender', 'weight')
RACE = ('white', 'black', 'asian', 'middle-eastern', 'hispanic', 'islander', 'native')
DESCRIPTORS = {
    'skin-color': ('light skin', 'dark skin', 'medium skin tone', 'an uneven complexion'),
    'hair-color': ('light hair', 'dark hair', 'gray hair', 'red hair'),
    'long-haired': ('long hair', 'short hair', 'shoulder-length hair', 'a shaved head'),
    'outdoors': ('an outdoorsy appearance', 'an indoor appearance', 'an outdoor background', 'an indoor background'),
    'happy': ('a happy expression', 'an unhappy expression', 'a broad smile', 'a neutral expression'),
    'alert': ('an alert expression', 'a tired expression', 'wide-open eyes', 'sleepy eyes'),
    'well-groomed': ('a well-groomed appearance', 'an unkempt appearance', 'neatly styled hair', 'messy hair'),
    'age': ('a young appearance', 'an old appearance', 'a middle-aged appearance', 'visible facial wrinkles'),
    'gender': ('a feminine appearance', 'a masculine appearance', 'an androgynous appearance', 'an ambiguous gender presentation'),
    'weight': ('a thin face', 'a heavy face', 'full cheeks', 'a narrow face'),
    'asian': ('an Asian appearance', 'a strongly Asian appearance', 'a slightly Asian appearance', 'an ambiguous racial appearance'),
    'white': ('a White appearance', 'a strongly White appearance', 'a slightly White appearance', 'an ambiguous racial appearance'),
    'black': ('a Black appearance', 'a strongly Black appearance', 'a slightly Black appearance', 'an ambiguous racial appearance'),
    'middle-eastern': ('a Middle Eastern appearance', 'a strongly Middle Eastern appearance', 'a slightly Middle Eastern appearance', 'an ambiguous racial appearance'),
    'hispanic': ('a Hispanic appearance', 'a strongly Hispanic appearance', 'a slightly Hispanic appearance', 'an ambiguous racial appearance'),
    'islander': ('a Pacific Islander appearance', 'a strongly Pacific Islander appearance', 'a slightly Pacific Islander appearance', 'an ambiguous racial appearance'),
    'native': ('a Native American appearance', 'a strongly Native American appearance', 'a slightly Native American appearance', 'an ambiguous racial appearance'),
    'attractive': ('an attractive appearance', 'an unattractive appearance', 'a beautiful face', 'a plain appearance'),
    'cute': ('a cute appearance', 'a stern appearance', 'a youthful face', 'a mature face'),
    'dominant': ('a dominant appearance', 'a submissive appearance', 'an assertive expression', 'a timid expression'),
    'trustworthy': ('a trustworthy appearance', 'an untrustworthy appearance', 'a reassuring expression', 'a suspicious expression'),
    'smart': ('an intelligent appearance', 'an unintelligent appearance', 'a thoughtful expression', 'a confused expression'),
    'typical': ('a typical appearance', 'an unusual appearance', 'an ordinary face', 'a distinctive face'),
    'familiar': ('a familiar appearance', 'an unfamiliar appearance', 'an ordinary face', 'a distinctive face'),
    'outgoing': ('an outgoing appearance', 'a reserved appearance', 'a friendly expression', 'a shy expression'),
    'memorable': ('a memorable appearance', 'a forgettable appearance', 'a distinctive face', 'an ordinary face'),
    'smug': ('a smug expression', 'a humble expression', 'a self-satisfied smile', 'a modest expression'),
    'dorky': ('a dorky appearance', 'a socially confident appearance', 'an awkward expression', 'a stylish appearance'),
    'privileged': ('a privileged appearance', 'an underprivileged appearance', 'an affluent appearance', 'a modest appearance'),
    'liberal': ('an appearance perceived as liberal', 'an appearance perceived as conservative', 'an unconventional appearance', 'a conventional appearance'),
    'looks-like-you': ('a familiar face', 'an unfamiliar face', 'a typical face', 'an unusual face'),
    'gay': ('an appearance perceived as gay', 'an appearance perceived as straight', 'a gender-nonconforming appearance', 'a gender-conforming appearance'),
    'electable': ('an electable appearance', 'an unelectable appearance', 'a leader-like appearance', 'an unconfident appearance'),
    'godly': ('an appearance perceived as religious', 'an appearance perceived as nonreligious', 'a devout appearance', 'a secular appearance'),
}
SHARED = (
    'a close-up portrait of a face', 'a neutral facial expression', 'a smiling face',
    'a serious facial expression', 'a face with visible teeth', 'a face with closed lips',
    'a face with glasses', 'a face without glasses', 'a face with facial hair',
    'a clean-shaven face', 'a face with makeup', 'a face without makeup',
    'a face with rounded cheeks', 'a face with angular features', 'a face with a broad jaw',
    'a face with a narrow jaw', 'a face with thick eyebrows', 'a face with thin eyebrows',
    'a brightly illuminated face', 'a face with shadows', 'a face with visible hair',
    'a face with an obscured hairline', 'a face against an outdoor background',
    'a face against an indoor background',
)


def phrase_bank(targets, overrides=None):
    overrides = overrides or {}
    own = {}
    for name in targets:
        if name in overrides:
            own[name] = list(overrides[name])
        else:
            if name not in DESCRIPTORS:
                raise ValueError(f'Add hand-designed phrases for new dimension {name!r} via --phrase-overrides.')
            own[name] = [template.format(d=d) for d in DESCRIPTORS[name]
                         for template in ('a face with {d}', 'a portrait giving the impression of {d}')]
        if not own[name] or any(not isinstance(s, str) or not s.strip() for s in own[name]):
            raise ValueError(f'Invalid phrase list for {name}')
    phrases = list(dict.fromkeys([*SHARED, *race_bank()['phrases'], *[p for t in targets for p in own[t]]]))
    return {'phrases': phrases, 'shared': list(SHARED), 'own': own}


def hierarchy(targets, skill, overrides=None, order='learned'):
    """First two levels parallel; race and complex judgments sequential."""
    groups = [list(VISUAL), list(BASIC), list(RACE), [t for t in targets if t not in (*VISUAL, *BASIC, *RACE)]]
    if overrides is not None:
        groups = overrides
        flat = [t for g in groups for t in g]
        if len(flat) != len(set(flat)) or set(flat) != set(targets):
            raise ValueError('--hierarchy must list each requested target exactly once.')
        return [[targets.index(t) for t in group] for group in groups if group]
    levels = [[targets.index(t) for t in group if t in targets] for group in groups]
    result = []
    for i, level in enumerate(levels):
        if i < 2:
            if level:
                result.append(level)
        else:
            if order == 'learned':
                level.sort(key=lambda t: (-skill[t], targets[t]))
            result.extend([[t] for t in level])
    return result
