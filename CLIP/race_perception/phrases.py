
"""Fixed, label-independent expansion for subjective ratings of generated faces.

Captions are candidate encoder directions, never definitions of ancestry. The
bank and candidate grid must be frozen before inspecting the new OOF results.
Earlier Asian experiments already used this dataset: this is an internal
validation experiment, not a pristine external test.
"""

TARGETS = ("asian", "middle-eastern", "hispanic", "islander", "native",
           "black", "white", "skin-color")
LABELS = ("asian", "middle eastern", "hispanic", "pacific islander",
          "native american", "black", "white")
REFERENCES = tuple(f"a {label} person's face" if label != "asian"
                   else "an asian person's face" for label in LABELS) + (
                       "a face with dark skin",)

SOURCES = [
    {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3399873/",
     "title": "Skin and Bones: The Contribution of Skin Tone and Facial Structure to Racial Prototypicality Ratings",
     "finding": "Contributions of skin tone and facial metrics to typicality ratings varied with the perceiver group and face group.",
     "implication": "Model skin appearance and facial geometry as separate candidate directions; do not prescribe their coefficients."},
    {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7728794/",
     "title": "Shining a Light on Race: Contrast and Assimilation Effects in the Perception of Skin Tone and Racial Typicality",
     "finding": "Surrounding faces produced different effects on perceived skin lightness and racial typicality in this experiment.",
     "implication": "Keep skin-color as its own continuous target and include illumination/context phrases."},
]


def build_phrase_bank():
    groups = {"references": list(REFERENCES)}
    templates = (
        "the face of a person who looks {label}",
        "a person with a {label} appearance",
        "a face perceived as {label}",
        "a face that looks strongly {label}",
        "a face that looks slightly {label}",
        "a face that looks somewhat {label}",
        "a face with a typically {label} appearance",
        "a portrait of a {label} person",
        "a close-up of a {label} person's face",
        "a {label} person's facial appearance",
        "a person who could be perceived as {label}",
        "a face with an ambiguous {label} appearance",
        "a young adult who looks {label}",
        "an older adult who looks {label}",
        "a man who looks {label}",
        "a woman who looks {label}",
        "a smiling person who looks {label}",
        "a neutral face that looks {label}",
        "a {label} appearance with light skin",
        "a {label} appearance with medium skin tone",
        "a {label} appearance with dark skin",
        "a {label} appearance with straight hair",
        "a {label} appearance with curly hair",
        "a {label} appearance with dark eyes",
    )
    for target, label in zip(TARGETS, LABELS):
        groups[f"perceived_{target}"] = [t.format(label=label) for t in templates]
    # These category wordings are additional text bases, not extra targets or
    # claims that a regional appearance is equivalent to the rated dimension.
    regional = ("east asian", "southeast asian", "south asian", "central asian",
                "chinese", "japanese", "korean", "vietnamese", "filipino", "indian",
                "arab", "persian", "north african", "mediterranean", "latino",
                "latina", "latin american", "mexican", "central american",
                "south american", "caribbean", "polynesian", "melanesian",
                "micronesian", "native hawaiian", "samoan", "indigenous american",
                "american indian", "alaska native", "african", "african american",
                "european", "northern european", "southern european", "multiracial")
    groups["regional_wordings"] = [f"a person with a perceived {s} appearance" for s in regional]
    groups["ambiguous_appearance"] = [
        "a face with a racially ambiguous appearance", "a face with mixed features",
        "a face whose racial appearance is difficult to judge", "a face with an ethnically ambiguous appearance",
        "a person with a mixed asian and white appearance", "a person with a mixed black and white appearance",
        "a person with a mixed asian and black appearance", "a person with a mixed native american and white appearance",
        "a person with a mixed hispanic and white appearance", "a person with a mixed asian and pacific islander appearance",
        "a person with a mixed middle eastern and white appearance", "a person with a mixed hispanic and native american appearance",
    ]
    tones = ("extremely pale", "very pale", "pale", "fair", "very light", "light",
             "light beige", "beige", "medium beige", "cream-colored", "ivory-colored",
             "pinkish", "rosy", "reddish", "warm", "cool", "neutral-toned", "golden",
             "yellowish", "olive", "light olive", "dark olive", "tan", "light tan",
             "medium tan", "deep tan", "bronze", "copper-toned", "light brown",
             "medium brown", "brown", "deep brown", "dark brown", "very dark brown",
             "dark", "very dark", "unevenly colored", "evenly colored", "freckled", "flushed")
    groups["skin_tone"] = [f"a face with {s} skin" for s in tones]
    groups["skin_lightness_variants"] = [f"a portrait showing a {s} complexion" for s in tones[:36]]
    cues = {
        "eyes_eyelids": ("narrow eyes", "wide open eyes", "almond-shaped eyes", "round eyes",
            "large eyes", "small eyes", "deep-set eyes", "prominent eyes", "widely spaced eyes",
            "closely spaced eyes", "a visible fold at the inner corners of the eyes",
            "upper eyelids without a visible crease", "a visible upper eyelid crease", "hooded eyelids",
            "heavy upper eyelids", "upturned outer eye corners", "downturned outer eye corners",
            "dark brown eyes", "light brown eyes", "black eyes", "blue eyes", "green eyes",
            "hazel eyes", "gray eyes"),
        "nose": ("a broad nose", "a narrow nose", "a wide nose base", "a narrow nose base",
            "a low nose bridge", "a high nose bridge", "a flat nose bridge", "a prominent nose bridge",
            "a straight nose", "a curved nose", "a rounded nose tip", "a pointed nose tip",
            "a small nose", "a large nose", "a long nose", "a short nose", "flared nostrils",
            "small nostrils", "a broad rounded nose tip", "an upturned nose"),
        "mouth_lips": ("full lips", "thin lips", "a full lower lip", "a thin upper lip",
            "a full upper lip", "a wide mouth", "a small mouth", "a pronounced cupid's bow",
            "a flat upper lip line", "a long philtrum", "a short philtrum", "darkly pigmented lips",
            "pink lips", "lips close in color to the skin", "a closed mouth", "an open mouth"),
        "face_geometry": ("high cheekbones", "broad cheekbones", "prominent cheekbones", "flat cheeks",
            "rounded cheeks", "hollow cheeks", "a broad face", "a narrow face", "a long face",
            "a short face", "a round face shape", "an oval face shape", "a square face shape",
            "a heart-shaped face", "a broad jaw", "a narrow jaw", "an angular jaw", "a rounded jaw",
            "a pointed chin", "a broad chin", "a receding chin", "a prominent chin", "a high forehead",
            "a low forehead", "a broad forehead", "a narrow forehead", "a flat facial profile",
            "a protruding facial profile"),
        "hair_brows": ("straight black hair", "wavy black hair", "curly black hair", "tightly curled hair",
            "coiled hair", "straight brown hair", "wavy brown hair", "curly brown hair", "blond hair",
            "red hair", "gray hair", "white hair", "very short hair", "long hair", "a shaved head",
            "a receding hairline", "thick eyebrows", "thin eyebrows", "straight eyebrows", "arched eyebrows",
            "dark eyebrows", "light eyebrows", "a thick beard", "a sparse beard", "a mustache",
            "no visible facial hair", "dark facial hair", "gray facial hair"),
        "illumination_texture": ("brightly illuminated skin", "skin in shadow", "softly lit skin",
            "harshly lit skin", "warm light on the skin", "cool light on the skin", "natural daylight",
            "uneven facial lighting", "strong highlights on the forehead", "shadows around the eyes",
            "deep shadows beside the nose", "low contrast facial features", "high contrast facial features",
            "smooth skin", "rough skin", "shiny skin", "matte skin", "wrinkled skin", "visible skin pores",
            "freckles across the cheeks", "reddened cheeks", "dark circles under the eyes", "facial makeup",
            "no visible makeup", "a bright background", "a dark background", "a blurred background",
            "a neutral background", "glasses", "sunglasses", "partly covered eyes", "visible ears"),
        "age_expression_pose": ("a youthful appearance", "an older appearance", "a middle-aged appearance",
            "a childlike facial shape", "a masculine appearance", "a feminine appearance",
            "an androgynous appearance", "a neutral expression", "a broad smile", "a subtle smile",
            "visible teeth", "a serious expression", "a frown", "raised cheeks", "a direct gaze",
            "a sideways gaze", "a front-facing pose", "a three-quarter pose", "a tilted head",
            "a slightly raised chin", "a slightly lowered chin", "a relaxed expression"),
    }
    for group, descriptions in cues.items():
        groups[group] = [f"a face with {s}" for s in descriptions]
    groups["skin_geometry_combinations"] = [
        f"a face with {tone} skin and {cue}"
        for tone in ("pale", "light brown", "olive", "medium brown", "dark brown", "very dark")
        for cue in ("a broad nose", "a narrow nose", "full lips", "thin lips", "high cheekbones",
                    "a round outline", "almond-shaped eyes", "a visible upper eyelid crease")
    ]
    phrases, indices = [], {}
    for group, texts in groups.items():
        indices[group] = []
        for phrase in texts:
            if phrase not in phrases:
                phrases.append(phrase)
            indices[group].append(phrases.index(phrase))
    return {"version": 1, "targets": list(TARGETS), "phrases": phrases,
            "groups": indices, "reference_indices": [phrases.index(p) for p in REFERENCES],
            "sources": SOURCES,
            "provenance": "Wording frozen before new target-level CV. Direct prompts inherit the historical method; morphology/illumination families are literature-motivated hypotheses, not validated diagnostic criteria.",
            "semantics": "Eight independent continuous mean participant ratings of StyleGAN2-generated faces; no ancestry labels, probabilities, forced coefficient signs, or sum-to-one constraint."}
