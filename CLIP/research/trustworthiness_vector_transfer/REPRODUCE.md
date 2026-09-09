# Vector-to-flow transfer experiment

The original encoder and 256-phrase bank remain frozen. Heads train on 804 original faces plus 200 exactly rated vector images, levels 1–4. The test sets are 200 excluded original faces and all 231 supplied flow images. The methods share identities; no flow labels or baseline rating aliases enter training.

Run from the project root with `/opt/anaconda3/envs/manip311/bin/python` outside the sandbox because the module imports PyTorch. All image scores are cached, so no new encoder pass is required.

To reproduce in a **new** output directory:

```python
from CLIP.fgclip2_face_impressions import (
    run_trust_transfer, freeze_trust_transfer_tradeoffs,
    evaluate_trust_transfer, report_trust_transfer,
)
original = 'CLIP/research/trustworthiness'
manipulations = 'CLIP/research/trustworthiness_hidden_test'
output = 'CLIP/research/trustworthiness_vector_transfer_reproduction'

run_trust_transfer(original, manipulations, output, evaluate=False)
# The strict 15% source-MSE guard rejects every adapted head.
# This validation-only step adds the balanced and vector-focused comparisons;
# it refuses to run after test_results.json exists.
freeze_trust_transfer_tradeoffs(original, manipulations, output)
evaluate_trust_transfer(original, manipulations, output)
report_trust_transfer(output)
```

The main model minimizes balanced grouped-validation NMSE: .50 original absolute + .35 vector absolute + .15 vector centered. The recipe grid has 157 candidates. All preprocessing, source-only models and Nyström features are refit inside each fold. The 50 vector identities are kept intact across folds.

```python
from CLIP.fgclip2_face_impressions import TrustworthinessPredictor
model = TrustworthinessPredictor.load(
    'CLIP/research/trustworthiness_vector_transfer/transfer_model.joblib'
)
means = model.predict_images(['/path/to/a.png', '/path/to/b.png'])
change = means[1] - means[0]
```

Other deployable comparators are `pooled_model.joblib` (mixed-data RBF) and `contrast_model.joblib` (change-aware correction). They were selected within their families on validation data. The mixed-data comparator happened to outperform the balanced selection on both current tests; using this observation to choose a new winner requires another independent test. `comparison_models.joblib` stores all fitted comparisons.

Predictions need no method, identity, level, domain flag or human rating. They remain unclipped in the original slider units. The MPS encoder configuration and scikit-learn 1.8.0 contract are inherited from the original model; use trusted local joblib artifacts only.

`original_test_predictions.csv` and `flow_test_predictions.csv` contain all held-out outputs. `validation_predictions.npz`, `validation.json`, `protocol.json` and source snapshots preserve selection, exact splits and grouping. `test_results.json` contains full-precision absolute, level-specific, centered, slope, direction and paired-bootstrap results. The HTML report embeds five figures; PNG/SVG versions are also saved.

The test sets were inspected in prior research, and flow/vector use the same underlying identities. This is an exploratory method-transfer experiment with direct train/test separation in the current fit, not a blind external validation or a test on new identities. Original-face retention must be considered alongside improvements on flow.
