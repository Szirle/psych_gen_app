# Reproduce the frozen manipulation test

Use `/opt/anaconda3/envs/manip311/bin/python` from the project root. PyTorch/MPS must run outside the Codex sandbox.

```python
from CLIP.fgclip2_legacy_research import (
    encode_trust_testset, analyze_trust_testset, report_trust_testset,
)
output = 'CLIP/research/trustworthiness_hidden_test'
encode_trust_testset(
    '/Users/adamsobieszek/Deixis/fgclip_testset',
    'CLIP/research/trustworthiness/trustworthiness_model.joblib',
    output,
)
analyze_trust_testset(
    output, 'data/dim_to_photo_to_ratings.pkl', shared_baselines=False,
)
report_trust_testset(output)
```

Encoding is cached by frozen-bundle hash and input image records. Analysis and reporting can run independently without reloading So400m or fitting any model.

The current analysis uses exact filenames plus byte-verified duplicate baseline aliases. All 31 available flow baseline images exactly match the corresponding vector baselines. Nineteen additional vector baseline images lack a verified match because their flow baseline files are absent. Their predictions are saved but their ratings are excluded pending dataset-owner confirmation.

Only after the owner confirms that all vector level-0 images share the corresponding `flow_level_0.png` ratings, rerun analysis with `shared_baselines=True`. This permits all 50 baseline rating aliases and reuses the vector baseline prediction for the 19 missing flow baseline files. Every reused image/rating is marked in `rated_predictions.csv`; combined absolute level-0 statistics count each common baseline once. No model changes or new encoding are needed.

Primary metrics use the available rated levels within each identity × method. The report also includes a levels-1–4-only analysis: exactly matched ratings for all 50 identities per method, unaffected by any baseline mapping. Direction classification uses the endpoint and neutral bands 0, .01, .02, .03 and .05; .02 is the declared primary descriptive threshold.

All numerical definitions are implemented in `trust_change_metrics`. `test_results.json` contains full-precision metrics; `trajectory_metrics.csv` contains every trajectory. The report includes every level, both methods, all face trajectories, confusion matrices and paired identity-bootstrap RMSE comparisons. The model bundle is unchanged from the development experiment.
