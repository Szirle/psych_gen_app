# CLIP psychometry: compressed context for the next agent

Our starting framework is **causal hypotheses about visual substrates → measurement of those substrates with phrase sets → nonlinear prediction of mean human impression intensity**. The target is what participants felt when seeing the whole image, not actual personality, intelligence, religion or ancestry. A direct word such as trustworthy can be badly aligned with the rating while a concrete affect cue predicts it strongly. A cue with little marginal association can still matter as a moderator, visibility indicator, or component of a conjunction. Causal reasoning guides design; observational predictive utility does not identify causal effects.

Read [METHODOLOGY.md](METHODOLOGY.md) for derivations, diagnostic boundaries and source links. Read [INSIGHTS.md](../../research/clip_psychometry/INSIGHTS.md) before proposing experiments; add surprising observations there and update the relevant chapter. The user subsequently authorized integration with `fgclip2_multitarget`; preserve opt-in legacy/neural variants.

1. **Specify the estimand.** Preserve slider wording, sampled perceiver population, scale, image conditions and aggregation. Appearance ratings are not verified identity labels. Individual disagreement and SEM of a mean differ.
2. **Keep the two validity layers separate.** A predictive phrase can be a bad detector; a valid detector can be redundant; agreeing prompts can share a confound. `(scores, impressions)` alone cannot distinguish these.
3. **Register functional roles.** Include every positive/negative variant, channel, paraphrase and declared dependent conjunction. Opposite English sentiment does not determine the score sign. Broad cue groups are not necessarily reflective scales.
4. **Test whole alternative sets.** Wording A/B isolates the same construct. Content replacement compares coverage. For “woman” + “square jaw” versus the combined phrase, compare both joint held-out risk and retrieval disagreements. Neither evaluation substitutes for the other.
5. **Use conditional mean-prediction utility.** `U(G|B)=R(B)−R(B+G)`, estimated with paired nested OOF losses. Do not rank complex-target banks by univariate MI. Marginal correlations remain useful descriptive checks for directly annotated cues.
6. **Expose substitution and complementarity.** Group deletion, multiple scientific backgrounds, and four-bank `R(BA)+R(BD)−R(B)−R(BAD)` answer distinct questions. This last quantity is predictive complementarity, not causal synergy or a unique information decomposition.
7. **Control kernel geometry.** More paraphrases reweight distances even without new information. Removal must precede per-image centering. Standardization of sparse gates can amplify rare cues. Compare raw and explicit equal-role mass when relevant; tune both readouts with the same budget.
8. **Validate response scales.** Absence may be unordered. Audit intermediate/low score ranges before gating, ranking or calibration. Train-only quantile thresholds are not presence probabilities. Keep transformations inside folds.
9. **Use meaningful psychometrics.** One-factor omega assumes a reflective model and diagonal errors; inspect held-out covariance residuals. MTMM/context drift require independent anchors and support before stronger claims. High consistency can mean duplicated wrong detectors. Formative cue diversity should not be optimized for alpha.
10. **Use augmentation intentionally.** Declare cue×transform×target relations. Lighting and blur may change judgments. Separate systematic bias, within-view variability and collapse. All views stay with their original image; original-label augmented risk requires a justified preservation assumption.
11. **Treat images as exposure.** Extreme/residual/conjunction cards are hypothesis generators. Once inspected, their images are development data. Cross-validation cannot erase previous agent/human inspection. Use blinded annotation, sampling probabilities, and untouched confirmation images.
12. **Distinguish frozen meanings.** Frozen encoder, fixed learning algorithm, fixed recipe and fixed dual coefficients are different. Bank ablations refit the kernel; a permanently frozen head answers model reliance. Future prompt training should use support/query KRR with gradients through the solve and an independent outer holdout.
13. **Do not import a paper by copying its LR.** PromptSRC and ManiPT differ. The source-protocol capsules in §12 record optimizer, prompt structure and schedule. Text-only FGCLIP2 regression is an adaptation; keep an explicit departure ledger.
14. **Spend computation on discriminating questions.** Prefer existing caches, fixed small comparisons and high-information annotations. Do not run an exhaustive search, obvious synthetic demonstrations or visual verification without need. Preserve unrelated work.

Entry points: [README](../README.md); `evaluation.compare`; `features.Representation`; `Registry.columns(closure=True)`; `cards.extreme_cards/conjunction_card/residual_cards`; `metrics.complementarity`; `psychometrics.factor_audit`; `robustness.augmentation_audit`; `adapters.from_assets/augmentation_scores`; `differentiable.kernel_predict`. All substantive metrics have derivation/interpretation in the textbook; any future metric should be added there before its implementation.

Latest iteration: [four studies](../../research/clip_psychometry/iteration_01/REPORT.md).
The tested role-PCA geometry lost against matched controls; targeted weighting remains open. Do not interpret
minimum percentile as presence AND. Use `retrieval_audit` missing-label bounds for
new retrieval sets; partial selected annotations cannot estimate population recall.
Historical smile additions are a provisional trustworthiness lead. Direct-family
deletion did not answer which wording should replace it; use addition-first,
role-matched alternatives on the actual current bank. Two split seeds on the same 400 rows
are sensitivity analysis, not replication. Opt-in bank indices live in
`iteration_01/candidate_banks.json` and refer specifically to `expanded.npz`.

Iteration 02 corrects the baseline to the full current bank and original global/local
features. Prioritize engineered accuracy gains and complementary combinations.
An unsuccessful phrase probe is not evidence that an entire missing-cue hypothesis
is false. Broad exploratory search is authorized; distinguish its selected gains
from later confirmation. See `iteration_02/PROTOCOL.md`.

Latest results: [iteration 02](../../research/clip_psychometry/iteration_02/REPORT.md).
The actual current-bank experiment found productive wording additions and contrast
amplification; a moderate energy-normalized combination improved all five tested
targets in both fold assignments. Strong target-specific emphasis has cross-target
costs. `checkpoint_candidate.json` specifies the proposal; `phrase_overrides.json`
adds 32 captions without removing old ones, but does not itself enable contrasts.
Standalone detector alignment and full-bank incremental utility disagreed in both
directions: never use the former as a hard filter. The new cache contains p576
original global/local features for the 400 development images, not all 1,004.

Latest: [iteration 03](../../research/clip_psychometry/iteration_03/REPORT.md) evaluates
AGOP/RFM-inspired per-target pruning across all **34** targets, with nested mask and
retention selection. It improves macro error and reduces per-target dimensions,
but no nontrivial tested policy meets strict all-target nondegradation; defaults
were not changed. Important distinctions: actual-readout vs surrogate attribution,
stable opt-in bank extension vs global reranking, and per-target sparsity vs union
encoder cost. `importance.json` in each run covers every phrase/target and selection
stability; `fit_phrase_pruner` exports an explicit target mask for a validated keep
setting. Keep the 604 evaluation images excluded until an intentional confirmation
protocol is chosen. Do not cherry-pick outer-label fallbacks and call them validated.


Current default: user adopted expanded-bank continuous AGOP weighting **B2**.
See [CURRENT_MODEL.md](../../fgclip2_multitarget/CURRENT_MODEL.md) and
[PRODUCTION.md](../../fgclip2_multitarget/PRODUCTION.md). The full 2×2 study found
B2 −2.648% mean relative MSE versus the old expanded/pruned default, 21/32
improved and 11 worsened; trustworthy −3.116%. Primary summaries exclude
looks-like-you and memorable, but all 34 outputs are retained. No bank exclusions
or hard pruning remain in the default. Do not resume model iteration unasked.

The user will run the prepared 10 outer × 10 inner production pipeline. It refits
each fold on all 90% outer-training rows, retains ten models, and averages their
predictions equally for new images. The OOF report evaluates only each image's
held-out model, not the ten-member ensemble. Noise-ceiling figures use the square
root of mean-rating reliability from per-image sample variance/rater count, not
split-half reliability. Old reports/OOF predictions and full score cache remain;
obsolete fold checkpoint files and one-off exclusion/geometry runners are removed.
