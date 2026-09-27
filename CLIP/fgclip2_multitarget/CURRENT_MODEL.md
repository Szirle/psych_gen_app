# Current default: expanded-bank continuous AGOP weighting (B2)

`agop-kernel` now trains **B2**, the expanded-bank continuous AGOP kernel. Every
target receives the same full phrase bank. There are no target-bank exclusions,
hard pruning, local partitions or outer-score fallbacks. The frozen encoder and
neural opt-in variants remain unchanged. Predictions represent perceived ratings,
not verified characteristics of a person.

## Why this version

The completed [factorial experiment](../research/clip_psychometry/target_bank_factorial/REPORT.md)
compared original/expanded banks under pruning and continuous weighting, on all
1004 images with five outer and four inner folds. Relative to the previous
expanded/pruned default, B2 reduced mean relative MSE **2.648%**, improving 21 and
worsening 11 of the 32 priority targets. Trustworthy improved **3.116%**. Original
bank + weighting improved mean MSE 2.069% on the same relative metric. The user
accepted B2's tradeoffs and requested stopping model iteration for now.

Primary success summaries omit looks-like-you and memorable, but models, exports
and per-target reports always include all 34 targets. These are internal
cross-validation results after extensive development on the same cohort, not
independent external confirmation. No claim of all-target nondegradation is made.
The abandoned seven-target policy and its one-off runner were removed. Historical
reports and predictions remain for provenance; old `agop_v1` checkpoints still
mean **pruning**, never this new algorithm.

## Representation and algorithm

1. Original 772 phrases plus 80 fixed additions from
   `clip_psychometry/phrase_candidates.py` form the same 852-phrase bank for every
   target. Global-short and local-box-max FGCLIP2 scores give 1704 coordinates.
   The frozen So400M encoder, pinned revision, p576 cache and precision are recorded.
2. Standardize coordinates on the current fitting subset only. A two-fold pilot
   selects the original centered kernel-ridge readout: linear, degree-two polynomial,
   RBF gamma 0.1/1; eleven alphas from 1e-4 to 10; original two-family blends.
3. Compute the pilot's gradients in standardized coordinates, summing ensemble
   component gradients before the outer product. For each target t, G_t is the
   matrix of gradients, A_t = G_t' G_t / n. This is uncentered AGOP, not gradient
   covariance or causal importance. It is a single update, not recursive xRFM.
4. Inner CV compares identity, diagonal AGOP with beta 0.5/1, and full AGOP with
   beta 0.25/0.5/1. Normalize AGOP to trace p and shrink toward identity:
   M = (1−beta)I + beta p A/tr(A); diagonal candidates replace A by diag(A).
   Zero gradient energy falls back to identity. Use dot z'Mz/p and its induced
   squared distance in the existing linear/poly/RBF readouts. Never standardize
   again after weighting, because doing so would undo diagonal weighting.
5. Every inner training split refits normalization, pilot and metric. Select metric
   and kernel strategy by pooled inner MSE separately per target; exact ties favor
   earlier options, starting with identity. Refit on **all** outer-training rows,
   then predict the untouched outer test fold. Missing-label patterns are split
   into independent complete-label blocks in the general training interface.
6. Saved readouts retain the training supports, normalization, chosen metric and
   centered-kernel coefficients. Full metrics use normalized gradient factors
   instead of dense 1704×1704 matrices; diagonal metrics need only 1704 weights.
   Supports are shared across targets in each block. All phrases remain available.

Implementation: `clip_psychometry/weighted_kernel.py` owns nested B2 selection and
portable score prediction; `learned_geometry.py` supplies AGOP mathematics;
`fgclip2_multitarget/agop.py` handles mask blocks and checkpoints. The original
`race_perception` owns pilot/kernel-grid selection. Checkpoint format is
`fgclip2_multitarget_agop_weighted_v1`; legacy pruned checkpoint loading is retained.
The removed `--agop-keeps` flag no longer has a role in the default.

## Training and downstream inference

The standard `python -m CLIP.fgclip2_multitarget` entry now fits B2 whenever
`agop-kernel` is selected (the default). Its general-purpose defaults remain five
outer/four inner folds; `--final-fit` selects/refits on all images as before.
Use a fresh output directory rather than resuming an old pruned run.

For the requested deployable ten-fold ensemble, use the dedicated cached-score
[production pipeline](PRODUCTION.md). It defaults to **10 outer × 10 inner folds**,
refits each outer model on 90% of the images and combines the ten outputs with
fixed equal weights. It creates all-image OOF predictions, per-fold and pooled
accuracy reports, and Pearson-versus-mean-rating-ceiling figures. It is prepared,
not run. There is no new weighted final checkpoint until the user runs training.

Equal-weight fusion is an intentional low-variance deployment choice: ordinary
OOF data supply only one unseen member prediction per image, so they cannot
honestly tune ten fold-specific stacking weights. The report evaluates held-out
fold learners, not the full deployed ensemble. Future external evaluation can
measure the ensemble's actual accuracy and domain shift without using training
images as test cases. There are no further model-selection experiments scheduled.

The previous final pruned and original baseline checkpoints are temporarily
retained as historical fallback artifacts. The ten old comparison fold checkpoints
were deleted to reclaim space; their OOF predictions, reports and selection records
remain. The shared score cache is retained because production fitting needs it.
