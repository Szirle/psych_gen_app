# FG-CLIP2 face-rating regression: literature review and experiment proposals

Research cutoff: **19 September 2026**. This is a research proposal, not a claim that a new model has already improved the results. No training or tests were run.

**Recommendation.** Keep FG-CLIP2 So400m and the current kernel model as the reference. First test its complete visual embeddings, then train a small visual adapter and rating-conditioned pooling, then add late-layer visual LoRA. Evaluate learned text prototypes and ordinal/distribution losses as controlled additions. The strongest candidate is a model that can use visual information outside the fixed phrase span while retaining useful pretrained geometry.

**Scope and evidence.** I inspected the active encoder, shared evaluation code, race-perception implementation, historical phrase experiments, relevant numerical tests, the exact saved `race_perception/results.json`, and the pinned checkpoint's model/configuration source. Literature search covered FG-CLIP2, CLIP adaptation, ordinal regression, subjective image quality/aesthetic assessment, and relevant 2026 publications. Primary papers and official author repositories support the recommendations. Classification, age, and image-quality results are transfer evidence; they are not performance estimates for these subjective generated-face judgments. There is no verified published benchmark establishing the best FG-CLIP2 adaptation for this exact dataset.

**1. What the current implementation actually does.**

The current path is frozen FG-CLIP2 → normalized global image and text features → phrase agreement logits → supervised kernel regression. The target is a mean human impression of a generated face, not an objective personal attribute. The eight output dimensions remain independent continuous scores.


| Property                            | Verified implementation or saved result                                                       |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| Images                              | 1,004                                                                                         |
| Backbone                            | `qihoo360/fg-clip2-so400m`                                                                    |
| Pinned revision                     | `d57d30fe94a107dd6a2610eb4e9a135004d823a4`                                                    |
| Visual feature width                | 1,152                                                                                         |
| Vision/text transformer depth       | 27 layers each                                                                                |
| Phrase features                     | 516 phrases × patch budgets 128 and 256 = 1,032 features                                      |
| Score                               | `exp(logit_scale) * cosine(image, text) + logit_bias`                                         |
| Readout                             | Linear, quadratic, and RBF kernels; target-specific regularization and two-component blending |
| Validation                          | 50 outer folds, 10 inner folds; 983–984 outer-training faces                                  |
| Selected pooled RMSE                | **0.0697065**                                                                                 |
| Phrase-ridge pooled RMSE            | 0.0738303                                                                                     |
| Single-reference-phrase pooled RMSE | 0.1602313                                                                                     |


“Pooled RMSE” here is the square root of MSE averaged over faces and targets, not the arithmetic mean of the eight RMSEs. Patch budgets are token counts, not pixel dimensions. The selected baseline is already substantially more expressive than a linear phrase readout.


| Subjective target | Selected RMSE | R²     |
| ----------------- | ------------- | ------ |
| asian             | 0.07397       | 0.9304 |
| middle-eastern    | 0.07345       | 0.8738 |
| hispanic          | 0.08046       | 0.8528 |
| islander          | 0.06614       | 0.9046 |
| native            | 0.07111       | 0.8591 |
| black             | 0.05063       | 0.9257 |
| white             | 0.07975       | 0.9312 |
| skin-color        | 0.05635       | 0.8770 |


Sources: [saved results](/Users/adamsobieszek/PycharmProjects/psych_gen_app/CLIP/research/race_perception/results.json), [kernel implementation](/Users/adamsobieszek/PycharmProjects/psych_gen_app/CLIP/race_perception/__init__.py), [training/evaluation CLI](/Users/adamsobieszek/PycharmProjects/psych_gen_app/CLIP/race_perception/__main__.py), [shared infrastructure](/Users/adamsobieszek/PycharmProjects/psych_gen_app/CLIP/fgclip2_face_impressions.py), [checkpoint configuration](/Users/adamsobieszek/PycharmProjects/psych_gen_app/models/fgclip2/models--qihoo360--fg-clip2-so400m/snapshots/d57d30fe94a107dd6a2610eb4e9a135004d823a4/config.json).

**2. The main opportunity is a restricted representation.**

For one patch budget, let `z` be the normalized 1,152-dimensional image vector and let `T` contain the 516 normalized phrase vectors as columns:

```text
s = a Tᵀz + b
prediction = kernel_readout(s)
```

The linear map `Tᵀ` has rank at most 516. Its nullspace therefore has dimension at least 636. The downstream kernel cannot distinguish two image representations with the same phrase projection. Two patch budgets provide two visual views, but still only project each view into the phrase bank. This is an algebraic property, not a measurement of how much predictive signal is lost: the data could lie on a sufficiently informative low-dimensional manifold.

This distinction suggests a clean first experiment: give a matched readout all visual coordinates. If it helps, the fixed phrase representation omitted useful signal. If it does not, the phrase bank may be a valuable regularizer, and local features or encoder adaptation become more interesting.

A second algebraic point prevents misleading experiments: **freely tuning output phrase embeddings and then using a linear readout is still linear regression on the frozen visual vector.** If `prediction = wᵀTᵀz`, then it equals `(Tw)ᵀz`. Text anchoring, nonlinear normalization of adapted image vectors, ordered prototypes, or nonlinear readouts can change the inductive bias or function class; merely naming free vectors “phrases” does not do so. Likewise, tuning only the common logit scale/bias adds little when training-fold standardization and an intercept already absorb them.

**3. What FG-CLIP2 itself supports.**

FG-CLIP2 builds on SigLIP2, combining global alignment with region supervision, fine-grained negative descriptions, textual intra-modal contrast (TIC), and cross-modal ranking (CMR). It uses attention pooling and a separate dense-feature path. Its evaluation demonstrates fine-grained alignment and downstream transfer, not subjective face-rating regression. TIC separates nearby descriptions; CMR separates a matched description from a negative one. Neither directly trains distances between human rating means. [FG-CLIP2, ICML 2026, revised paper](https://arxiv.org/html/2510.10921v3).

My adaptation proposal is to preserve useful pretrained structure while introducing supervision for the actual rating target. Do not reproduce the full pretraining recipe on 1,004 faces. In particular, indiscriminately repelling phrase paraphrases could damage the smooth semantic structure needed for continuous ratings. The stronger opportunity specific to this code is to exploit its existing global, dense, and region APIs.

**4. Literature most relevant to implementation.**

The table separates direct regression evidence from more indirect adaptation evidence. Conference years reflect the publication, even when an earlier preprint exists. “Transfer” means the proposed FG-CLIP2 use still requires an experiment.


| Work                                                                                                                                                                                        | Relevant result or mechanism                                                                                                      | What to transfer here                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [OrdinalCLIP, NeurIPS 2022](https://arxiv.org/abs/2206.02338)                                                                                                                               | Learns context and continuous rank embeddings, producing ordered language prototypes.                                             | Train a small ordered prototype bank for each rating dimension.                                                                  |
| [L2RCLIP, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2a11632520f4b7473d7838f074a7d25-Abstract.html)                                                          | RankFormer and cross-modal ordinal pairwise supervision strengthen ordering alignment.                                            | Couple rating prototypes and visual features using rating-distance supervision.                                                  |
| [NumCLIP, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11339.pdf)                                                                                                    | Uses linguistic concepts for coarse bins, ordinal regularization, and a fine regression stage.                                    | Initialize with verbal levels, then retain continuous regression rather than rounding ratings.                                   |
| [Ranking-aware adapter, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c86d3f43371270c70ed94e024852895d-Abstract-Conference.html)                                       | Learns text-guided image ordering with a lightweight adapter and an auxiliary comparison branch.                                  | Learn from between-face differences; retain a calibrated single-face predictor.                                                  |
| [ConOrd, ICML 2026](https://arxiv.org/html/2607.08109v1)                                                                                                                                    | Soft affinity/disparity weights encode rank distances in contrastive learning; age and IQA experiments use CLIP ViT-B.            | Add target-specific continuous representation supervision to adapters or LoRA.                                                   |
| [SOL, ICML 2026](https://arxiv.org/html/2607.08103v1)                                                                                                                                       | Models uncertain ranks with probabilistic ordering rather than treating each annotation as exact.                                 | Incorporate uncertainty in mean-rating comparisons. Do not automatically relabel subjective disagreement as corruption.          |
| [CLIP-IQA/CLIP-IQA+, AAAI 2023](https://arxiv.org/abs/2207.12396)                                                                                                                           | Applies language priors to perceptual scoring; the learned-prompt extension adapts text context.                                  | A simple supervised prompt-tuning reference, not the presumed final winner.                                                      |
| [LIQE, CVPR 2023](https://arxiv.org/abs/2303.14968)                                                                                                                                         | Uses vision-language correspondence and multitask learning for quality prediction.                                                | Shared visual adaptation with separate target heads and auxiliary human-rated dimensions.                                        |
| [QualiCLIP](https://arxiv.org/html/2403.11176v3)                                                                                                                                            | Learns quality-aware image–text alignment from synthetic degradation order without human MOS.                                     | Task-specific pretraining can change CLIP's representation; degradation ordering itself is unsuitable for most face impressions. |
| [GRMP-IQA, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.pdf)              | Few-shot IQA combines meta-prompt initialization and gradient regularization; naïve prompt tuning is sensitive to initialization. | Use initialized, regularized prompts and multiple seeds. Its distortion meta-tasks cannot simply be reused as face-rating tasks. |
| [PPA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.html)         | Models scoring as a mixture over prompts, conditioned on image and task.                                                          | Small image-dependent mixtures of learned rating experts; lower priority with 1,004 faces.                                       |
| [CoOp](https://arxiv.org/abs/2109.01134)                                                                                                                                                    | Optimizes continuous context tokens while freezing pretrained encoders.                                                           | Shared learned text context is more constrained than independently moving every phrase vector.                                   |
| [MaPLe, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.html)                                                      | Couples visual and language prompts at multiple depths.                                                                           | Joint adaptation if one-sided adaptation is insufficient; requires custom porting to this backbone.                              |
| [CLIP-LoRA, CVPR workshop 2024](https://arxiv.org/abs/2405.18541)                                                                                                                           | Low-rank adaptation is a strong few-shot classification baseline across 11 datasets.                                              | Adapt a restricted set of late visual attention projections before full fine-tuning.                                             |
| [ProLIP, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Fahes_CLIPs_Visual_Embedding_Projector_is_a_Few-shot_Cornucopia_WACV_2026_paper.pdf)                              | Regularizes projector updates toward pretrained weights; also develops a regularized linear adapter.                              | Small anchored visual transformations; native pooling-head adaptation as a separate experiment.                                  |
| [PromptSRC, ICCV 2023](https://github.com/muzairkhattak/PromptSRC)                                                                                                                          | Regularizes prompting against frozen-model representations and uses prompt self-ensembling.                                       | A frozen FG-CLIP2 reference can restrain overfitting.                                                                            |
| [MPS-Tuning, ICLR 2026](https://openreview.net/pdf?id=ZGJJF1e2u0)                                                                                                                           | Preserves global and local feature geometry while adapting a classification model.                                                | Test a Gram-matrix preservation penalty; replace categorical sculpting with rating-aware supervision.                            |
| [Prompt-OT, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Chen_Prompt-OT_An_Optimal_Transport_Regularization_Paradigm_for_Knowledge_Preservation_in_WACV_2026_paper.pdf) | Uses optimal transport regularization for knowledge preservation during VLM adaptation.                                           | A later alternative to simpler feature/Gram penalties, not an initial complexity requirement.                                    |
| [LP++, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_LP_A_Surprisingly_Strong_Linear_Probe_for_Few-Shot_CLIP_CVPR_2024_paper.html)                                   | Demonstrates the importance of strong linear baselines combining text and image knowledge.                                        | Require a tuned direct-feature baseline before crediting complex adaptation.                                                     |
| [LP-FT, ICLR 2022](https://par.nsf.gov/servlets/purl/10472125)                                                                                                                              | Linear-head training before backbone updates reduces feature distortion in transfer experiments.                                  | Warm up the regression head before unfreezing visual parameters.                                                                 |
| [WiSE-FT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)                                              | Interpolates pretrained and adapted model weights to improve robustness.                                                          | Optional adapter shrinkage, with the regression readout fitted to the interpolated representation.                               |
| [Rank-N-Contrast, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)                                               | Learns regression representations by comparing target-space distances.                                                            | A simpler historical comparison for ConOrd-style supervision.                                                                    |
| [ConR, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/704357127afabbd5a6eb979a57810767-Abstract-Conference.html)                                                        | Regularizes representation geometry for imbalanced continuous labels.                                                             | An optional tail-error experiment if rare rating ranges drive failures.                                                          |
| [Q-Align, ICML 2024](https://q-align.github.io/) and [DeQA-Score, CVPR 2025](https://arxiv.org/abs/2501.11561)                                                                              | Text-defined scoring levels; DeQA uses soft score distributions and pairwise relationships. These are MLLM methods.               | Borrow distribution supervision, without replacing FG-CLIP2 with a language generator.                                           |




Access qualification: the PPA proceedings abstract was available, but its full PDF repeatedly failed to load. Its entry supports a conceptual candidate, not a reproduced recipe or a numerical comparison. MPS-Tuning's accessible arXiv method was read alongside the indexed ICLR 2026 paper. ConOrd/SOL conference status is also supported by the [authors' ICML announcement](https://mcl.korea.ac.kr/icml2026-order-learning-papers/). None of the cited improvements is used as a predicted percentage improvement on this dataset.

**5. Proposal A: expose the full visual representation and preserve the useful text prior.**

This is the highest-value first experiment, although the backbone remains frozen. It answers whether fine-tuning is necessary before introducing training cost.

Use `FaceImpressionPipeline.features(128/256)` to obtain full embeddings. Compare p256 alone, concatenated p128+p256, and a combination with the original phrase kernel. Run ridge and the existing kernel families with matched tuning budgets. For normalized embeddings, include the native cosine/dot-product geometry; independently standardizing every coordinate is a separate choice, not automatically optimal.

An especially informative variant splits each `z` into `z_parallel`, its orthogonal projection onto the text-vector span, and `z_perp`, the residual. A label-independent SVD of the frozen phrase matrix supplies this basis. Retain the current phrase model and add a separately regularized predictor using `z_perp`. Alternatively, combine normalized phrase and full-image kernels with a training-selected mixing weight.

This asks whether information inaccessible to the phrase projection improves prediction. An unrestricted visual ridge model is also essential: factorizing its coefficients through trainable phrase vectors must beat this baseline to justify its extra machinery.

For text anchoring, initialize a target vector from existing positive/negative phrase embeddings, then learn a penalized correction with an independently fitted slope/intercept. Do not assume verbal similarity already has the slider's calibration. Shrinkage strength is selected inside the training data.

**Failure condition:** direct features add variance but no predictive signal. In that case, retain the phrase prior and move to local pooling, rather than escalating the size of a global MLP.

**6. Proposal B: regularized visual adaptation on cached features.**

Start from a near-identity transformation of the full image embedding:

```text
z' = normalize(z + U Vᵀ z)               rank 4 or 8
or
z' = normalize(z + U GELU(Vᵀ z))        bottleneck 16 or 32
y_hat = W z' + c
```

Initialize one residual factor to zero, the other randomly; initializing both to zero prevents learning. Penalize the effective update or feature deviation and fit a direct eight-output head. The low-rank rank-8 linear residual has 18,432 parameters at width 1,152, plus the head. A normalized adapter or nonlinear bottleneck can change the predictor beyond linear ridge; an unnormalized linear adapter followed by a linear head cannot, so retain that as a regularization control rather than claiming new expressiveness.

This is inspired by regularized adaptation in ProLIP, but it is our regression design. The native FG-CLIP2 global output comes from an attention-pooling head with a residual MLP; there is no justification for blindly copying code that expects OpenAI CLIP's `visual.proj`.

Train first with ordinary mean MSE. Only then add an ordinal or distribution objective. This keeps any gain attributable to representation adaptation separate from the effect of a new loss. Compute and memory are low because the encoder outputs can be cached and the large text encoder is unnecessary during these updates.

**7. Proposal C: learn text directions, with three distinct levels of constraint.**

**C1 — anchored output embeddings.** Initialize from the current phrase matrix `T0` and learn

```text
T = row_normalize(T0 + A Bᵀ)
```

with rank 4 or 8, a penalty toward `T0`, and a modest nonlinear or ordinal scoring head. For 516 vectors of width 1,152, rank 8 needs 13,344 parameters instead of 594,432 unconstrained coordinates. The language encoder is not needed during training. This genuinely changes the measured directions, but the resulting vectors are latent predictors: their original wording ceases to be an exact interpretation if they move substantially.

**C2 — shared soft context.** Learn 4–8 context-token vectors around fixed task/level wording and pass them through the frozen text transformer. Share most context across targets initially. At width 1,152, one shared context has 4,608–9,216 parameters. This preserves the pretrained text encoder as a constraint, but backpropagation through it is computationally substantial despite the small parameter count.

**C3 — joint visual and text adaptation.** Combine the successful prototype/context method with late visual LoRA. Text-tower LoRA or MaPLe-style visual prompts come later, because adapting both large towers simultaneously makes small-data attribution and regularization harder.

Use an ordered level bank per target, initially five levels spanning the slider. Initialize those levels with verbal descriptions reflecting degree of the *rating*, not exact numeric strings. A NumCLIP/OrdinalCLIP-inspired predictor is:

```text
p_itk = softmax_k(cosine(z'_i, t_tk) / temperature_t)
mean_it = sum_k(p_itk * anchor_k)
```

The softmax is over levels **within one target**, never across the eight targets. Compare five and nine levels only if data support the larger bank. Add either a small continuous correction or a separate scalar head so the model is not forced into coarse rounded predictions. Ordered anchors and smooth prototype parameterization are useful priors, but their benefit must be measured against a direct head with the same visual adaptation.

**Implementation detail:** the checkpoint's text embeddings accept `inputs_embeds`, but its enclosing text transformer requires `input_ids` and does not expose a ready-made soft-prompt path. Add a small controlled wrapper around the embedding/encoder path. Preserve its noncausal attention, positional handling, fixed sequence length, final-position pooling, and `walk_type` behavior. OpenAI CLIP's EOS-selection and causal-mask prompt code is not directly compatible.

**8. Proposal D: learn what facial regions to pool.**

The current global scores discard spatial structure before supervision. Use the existing dense and RoI feature APIs to compare three levels:

1. Frozen dense features, pooled over a small fixed set of consistently aligned face regions, plus the global vector.
2. One learned, masked attention query per target over dense patch features, with small shared key/value projections and a scalar head.
3. The winning pooling scheme with late visual LoRA.

For aligned generated faces, a fixed grid is an inexpensive reference; landmark regions are useful only if reliable alignment is available. Do not remove the full image, hair, or surrounding context by assumption, because they may contribute to the measured impression. Compare global-only with global-plus-local predictions.

Start with patch budget 256; assess 576 only for the winning local model. Padding must be masked, and image-region coordinates must follow the official processor's spatial layout. For region-text alignment use the `box` text path; for direct local regression there is no need to score region phrases at all. Learned queries are pooling parameters, not proven causal explanations.

This is a strong candidate for gains when different targets depend on different local cues. The main risks are unreliable region alignment and overfitting the spatial layout of one generator. Cache dense features when the backbone is frozen; do not cache them across a subsequent encoder update.

**9. Proposal E: late visual LoRA, then limited unfreezing.**

This is the principal encoder fine-tuning proposal. Keep the pinned So400m checkpoint as initialization. Train the direct head first. Freeze the text tower and most of the vision tower; add LoRA to `q_proj` and `v_proj` in the last four visual transformer blocks (indices 23–26). Compare ranks 4 and 8. Only broaden to eight blocks if the initial experiment earns it.

At width 1,152, two square projections in four blocks require 73,728 LoRA parameters at rank 4 or 147,456 at rank 8, excluding biases and the output head. These counts follow from the inspected architecture; they are not runtime measurements. Explicitly restrict module paths to `vision_model.encoder.layers.*.self_attn.*` so suffix matching does not unintentionally adapt the text tower.

Proposed starting settings, not literature-established optima for this dataset:


| Item               | Initial choice                                                          |
| ------------------ | ----------------------------------------------------------------------- |
| Input              | p256; p128 as a cheap ablation                                          |
| Optimizer          | AdamW                                                                   |
| LoRA learning rate | 1e-5 or 5e-5                                                            |
| Head learning rate | 1e-4 or 5e-4                                                            |
| Training           | Short warm-up; maximum 30 epochs; inner-validation early stopping       |
| Batch              | 8–16 images initially, subject to actual memory                         |
| Regularization     | Weight decay, plus one selected pretrained-feature preservation penalty |
| Seeds              | Three for finalists                                                     |
| Primary loss       | Mean MSE in original rating units                                       |


Do not cross every setting into a huge grid. Start with a few predeclared recipes. For the final fit, choose duration from inner training runs, without inspecting outer labels.

Compare three preservation settings: none; pointwise cosine feature matching to frozen FG-CLIP2; and a batch Gram penalty `mean((Z'Z'ᵀ − Z0Z0ᵀ)^2)`. The latter is an MPS-Tuning-inspired simplification, not a reproduction of its entire method. Geometry preservation alone allows rotations; pair it with text anchoring if fixed text alignment must remain meaningful. Excessive preservation can also prevent the desired adaptation.

After LoRA, consider training only the native pooling probe, then selected pooling-head parameters, then the final one or two transformer blocks with roughly 1e-6–5e-6 learning rates. Full vision fine-tuning is a later comparison. Full text-tower fine-tuning is especially unattractive initially because the vocabulary matrix is large and ratings supply little linguistic diversity.

A text-free direct regressor can unload the text tower after any initialization is computed. Late-only adaptation also allows a frozen visual prefix to run without gradient tracking. Soft visual prompts inserted early do not have that same advantage. LoRA reduces trainable optimizer state, but does not eliminate activation memory in the part of the network traversed by gradients. Use float32 for small heads/prototypes and a device-supported stable precision for backbone computation; the current fp16 inference setting is not itself a validated training recipe.

**10. Losses worth testing, separately from architecture.**

**Mean regression remains the anchor.** Begin with equal-weight MSE over the eight dimensions in their original units, matching the reported pooled metric. Huber is an optional robustness comparison, but high disagreement is not automatically a bad label. If target normalization is used for optimization, restore the intended original-unit weighting.

**Soft pairwise ordering.** The ratings already yield standard errors of the means. For independent mean estimates, an approximate ordering target is

```text
q_ijt = Phi((mean_it - mean_jt) /
            sqrt(SE_it^2 + SE_jt^2 + epsilon))
```

Match that to a pairwise probability from the predicted mean difference, using a training-selected scale. This normal approximation is an optional design, not the exact SOL algorithm. If raters are shared and their identities are retained, account for covariance or bootstrap raters instead. The objective is to discourage confident reversal of clear differences without enforcing arbitrary order among nearly tied noisy means. Keep mean MSE alongside it: ranking alone does not fix scale or offset. Pair generation stays entirely inside training folds; O(n²) pairs are not O(n²) independent faces.

**Continuous contrastive geometry.** Compare a ConOrd-style auxiliary objective on small target-specific projection heads. The paper's soft distance weighting is particularly relevant, but its full centroid/inference procedure need not be adopted. Eight targets do not define one common scalar order; applying a single ordinal loss to the entire shared representation can create conflicting constraints. Use separate target projections or a carefully scaled multidimensional distance, and retain a regression head.

**Rating distribution supervision.** More than 30 individual ratings per face/target provide information discarded by a mean. Predict a small distribution over the slider and optimize its CDF distance or soft cross-entropy together with mean MSE. Construct soft labels from individual ratings using interpolation onto fixed anchors; this can preserve the empirical mean more accurately than hard binning. Histograms can represent multimodality that a Gaussian cannot.

Distinguish two quantities: individual-rater variance describes disagreement; variance divided by rater count describes uncertainty in the estimated mean, under the independence assumption. Use the former for a rating-distribution target, the latter for uncertainty about mean order. Do not exchange them. A separate mean head avoids tying predictive precision to a coarse distribution discretization. Compare mean-only versus mean-plus-distribution with the same architecture.

**Multitask supervision.** Share the visual adaptation, with separate heads for each dimension and a mask for unavailable labels. Additional existing human-rated dimensions can be auxiliary tasks, but exclude *all* labels of an outer-test face from training, including auxiliary labels. Shared encoders can borrow useful visual structure; separate-head or target-specific adapter controls detect negative transfer. Do not require outputs to sum to one.

**11. Lower-priority ideas and methods to avoid overinterpreting.**

An image-dependent mixture of three or four prompt/prototype experts is a reasonable PPA-inspired experiment after a static learned prototype bank works. Regularize the gate and compare equal weighting. Merely gating the same frozen phrase scores remains within their information limit; use full visual features and learned prototypes or local features if the aim is to go beyond it.

Unlabeled generated faces may support teacher consistency or masked/local adaptation, but introduce them only with an explicit inductive protocol. Do not train on outer-test images without labeling that experiment transductive. Pseudo-ratings from the current model cannot by themselves supply independent evidence about its systematic errors. Synthetic degradation ranks from IQA should not be assumed to preserve or order perceived face ratings. Similarly, latent interpolation does not guarantee linear interpolation of human judgment.

Weight interpolation is distinct from prediction ensembling. A new regression head has no original zero-shot counterpart to average with; if interpolating visual weights or shrinking a LoRA update, refit/calibrate the head on training data. A blend with the existing kernel predictor should be selected from cross-fitted training predictions. Fit any residual correction to out-of-fold training residuals, not the baseline's optimistic in-sample residuals.

Heavy MLLM scorers, reinforcement learning, broad caption generation, full bilingual re-pretraining, and large independent prompt banks are not the first experiments I would fund at this sample size. Improvements in their original benchmarks do not establish an advantage under the fixed FG-CLIP2-backbone constraint.

**12. An experiment sequence that can establish a real gain.**


| Stage | Candidate                                                                          | Main question                                                   | Relative cost                        |
| ----- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| A     | Existing phrase kernels, direct-image ridge/kernels, phrase-plus-image combination | Is the phrase projection omitting useful information?           | Low once embeddings are cached       |
| B     | Anchored low-rank visual adapter; learned output prototypes                        | Does supervised geometry improve frozen features?               | Low                                  |
| C     | Masked local pooling with a global branch                                          | Is global pooling hiding useful spatial information?            | Moderate one-time feature extraction |
| D     | Last-four-block visual LoRA with a direct head                                     | Must the visual representation change?                          | Moderate/high                        |
| E     | Winning adapter/LoRA with ordinal or distribution loss, one at a time              | Does richer supervision improve generalization?                 | Incremental                          |
| F     | Learned soft context plus winning visual adaptation                                | Does adapting language add value beyond direct visual learning? | Higher training cost                 |
| G     | Limited unfreezing and training-selected blend with original kernels               | Is a further gain worth the compute?                            | High                                 |




Use a small grouped development protocol for triage, with identical splits and training sizes for the baseline and new methods. A five-fold model trains on substantially fewer faces than the original 50-fold model, so do not compare its RMSE directly to 0.06971 and attribute the difference to architecture. Within each outer fold, hyperparameters, training duration, prompts, loss weights, calibration, and ensemble weights must use only inner training/validation data. Every fold starts from the original checkpoint: supervised embeddings or adapter weights trained on all faces cannot be shared between folds.

For finalists, either rerun the original 50/10 protocol on identical saved splits, or run both baseline and finalists under the same predeclared, less expensive nested protocol. Report pooled RMSE, every target's RMSE/MAE/R² and correlation, calibration, tail errors, and paired group-bootstrap differences. Bootstrap intervals over fixed predictions describe conditional uncertainty; repeated seeds/folds add information about training variability. Previously searched faces remain an internal validation sample. A fresh generated-face cohort with new ratings is the strongest final confirmation.

Use a practical threshold to decide whether added training complexity pays off. For example, a **5% relative pooled RMSE reduction** means approximately **0.06622**, and a **10% reduction** means **0.06274**, on the original comparable protocol. These are decision targets, not forecasts. A consistent smaller improvement may still be useful for a particularly important target; a pooled win hiding material losses elsewhere needs examination.

The stored standard-error diagnostics give an aggregate RMS mean-SE of about **0.03328**. Under unbiased independent sampling, this suggests finite-rater noise contributes to evaluation error, but it is not a certified achievable floor: rater dependence, population differences, and model bias matter. It suggests investigating remaining signal rather than assuming the baseline is already at human reliability.

**13. Concrete integration boundaries.**


| Existing component                                          | Proposed extension                                                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FaceImpressionPipeline.features`                           | Reuse full frozen embeddings rather than reducing them immediately to scores.                                                                             |
| `FGCLIP2.encode_image_preprocessed`                         | Differentiable visual training entry point.                                                                                                               |
| `encode_dense_preprocessed` / `encode_regions_preprocessed` | Local features and masks for pooling experiments.                                                                                                         |
| `HumanRatingsStore`                                         | Expose individual-rating distributions alongside means, counts, and SEs.                                                                                  |
| `make_cv_splits`, `cross_validate_predictor`                | Preserve split contracts; image training can use row indices to resolve image paths inside callbacks.                                                     |
| `RegressionResults`                                         | Reuse the held-out prediction/metric contract across all architectures.                                                                                   |
| `ImpressionPredictor`                                       | Keep the phrase bundle intact; use a versioned bundle for adapted image models with preprocessing, checkpoint revision, adapters, head, and target order. |


The convenience image scoring methods and pipeline encoding are under `torch.inference_mode()`. They are appropriate for caching frozen features, not encoder training. Use the differentiable preprocessed path, freeze the base explicitly, and selectively enable the intended modules. Keep learned feature caches fold-specific and include adapter identity if cached. For local PyTorch execution use the requested `manip311` environment outside the sandbox so MPS is available. No external training service or hardware purchase is assumed by this proposal.

**My first implementation choice** would be direct-image and phrase-complement baselines, followed by a regularized small visual adapter. The first serious backbone experiment would be **late visual LoRA + a shared eight-output regression model + one uncertainty-aware ordinal auxiliary loss**, evaluated both alone and in a training-selected blend with the existing kernel model. Learned ordered text prototypes and target-specific spatial pooling are the next controlled branches. This order gives each additional source of capacity a fair chance to demonstrate that it contributes beyond the strong baseline.