# CLIP psychometry: measuring visual evidence for continuous human impressions

**A living methodology, 24 September 2026.** This is a proposed research framework and an executable diagnostic companion, not a validated psychometric instrument or a claim to have identified the causes of impressions. Evidence labels used throughout: **L** published literature; **E** repository observation with a recorded protocol; **H** hypothesis; **D** mathematical deduction under stated assumptions. New observations belong in [the ledger](../../research/clip_psychometry/INSIGHTS.md), with their implications folded back into the relevant chapter.

## 1. The framework: two measurement problems, not one adjective lookup

Our target is the average **intensity of an impression elicited by an image**, on the study's continuous slider scale. It is not whether the depicted person actually has a trait. The image includes expression, morphology, hair, accessories, lighting, background, framing, and synthesis artifacts. The target also depends on the raters, instructions, response scale, and context. A stable encoding of a visible cue can transfer to new images even when directly regressing the entire image embedding overfits. This motivates, but does not establish, an advantage for language-mediated features.

The starting hypothesis is that literal high-level words often do not operationalize these targets. In the existing trustworthiness study, “a joyful face” correlated .7010/.6283 with development ratings at p128/p256, versus .2966/.1138 for “a trustworthy face.” This is **E**, specific to that bank and those 400 development images. It does not establish that all direct impression prompts fail, or that CLIP pretraining cannot encode impressions. The key question is alignment between a particular operational definition and the model's learned language–image relation. [Local report](../../research/trustworthiness/report.md).

We therefore construct a **visual questionnaire**: each phrase is an item administered to an image through a frozen vision–language encoder. Several items may measure one visual construct. A second model maps their joint responses to human impressions. Neither layer validates the other automatically:

1. **Visual measurement:** do “square jaw,” “broad jaw,” and related phrases measure the intended visible form, including under different contexts?
2. **Impression prediction:** does that measured form, alone or in combination with presentation, expression, and image conditions, improve prediction of the study's mean judgment?

A phrase that predicts trustworthiness can be a poor square-jaw detector. A good square-jaw detector can be unnecessary in an already sufficient bank. A useful predictor can be a context-specific proxy. These are different outcomes requiring different actions.

### 1.1 A generative account

Let `X` be the displayed image; `V` its intended visual constructs; `C` photographic/contextual cues; `P_r` perceiver characteristics; and `I_t` the instructions for target `t`. A working model is

\[
J_{irt}=h_t(V_i,C_i,P_r,I_t,U_{ir}),\qquad
Y_{it}=\frac1{n_{it}}\sum_r J_{irt},\qquad
S_{ij}=\langle f(X_i),g(q_j)\rangle.
\]

`J` is the response of a participant, `Y` its sampled mean, and `S` a phrase score. The scoring convention, text mode, encoder revision, resolution and pooling are part of the item definition. `Y` approximates `μ_t(X)=E_r[J_t|X,I_t,population]`; it need not recover a single psychological process shared by everyone. Mixtures of perceiver strategies can produce nonlinearities even if each individual's strategy is simple.

A useful graph is `V,C → X → perceived evidence → J`, with perceiver/instruction effects into perceived evidence and `J`, and a separate `X → S` measurement branch. Dataset selection depends on image availability and possibly `V,C`. The graph is a hypothesis about generation, not learned causal structure. Apparent gender and race judgments are themselves observer-dependent measurements, not verified identity attributes.

### 1.2 What a strong kernel result licenses

A strong kernel is a useful **probe of a representation under a declared training algorithm and budget**. Its success does not prove Bayes optimality. Architecture comparisons in this repository used different inputs and tuning budgets at several stages. The earlier eight-target TabM benchmark found only a small, uncertain improvement over the kernel; later all-target observations are user-reported and not a completed controlled benchmark in this package. We treat representation design as the next productive hypothesis, not as a theorem that head improvements are exhausted.

**D: a useful error decomposition.** For a fixed fitted predictor on independent test images, deterministic `S=s(X)`, squared loss, and `m(X)=E[Y|X]`,

\[
E(Y-\hat f(S))^2=E\operatorname{Var}(Y|X)
 +E[m(X)-E(m(X)|S)]^2
 +E[E(Y|S)-\hat f(S)]^2.
\]

These are irreducible response/mean-estimation noise, information lost by the representation, and readout error. Within the middle term, missing constructs and poor detectors are generally **not separately identifiable from `(S,Y)`**. Their separation needs construct annotations, interventions, or genuinely independent measurement methods. This is why the package returns diagnostic evidence and unresolved alternatives, not a confident automatic causal diagnosis.

## 2. Target and cue ontology

“Simple visual” is a useful operational category, not a promise of objective ground truth. Perceived happiness is not a muscle-action annotation; rated gender presentation is not sex or identity; a color slider can combine pigment, illumination and semantic categorization. Read the original question and slider anchors before naming a construct.

| Type | Example | Measurement implication | Useful check |
|---|---|---|---|
| Unipolar presence | clear spectacles | absence may have no meaningful severity ordering | annotated presence, precision/recall, calibration, lower-tail audit |
| Bipolar continuum | light–dark visible hair | opposite ends may form a meaningful contrast | signed agreement, full-range monotonicity |
| Multicategory | spectacles / sunglasses / no eyewear | mutually exclusive classes only if ontology says so | category coverage and confusion, separate scores |
| Graded intensity | mouth opening | nonlinear saturation and noise floor | monotone calibration, binned errors |
| Configural cue | smiling mouth with tense eyes | relation among components matters | explicit conjunction versus separate components |
| Context moderator | presentation category | can matter only through interaction | conditional bundle gain, overlap cells |
| Relational judgment | typicality, familiarity, looks-like-you | reference population is part of construct | population-specific validation |
| Complex evaluation | trustworthy, electable, religious-looking | composite, culturally mediated visual judgment | joint cue banks, external population tests |
| Visibility/quality | shadowed eyes, blur | may alter human evidence as well as detector fidelity | missingness/occlusion checks and targeted perturbations |

A registry item records phrase ID, literal text, intended construct, functional roles, method/channel, polarity hypothesis, expected support, direct validation target if any, evidence source, and possible confounds. Polarity is an **annotation hypothesis**, never an instruction to negate regression inputs. Groups are hypotheses about functional roles. A broad literature group is not automatically a unidimensional reflective scale.

The same phrase can participate in several roles. Ablations must specify whether removing a role also removes conjunction phrases and its proxies. Implemented `Registry.columns(..., closure=True)` follows explicitly declared dependent groups; it does not discover all semantic substitutes by correlation.

## 3. From psychology to candidate phrases

The aim is content coverage and falsifiable detectors, not synonym multiplication. Translate an explanatory claim into an imageable cue, a proposed moderator, the relevant population, and a failure case. Then formulate a small comparison with a fixed functional role.

| Evidence and scope | Design implication here |
|---|---|
| **L:** Oosterhof & Todorov's [face-evaluation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2516255/) relates broad evaluations to valence/dominance and expression resemblance. | Separate visible affect and structural cues; allow joint mappings. Do not equate perceived trust with honesty. |
| **L:** Vernon et al. model impressions from [variable ambient photographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC4136614/). | Include visible image and pose information alongside facial geometry. |
| **L:** [Smile-authenticity research](https://pmc.ncbi.nlm.nih.gov/articles/PMC4053432/) examines contributions of facial actions and dynamics. | Describe visible mouth/eye components; a static image cannot validate a dynamic mechanism or true sincerity. |
| **L:** [Typicality research](https://pubmed.ncbi.nlm.nih.gov/25512052/) motivates a distinction between typicality and attractiveness. | Preserve both roles; allow nonlinear responses instead of forcing fixed signs. |
| **L:** [Graham & Ritchie](https://pubmed.ncbi.nlm.nih.gov/31006340/) manipulated eyewear; sunglasses reduced perceived trustworthiness in their experiment. | Separate clear lenses from dark occlusion. This does not supply a universal intelligence coefficient. |
| **L:** [Head-tilt research](https://pubmed.ncbi.nlm.nih.gov/31009583/) connects pose and apparent brow form to dominance. | Pose can change an apparent morphological cue; add pose controls and conjunction hypotheses. |
| **L:** [Cross-region evaluation](https://www.nature.com/articles/s41562-020-01007-2) investigates applicability of valence–dominance structure across regions. | Test transport of the mapping; a pooled structure does not guarantee item or individual-rater invariance. |
| **E:** The [local literature synthesis](../../research/trustworthiness/report.md) motivates sex-typical appearance, typicality, and image-condition groups. | “woman” plus “square jaw” versus “woman with a square jaw” is a proposed measurement comparison, not an established optimum. |

For each target, ask: what is directly visible; which cues are interpreted through stereotypes or context; what changes their meaning; which cues determine whether other cues can be seen; and what alternatives could explain the same prediction? For religious-looking or privileged-looking judgments, concrete clothing, grooming, age, expression and context are **candidate observable inputs**, not known causes or facts about the person. Keep direct high-level wording as a control until a fair replacement comparison supports removing it.

### 3.1 Wording experiments isolate the operation being changed

Compare “a joyful face,” “a joyful person,” and “a portrait of a joyful person” as **wording alternatives** for a declared role. Compare joy against jaw shape as **content alternatives**. Compare a long phrase against a short one without changing both syntax and construct at once. The hypothesis that “giving the impression of” weakens detection, or that “face” beats “portrait,” remains **H** until measured for this encoder and cue.

“Bad photo,” “dark photo,” and “uncanny face” need their own roles whenever those properties might influence human judgments. They cannot automatically be treated as semantically inert text augmentation. Negation, intensifiers, modifiers and conjunctions deserve explicit audits: noun detection can dominate the qualifier. Record the failure mode before rewriting the item.

### 3.2 The glasses decision tree

Start with a candidate reason to include eyewear: direct association, moderator role, visibility of the eyes, or coverage of a rare context. Zero marginal correlation cannot reject those roles. Establish whether the detector identifies the intended eyewear. Split clear spectacles, sunglasses, and absence when they have distinguishable mechanisms or detector responses. Assess each one's coverage; too few sunglasses makes inference weak, not evidence of irrelevance.

Next compare a matched base bank `B` with `B+clear`, `B+dark`, and `B+clear+dark`; where justified, examine expression/eyewear conjunctions. Keep the rest of the bank fixed and tune each readout on the same inner folds. A zero unique gain can mean substitution by other items. A null result with narrow support says little about deployment to eyewear-rich images. Retain an item for coverage only as an explicitly declared design constraint, separate from a demonstrated MSE gain.

This generalizes to any coarse binary cue: **split a category when subtypes plausibly change either the human response or the measurement error, then check whether the data support the split**. Taxonomic detail without support increases variance and can harm a kernel's distance geometry.

## 4. Detector validity when visual labels are scarce

Use multiple kinds of evidence; do not collapse them to one score. `diagnostics.detector_summary` reports marginal associations as descriptive alignment checks. It does not rank the entire bank by target correlation. A direct labeled visual construct supports calibration tests; an impression target usually does not validate its alleged visual mechanism.

`cards.extreme_cards` produces a large phrase heading, labeled low/high panels with **2×2 images each**, exact IDs/scores, and a JSON manifest. Every phrase/channel gets its own card. `cards.boundary_card` samples score quantiles; `cards.residual_cards` separates overprediction from underprediction. These are generated artifacts for human/LLM inspection, not evidence of correctness by generation alone.

### 4.1 Extremes cannot validate the entire response curve

A detector can have excellent precision among its top four images and poor discrimination everywhere else. Inspect random images, intermediate quantiles, suspected threshold boundaries, and disagreement cases. Score-blinded annotation of observable cues should use a prespecified rubric and an “unclear/occluded” response. If an LLM views a labeled extreme card, its judgments are hypothesis generation; use a second, blinded pass to measure agreement. Two CLIP-derived methods and an LLM are not necessarily independent witnesses.

A minimal annotation design samples known strata over score and context, records inclusion probabilities, and preserves disagreements. `annotation_sheet` exports blinded image records and inverse sampling probabilities; `annotation_metrics` uses these weights for population-oriented precision, recall, Brier score and calibration bins. A cutoff must be learned on separate annotation-training data. Extreme-only samples cannot estimate prevalence, recall, or natural-population precision without a suitable sampling design.

### 4.2 Absence is not a negative amount of presence

For a unipolar cue, similarity below a boundary might represent arbitrary semantic differences among absent cases. But it may also encode visibility, ambiguous thin frames, or gradual presence. Do not erase the lower tail on intuition alone.

Compare raw scores, training-ECDF ranks, and a positive hinge with a **training-only** threshold. A score quantile is an operational gate, not a probability of presence. `features.FeatureMap` fits all such transformations inside each training fold. `metrics.lower_tail` checks criterion variation inside a training-defined lower region when annotations exist. Binary calibration should use presence annotations, with logistic or isotonic calibration and held-out Brier/AP diagnostics; these continuous scores are not ordinal IRT responses unless an observation model justifies that conversion.

An apparently harmless standardization can undo the purpose of gating: standardizing a sparse hinge to unit variance amplifies rare cases. Compare the gate with its resulting kernel distances, not just its histogram. Do not use zero cosine or zero logit as a universal semantic boundary.

### 4.3 Conjunctions and contrasts

For two detector scores, estimate empirical percentile functions on training images and set

\[
Q_{A\wedge B}(x)=\min\{\hat F_A(S_A(x)),\hat F_B(S_B(x))\}.
\]

This is a fuzzy ranking rule, **not a calibrated joint probability**. It prevents one large component from compensating for a small other component. Compare retrieval with the lexical conjunction, including images selected by only one method. `cards.conjunction_card` exposes overlap and disagreement, and `metrics.conjunction_overlap` reports top-set overlap. Low overlap identifies informative annotation cases; it cannot decide which detector is correct.

For prediction compare `B+{A,B}`, `B+{AB}`, and `B+{A,B,AB}`. If the composite substitutes well, it may compress a useful interaction; if it adds value, it may capture a new visible relation or merely new confounding. A “woman” detector can dominate “woman with a square jaw”; inspect all four component-support cells. Low scores on a conjunction have several possible causes and are not a coherent “opposite” trait.

Positive-minus-negative contrasts can cancel common image effects, but also discard asymmetry or qualifier information. Preserve separate components as a competing representation. Never orient negatively worded items solely by English sentiment.

**D: high agreement can hide a useful difference.** For unit-variance item responses `A,B`, with correlation `r`, `Var(A−B)=2(1−r)` and

\[
\operatorname{Corr}(A-B,Y)=\frac{\operatorname{Corr}(A,Y)-\operatorname{Corr}(B,Y)}{\sqrt{2(1-r)}}.
\]

Thus two highly correlated, positively target-associated “opposites” can have a meaningful contrast. The standardized sum and difference give a diagnostic common-response/contrast basis; they add no information to the original pair. `metrics.contrast_audit` fits item scaling on supplied training rows and reports both components; `cards.contrast_card` retrieves their disagreement extremes. A useful contrast still need not measure the English qualifier: it could separate smile intensity, age or another confound. Low difference variance also makes it sensitive to measurement noise. The development follow-up in the ledger illustrates why calling a modifier “ignored” from pair correlation alone is premature.

## 5. Psychometrics of a visual questionnaire

Write a measurement hypothesis before fitting a latent model:

\[
S_{ijm}=a_{jm}+\lambda_{jm}V_i+\beta_{jm}C_i+M_{im}+e_{ijm}.
\]

Here `m` indexes wording/encoder/pooling method. Shared encoders, near-identical captions and shared image confounds create correlated errors. Agreement can reflect `V`, a nuisance, or both. A high alpha can be achieved by duplicating an item; that is not new validity evidence. [Revelle & Zinbarg](https://personality-project.org/revelle/publications/rz09.pdf) explain why internal consistency, homogeneity and reliability must be distinguished.

### 5.1 What is implemented and what it means

`psychometrics.factor_audit` fits a one-factor model on training rows, reports signed loadings, model-based composite omega, and **held-out** covariance residuals. It requires at least three nonconstant items and explicit orientation. Omega is

\[
\omega=\frac{(\sum_j\lambda_j)^2}{(\sum_j\lambda_j)^2+\sum_j\psi_j},
\]

under a unit-variance common factor and diagonal residual model. Correlated residuals require the additional `2Σψ_jk` terms; the implemented omega therefore remains an assumption-dependent model summary. Opposite loadings, large held-out residual covariances or near-zero uniqueness are reasons to revisit the model, not to proclaim a reliable instrument. No automatic loading-based deletion occurs.

`psychometrics.multitrait_method` contrasts agreement within a role across channels against across-role agreement within a channel. This is descriptive MTMM evidence, not a fitted identifiable MTMM model. `psychometrics.conditional_drift` compares held-out measurement prediction with and without context and context×anchor terms, conditioned on an independent construct anchor. A gain indicates conditional measurement heterogeneity under that anchor model. Without a valid anchor and overlap, it cannot identify DIF; conditioning on a CLIP proxy may create the discrepancy.

For serious latent validation, plan confirmatory multi-trait/multi-method models, partial or approximate invariance, nonuniform DIF, and crossed image×phrase×view generalizability models. [Asparouhov & Muthén's alignment work](https://www.statmodel.com/Alignment.shtml) offers a route for many-group invariance analysis; its assumptions are not met simply by naming phrase groups. These advanced models are **not implemented here** because unverified identification and tiny subgroup counts would produce misleading precision. The implemented held-out checks expose when richer models and annotations are needed.

### 5.2 Reflective versus formative structures

Synonyms intended to detect the same smile cue can be reflective indicators. Jaw, age appearance, eyewear and lighting jointly contributing to an impression are **formative predictors**, not interchangeable items of one trustworthiness scale. Maximizing their internal consistency can destroy useful diversity. Distinguish three reasons for multiple phrases: semantic replication, complementary components, and moderators. Only the first motivates a unidimensional reliability model.

An independent geometric annotation, new encoder family, or controlled edit is usually more informative for discriminant validity than twenty paraphrases from one encoder. Even independent encoders may share training-data biases. Construct meaning requires external anchors; factor rotation and correlation structure alone cannot supply it.

## 6. Replacing marginal mutual information with conditional predictive utility

MI itself is not wrong. **Univariate MI as a selection criterion is misaligned with this task.** It can discard pure moderators and redundant individually strong features can consume the whole budget. In squared-loss regression we need information about the conditional **mean**, not every distributional difference.

### 6.1 Derivation of the replacement

Let `B` be the retained feature set and `G` a candidate functional group. Define oracle risk `R*(B)=E[(Y−E[Y|S_B])²]`. The conditional mean utility is

\[
U^*(G\mid B)=R^*(B)-R^*(B\cup G)
=E[(E[Y|S_{B\cup G}]-E[Y|S_B])^2]\ge0.
\]

The equality follows by conditional-mean orthogonality. It allows zero marginal utility and positive joint utility. Distributional conditional MI can be positive solely through changing conditional variance, even when this mean-prediction utility is zero. Thus there is no single universal “correct MI replacement”; the squared-loss estimand matches the stated output. This oracle contrast is related to [algorithm-agnostic variable importance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10652709/). Our implementation estimates a **finite algorithm contrast**, not that paper's efficient estimator or null test.

For a declared kernel learning procedure `A`, estimate paired out-of-fold loss differences

\[
\hat U_A(G\mid B)=\frac1n\sum_i
[(Y_i-\hat f_{B,-k(i)}(S_i))^2-(Y_i-\hat f_{B\cup G,-k(i)}(S_i))^2].
\]

Every scaler, gate, selected kernel and regularization value is fit inside the corresponding training split. Each representation gets the same inner-search budget. Negative estimates are allowed: finite-sample estimation cost can exceed information benefit. Code: `evaluation.compare`, `metrics.paired_contrast`.

### 6.2 Unique, redundant, and joint utility

For the full bank `F`, drop an entire role with all declared positive/negative variants and channels: `U(G|F\G)`. This is **unique utility given substitutes**. Also evaluate the role in a smaller scientifically meaningful context. No single leave-one-out score measures total contribution.

For disjoint groups `A,D` and background `B`, compute four risks and the complementarity contrast

\[
C(A,D\mid B)=R(B+A)+R(B+D)-R(B)-R(B+A+D).
\]

Positive values mean the joint gain exceeds the sum of separate gains under this procedure. Negative values can reflect redundancy. This is not a unique partial-information decomposition, not automatically a mechanistic interaction, and not a causal effect. Correlation, changing regularization and missing support can affect it. `metrics.complementarity` reports this contrast from matched predictions, with all four constituent losses. For example a centered product of independent zero-mean cues can carry signal while each marginal mean relationship vanishes; use that reasoning to protect plausible moderators, not to run an obvious synthetic demonstration.

Coalition averaging can distribute shared utility across a bank. [SAGE](https://arxiv.org/abs/2004.00668) formalizes global loss-based attribution accounting for interactions. Exhaustive coalition fitting is inappropriate for an initial 1,000-image bank. The package uses **budgeted, prespecified functional coalitions**, with shared prediction caches, rather than labeling a few dropouts “Shapley values.”

### 6.3 Three operations that must not be conflated

**Refit after removal** asks whether remaining features can compensate. **Perturb inputs to a fixed fitted predictor** asks how that predictor relies on them, possibly off distribution. **Intervene on an image and re-rate it** asks about changed human judgments, if the intervention is well defined. A frozen encoder does not mean frozen kernel coefficients during a feature comparison. We freeze the *learning rule*, then refit the readout for each bank. A literal fixed-head evaluation is separately useful for deployment robustness.

Conditional permutation can preserve realistic feature dependencies only with a good conditional sampler. Unconditional shuffling of one phrase while its synonyms stay fixed creates impossible feature combinations. Neither procedure establishes a causal visual mechanism. Conditional independence tests have additional sampling and null-distribution requirements; paired descriptive intervals here are not such tests.

### 6.4 Inference and search discipline

The uncertainty unit is the original image/identity/latent family, not a phrase, augmentation or image pair. The paired cluster bootstrap in `metrics` conditions on fitted OOF models; it omits training/search uncertainty and cross-fold dependence induced by overlapping training sets. Report it as a conditional descriptive interval. Near a zero-importance null, oracle importance inference is nonregular; do not attach naive significance claims.

Inspecting validation error cards and then writing new prompts turns those images into development data. Nested CV protects algorithmic tuning inside folds; it cannot undo an agent's prior inspection of the outer images or labels. Track an exposure ledger. Freeze the entire phrase-generation/selection procedure before confirmation on untouched images. [Cawley & Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html) analyze the overfitting risk of selection itself. No assertion of a universally optimal bank follows from many development comparisons.

## 7. Kernel geometry: a phrase bank is also a metric

With standardized features, an RBF kernel uses `exp(−γ Σ_j(z_j−z'_j)²/p)`. Copying a phrase changes the relative weight of its construct even though it adds no information. Adding many synonyms can therefore appear to improve “coverage” while merely reweighting distance. Changing bank width also changes the denominator and effective bandwidth. The same issue affects polynomial and linear kernels through their dot products.

**D: identify the question before comparing banks.** A raw-column bank comparison answers whether that full representation/metric works better. A semantic-information comparison needs controls for weighting: equal total mass per declared disjoint role, matched bandwidth search, or a comparison against duplication of existing items. `FeatureMap(balance_blocks=True)` gives each nonoverlapping role equal total squared-distance weight after standardization. It does not remove correlated-error inflation or guarantee duplication invariance inside a heterogeneous role. Avoid claiming a perfectly controlled information experiment from this one correction.

Another hidden path occurs with per-image score centering. If a deleted phrase remains in the row mean used to center retained phrases, its information survives the ablation. Every bank must be built **before** centering. The exact legacy adapter removes both channel copies before applying its existing bank-centering code. The generic analysis kernel uses training-column standardization and explicit optional transforms, without implicit per-image centering.

Small banks can also lose cue diversity through feature-level averaging. Averaging text embeddings, averaging phrase scores, averaging augmented features, and averaging final predictions are different operations. A nonlinear kernel generally satisfies `f(E[S]) != E[f(S)]`. Evaluate the inference operation actually intended for deployment.

## 8. Augmentation as a measurement experiment

An image perturbation is not automatically label preserving because it was useful in classification. Lighting, temperature, contrast, blur, crop and occlusion may alter the very impression being predicted. Horizontal reflection can alter asymmetry; tiny translation can affect local max-pool selection. Synthetic faces can contain local defects that matter to both people and encoders.

Declare the expected relation for each **cue × transformation × target**: invariant, expected directional response, or unknown. The augmentation adapter reuses `fgclip2_multitarget.tensor_views.PreparedImages` and the existing transformation settings. It exports scores indexed by original image and view; all views remain with the source image in folds. It creates no competing augmentation implementation or changes to training cache ownership.

### 8.1 Separate instability, bias and collapse

For clean score `s_i`, views `s_iv`, and training reference variance `σ²`, report

\[
B=E_{iv}(s_{iv}-s_i)/\sigma,\quad
D=E_{iv}(s_{iv}-s_i)^2/\sigma^2,\quad
W=E_i\operatorname{Var}_v(s_{iv})/\sigma^2.
\]

`D` includes systematic drift; `W` does not. Also report clean–view rank agreement, top-set retention, and a between-image/within-image variance ratio. A constant detector is perfectly stable but useless. `robustness.augmentation_audit` returns a null reliability ratio for collapsed scores, preserves raw drift alongside normalized drift, and flags low variance. This is a view-repeatability proxy, not full generalizability theory or construct validity.

Use `(prediction_on_view−prediction_on_clean)²` for model instability. Use `(prediction_on_view−original_rating)²` as augmented risk **only under the explicitly recorded label-preservation assumption**. Otherwise obtain new ratings or report perturbation sensitivity without calling it error. `prediction_audit` requires that declaration. For multi-target deployment the same perturbation can be admissible for one target and inadmissible for another.

A robust selection objective can combine clean mean-prediction risk with prespecified invariant-view drift and domain worst-group risk. We intentionally do not hide those choices in a single default weighted score. Robustness, construct fidelity, and MSE form a constrained/Pareto decision, not a universal ranking.

## 9. Causal reasoning without causal overclaiming

The framework uses causal hypotheses to select what to measure and what might moderate a cue. Observational kernel utility estimates remain predictive. A visual cause of a rating and a useful predictor of a rating need not coincide. [Hernán & Robins](https://miguelhernan.org/whatifbook) give the identification framework for interventions; the key requirements here are a defined intervention, consistency, exchangeability and overlap, not merely nonlinear adjustment.

| Failure hypothesis | Distinguishing evidence | What ordinary feature diagnostics cannot establish |
|---|---|---|
| Missing cue | valid new detector improves joint held-out risk | that cue causes the judgment |
| Missing moderator | four-bank contrast plus adequate joint support | biological or universal moderator mechanism |
| Confounding/proxy | context-stratified measurement and transfer experiments | remove confounding by adding all available columns |
| Collider conditioning | explicit selection/measurement graph and alternate designs | causal meaning of partial correlation |
| Mediator removal | define total versus direct effect before adjustment | interpret conditioned utility as a total effect |
| Measurement heterogeneity | external anchor, context drift and balanced annotations | distinguish DIF from anchor error automatically |
| Support failure | joint-cell counts, conditional ranges, nearest-training distances | extrapolate a rare conjunction reliably |
| Perceiver mixture | repeated individual ratings and rater covariates | identify mixture mechanisms from means alone |
| Domain shortcut | domain-held-out tests and cue-preserving interventions | prove invariance from one augmentation family |

Residual-conditioned selection can itself distort relationships: choosing the worst model errors selects on a function of `Y` and `S`. It is excellent for hypothesis generation but unsuitable for estimating population cue effects. Recheck a proposed cue on a fresh or prespecified sample.

Controlled same-identity edits should retain identity, edit realism and non-target cues as far as possible, measure manipulation success, and collect paired human ratings with randomized display/order. An edit to jaw shape may change apparent expression or synthesis quality. Randomizing an imperfect edit identifies the effect of that edit procedure, not automatically an isolated jaw mechanism. The package's residual and conjunction cards prioritize candidates for this work; it does not synthesize causal evidence.

## 10. Diagnosing failures and allocating the next observation

| ID / level | Observable signature | Diagnostic code | Next useful action; remaining ambiguity |
|---|---|---|---|
| T1 target definition | labels/anchors do not match proposed words | dataset metadata + detector summary | verify instructions; no metric repairs a changed estimand |
| T2 mean uncertainty | large errors at noisy means | `metrics.noise_accounting` | inspect rater data; SEM is not disagreement itself |
| M1 wrong construct | retrieval annotations contradict intended cue | cards + annotation metrics | rewrite cue; target correlation cannot settle validity |
| M2 qualifier failure | adjective variants retrieve almost same faces | conjunction cards + agreement | annotate disagreement; possible shared noun dominance |
| M3 response-scale error | lower scores vary despite confirmed absence | lower-tail + fold-local feature maps | compare gate/calibration; retain useful ambiguity |
| M4 common-method agreement | same-method correlation exceeds cross-method evidence | MTMM + factor residuals | add external measurement, not more synonyms |
| M5 context-dependent detector | context explains score conditional on anchor | conditional drift | check overlap and anchor validity before calling DIF |
| P1 substitution | single deletion null, role deletion matters | role-closure comparisons | group substitutes; do not prune singly |
| P2 complementarity | joint gain exceeds separate gains | four-bank contrast + support table | expand paired roles; inspect finite-sample stability |
| P3 metric distortion | gains follow phrase multiplicity | balanced-role versus raw-column comparison | choose metric intentionally; not “more knowledge” by default |
| P4 readout limitation | valid representation improves with matched readout search | nested kernel / exact legacy backend | separate head capacity and sample-size limits |
| R1 augmentation bias | systematic drift with low within-view variance | augmentation audit | label transformation relation; repeatability alone misses drift |
| R2 domain transport | clean internal gain reverses outside domain | context/domain metrics | reconsider proxy or population scope |
| E1 selection exposure | claimed holdout was used for prompt discovery | provenance/exposure manifest | acquire new confirmation data |

This table is an **evidence routing system**. Several rows can apply simultaneously. `diagnostics.evidence_routes` emits rule-triggered next actions and alternatives using explicit supplied thresholds; it never returns a posterior probability of the “true cause.”

### 10.1 Information-efficient annotation

Spend annotation effort where hypotheses disagree. If a lexical conjunction and component rule agree on almost every image, annotating their shared top cases adds little. Sample their disagreement regions, the proposed gate boundary, and contexts with weak coverage, plus a probability-sampled calibration component. Use independent annotations to discriminate M1/M2/M3 before spending on many kernel fits. Run a predictive comparison only when each outcome would change the next action.

For individual residuals, compare nearest training images in the **actual fitted feature geometry**, not raw image resemblance alone. A large nearest-neighbor distance suggests coverage risk; a close neighbor with different `Y` suggests unresolved visual information, rating noise, or an insufficient mapping. These are alternatives, not a definitive diagnosis. `metrics.support_distance` and residual cards support this distinction.

## 11. Mean ratings, uncertainty, and local accuracy

Observed means are noisy estimates, even with many raters. Under unbiased independent mean errors `Y=μ+ε`, independent of a held-out predictor, `E[(Y−f)²]=E[(μ−f)²]+E[SEM²]`. `noise_accounting` reports the plug-in subtraction **without clipping negative values**; negatives expose sampling error or broken assumptions. It is not a universal ceiling. Shared raters, missingness and response context can violate independence. Rater disagreement is psychologically meaningful heterogeneity and is not interchangeable with SEM of the mean.

Equal image weighting estimates average image risk. Inverse-SEM weighting estimates a different, reliability-weighted population and can favor easy/uncontroversial images; choose it deliberately. Joint target comparisons should report per-target errors and raw-unit macro MSE; variance-standardized macro loss answers a different question. Missing targets remain missing, never zero-filled observations.

Correlation can stay high with poor calibration; report MSE, MAE, slope/intercept and target-range diagnostics. Defining high-rating tails with test labels is descriptive, not an operational selection policy. Use training thresholds for predeclared regions. Image pairs are dependent because they reuse images; same-identity and random close-rating pairs answer different questions. Local ordering precision cannot be inferred from a high global correlation.

## 12. A later standalone prompt optimizer: specification before training

This task establishes the measurement framework first. No existing prompt, pool or head training is changed. The future optimizer belongs in its own module and imports these diagnostics.

### 12.1 Source protocol capsules

**PromptSRC (2023)** uses consistency at feature/logit levels, Gaussian epoch aggregation, and text-template diversity. Its appendix specifies SGD and learning rate .0025; deep vision/text prompt settings depend on benchmark. These choices are a protocol, not just a learning-rate suggestion. Classification templates can change relevant image semantics here, so their suitability must be re-evaluated. [Paper](https://arxiv.org/html/2307.06948).

**ManiPT (2026 preprint)** combines LLM description prototypes, cosine constraints, and normalized frozen+prompt feature fusion. The final objective constrains the fused features. Appendix A specifies CLIP ViT-B/16, 16 prompts per layer in both encoders, normal initialization SD .02, Adam .0025, weight decay .0005, betas .9/.999, epsilon 1e−8; one constant warm-up epoch at 1e−5 then cosine; λ=12. ImageNet uses 40 epochs/batch 128 for base-to-novel/few-shot; other datasets 50/32; transfer/domain runs use five ImageNet epochs. Its conditional geometric/generalization arguments do not guarantee improvement for FGCLIP2 regression or identify a true semantic manifold. [Paper, §4 and Appendix A/B](https://arxiv.org/pdf/2602.19198).

These are not identical algorithms, and the newer work is not established here as the first theoretical treatment of prompt tuning. Text-only adaptation of FGCLIP2 and regression supervision are explicit departures. Pin source revisions and record every departure, including optimizer, prompts/layer, losses and reduction conventions.

### 12.2 The correct differentiable regression surrogate

A permanently frozen dual vector with moving reference features is a different objective from re-optimizing the kernel head. For a support/query split inside development data, use

\[
A_\theta=(K_{SS,\theta}^{c}+\alpha I)^{-1}(Y_S-\bar Y_S),\quad
\hat Y_Q=\bar Y_S+K_{QS,\theta}^{c} A_\theta.
\]

`S` and `Q` here denote support/query rows, not the phrase-score symbol above. Center cross-kernels using **support statistics**. Differentiate through the solve with fixed positive alpha and a fixed kernel recipe during a prompt-update segment. The differential is `dA=−M⁻¹(dK)A+M⁻¹dYc`; labels do not change, so only the first term remains. Query ratings must never enter the support solve, standardizer or semantic anchors. Repeated optimization on query labels makes them training labels for prompt parameters; keep an independent outer validation/confirmation set.

**D:** for positive-semidefinite centered kernels, `||(Kc+αI)⁻¹||₂ ≤ 1/α`. Consequently kernel perturbations can be amplified by small α, with `||dA|| ≤ ||dK||·||Yc||/α²` for fixed labels. Feature cosine proximity alone cannot ensure stable regression, particularly after division by tiny feature standard deviations. Monitor readout condition numbers, output drift and semantic validity together. This is our regression-specific reasoning, not a transferred classification theorem.

`differentiable.kernel_predict` supplies this solve as a building block, including optional stop-gradient support/dual controls to compare estimands explicitly. It is **not a standalone prompt trainer**. Frozen image features allow text-only experiments to reuse the existing image cache. Adapted text vectors still require semantic anchors built from validated cue descriptions, not averaged trustworthiness synonyms.

Anchor each cue separately, regularize admissible view drift, and retain a frozen branch as an explicit baseline. Do not apply a class softmax over co-occurring visual cues: smile, glasses and shadow are not exclusive classes. If borrowing logit KL, specify the comparison distribution and justify it; cue-score preservation may be more interpretable. Tune loss weights in nested development, log gradient magnitudes, and evaluate whether prompt tuning improves the detector or simply makes it encode the target under a familiar name.

## 13. Research protocol and living knowledge

A cycle is: define estimand → cue mechanism hypothesis → registry and falsifiers → detector audit → budgeted functional-bank comparison → admissible-view/context audit → update ledger → freeze a proposal → confirmation. Early discovery uses development images only. Record source hashes, image/target order, original-group IDs, channels, exposure, split indices, candidate sets, transforms and selected recipes. Store per-image predictions and losses so a later question does not require refitting everything.

This package provides portable NumPy exchange data, optional adapters to the current repository, grouped/nested comparisons, diagnostics and card generation. It intentionally avoids automatic unrestricted prompt search, expensive exhaustive subsets, and causal claims inferred from regression. The first empirical analysis uses already cached trustworthiness scores and the original development partition. Its images/cards and numerical results are **development evidence**, never a refreshed headline benchmark.

Every ledger entry should contain: observation; status (E/H/D/L); provenance and exposure; competing explanations; the decision it changes; code and chapter references; and the next discriminating observation. A failed metric that taught us something belongs here as much as a successful phrase. See [agent brief](AGENT_BRIEF.md), [usage](../README.md), and [insight ledger](../../research/clip_psychometry/INSIGHTS.md).

### Selected-set retrieval validation and missing-annotation bounds

A disagreement card motivates a criterion audit, not a conclusion that the more
plausible conjunction wins. For a fixed retrieved set R of k images, let a be
annotated, h of those satisfy the visual criterion, and u=k-a remain unannotated.
Finite-sample precision lies in [h/k,(h+u)/k]. Only when u=0 is h/k identified.
Reporting h/a as precision silently assumes missing annotations are representative;
that assumption fails when annotations were collected for an earlier method's
retrieval union. New wording must either expand that union's annotations or retain
the bounds. These are missing-data bounds, not sampling confidence intervals.
Annotation uncertainty remains separate, even with complete coverage.

Percentile intersection is a rank-based retrieval rule, not a logical AND of
presence probabilities. A rare cue can be absent even at its 80th percentile.
For sparse attributes, compare the conjunction against a criterion-labeled union
before interpreting low set overlap as inferior lexical composition. Any learned
presence threshold needs separate calibration and evaluation observations; a
threshold learned on a selected union does not establish population calibration.
Implemented by `retrieval.retrieval_audit`.

### Localizing a predictive gain without claiming its cause

For paired out-of-fold predictions define image gain
`g_i=(y_i-p_baseline_i)^2-(y_i-p_candidate_i)^2`. For a declared, disjoint
partition S, total gain is exactly `sum_s (n_s/n) mean(g_i | S_i=s)`.
Report both conditional gain and population-weighted contribution: a large gain
on a rare subset need not explain the aggregate result. This is a decomposition
of observed prediction error, not a causal mediation analysis. Bins based on human
ratings may be used retrospectively to explain performance, but cannot be used as
deployment gates without an independently available predictor. Do not infer that
adding a smile phrase improved smile detection merely because its regression gain
is concentrated among high happiness ratings. Shared correlated cues, metric
reweighting and kernel selection are competing explanations. Implemented by
`retrieval.risk_partition`; original image-level paired gains remain the audit trail.

### Empirical decision capsule: iteration 01

On 400 exposed development images, role-PCA distance harmed prediction against
both raw RBF and the matched raw product-kernel control. Adding raw functional
coordinates recovered much of the loss, but did not establish metric-decoupling
utility. Keep the matched function-class control whenever testing a learned metric;
no optimal-transport elaboration is justified by this result.

The complete 71-image conjunction union supplied a more consequential correction:
minimum-percentile happy/glasses retrieval contained eyewear in only 9/40 images,
versus 39/40 for the lexical conjunction. Low overlap was an alarm for criterion
validation, not evidence about which detector was right. Concrete rewording did
not automatically improve retrieval, even when it sounded less abstract.

Across two fold assignments, concrete smile additions lowered trustworthiness
MSE by 1.72%/2.80%. Adding all four concrete families beat replacing the direct
family for that target; direct-family removal harmed perceived intelligence in
both assignments. The useful rule is target-conditional bank comparison, not
universal rejection of abstract phrases. These are development leads with search
and annotation limitations, not confirmed gains or causal identification.
Full numbers, exposure history, and runnable studies: [iteration report](../../research/clip_psychometry/iteration_01/REPORT.md).

### Addition-first engineering and targeted geometry

The relevant intervention is B → B∪A before B → (B\R)∪A. The first asks if a
proposed representation provides usable information or favorable geometry beyond
the current bank; the second asks whether A can substitute for a declared role R.
Ablating R alone is a supporting control, never an answer to which wording should
replace it. An unchanged error after adding A does not identify absence of missing
information: the specific detector, feature scaling, readout and finite sample all
remain possible limitations. A current-bank baseline must use the actual captions,
image channels and readout; historical banks cannot silently stand in for it.

For standardized scores z and declared nonnegative squared-distance weights w,
use D_w(i,j)=sum_l w_l(z_il-z_jl)^2/sum_l w_l. Normalizing weights removes a global
bandwidth-change confound. Target a concrete overcounting or spatial-support
hypothesis, e.g. downweight repeated categorical captions but preserve anatomy.
The product exp(-gamma D_w) (1+z_i'z_j/p) keeps the raw functional coordinates while
changing proximity. A matched w=1 product separates this change from introducing
the richer product kernel. `Representation.metric_weights/separate_metric` and
`gram(split_product)` implement these interventions. No optimal-transport claim
is needed. Adding standardized contrasts preserves information but alters the
regularized metric; predictive gain alone cannot establish a new detected cue.

### Contrast amplification: orientation versus overall kernel scale

Let each original coordinate have training variance one. Appending q standardized
contrasts at squared weight w gives total coordinate variance p+wq over p+q
coordinates. Without correction this changes mean kernel energy, as well as the
relative emphasis on contrasts. Multiplying the augmented representation by
sqrt((p+q)/(p+wq)) restores unit average training variance. Compare this normalized
version when interpreting a gain mechanistically: otherwise some gain may be a
regularization/bandwidth change. With deliberate base weights replace p in the
denominator by their squared scaling sum. All contrast standardization is fitted
inside each training fold. Semantic polarity remains a hypothesis; empirical
opposite-score covariance does not define a psychometric scale automatically.

### Successful intervention capsule: conditional utility and contrast geometry

Iteration 02 uses the actual 772-phrase bank and original p576 global/local score
pipeline, correcting the older historical-bank experiments. Addition-only simpler
alertness captions reduced full-bank alertness MSE by 5.89%, although their small
standalone detector set had lower out-of-fold target correlation (.554 vs .661).
A conditional trustworthy wording set showed the converse: strong standalone
alignment (.867) without improvement when added to the full bank. Thus detector
alignment and incremental regression utility need independent columns in every
comparison; neither may be substituted for the other. Unmeasured visual fidelity
is a third question, not established by either result.

The clearest engineering gain combined 32 wording additions with moderately
amplified, energy-normalized descriptor contrasts. Using the original kernel
selector, MSE reductions across two development fold assignments were 3.97/4.60%
for trustworthiness, 3.85/2.75% for perceived intelligence, 9.53/11.23% for alertness,
4.53/0.57% for grooming, and 1.97/2.04% for happiness. More aggressive target-specific
variants improved alertness about 13% but could damage other targets. This supports
investigating target-specific geometry and a moderate shared checkpoint; it does
not establish generalization of the searched winner.

Why contrasts can help without new information: if standardized original scores
are z, pair-difference matrix A and train-fitted contrast standard deviations D,
the appended map is [z, sqrt(w) D^-1 A z]. Its squared Euclidean distance is
`delta_z' [I + w A' D^-2 A] delta_z`, up to width/energy normalization. It changes
the penalty on specific differential directions while retaining the original
space. In a linear kernel this also changes coefficient regularization; in a
nonlinear kernel it changes similarities. A gain therefore supports engineered
geometry/regularization, not discovery of an independent cue or proof of improved
latent-construct reliability. The normalized-energy control retained the gains.

[Iteration 02 report](../../research/clip_psychometry/iteration_02/REPORT.md) gives
all interventions, source hypotheses, exact baseline equivalence, limitations,
and portable candidate manifests. Existing production defaults remain untouched.

### Target-specific AGOP pruning: estimand and protected baseline

The requested [Radhakrishnan et al. paper](https://arxiv.org/html/2212.13881v3)
updates a kernel's feature metric using the average gradient outer product (AGOP).
Its image pruning example masks pixels using a network feature-matrix diagonal
and retrains. Our phrase pruning is an adaptation, not that experiment's replication.
We use per-target scalar regression gradients; summing over outputs would hide
which target needs a phrase. Diagonal-only recursion sacrifices rotated feature
directions for coordinate attribution and computational economy.

For standardized coordinates z, define d²=(z-z_i)'M(z-z_i)/p and f(z)=b+sum_i a_i k(z,z_i).
For Gaussian k=exp(-gamma d²), grad f=-(2gamma/p) M sum_i a_i k_i(z-z_i).
For Laplace k=exp(-gamma sqrt(d²)), replace 2gamma by gamma/sqrt(d²) for each term.
Use zero subgradient at exact coincident points in Laplace. The centered-kernel
fit uses effective dual coefficients a-mean(a), accounting for query centering.
The diagonal score s_j=mean_x (partial_j f(x))² is a sensitivity of the fitted
predictor to standardized coordinate perturbations, not unique information,
causal importance or an estimate of measurement validity. Training-only scaling
is essential because this sensitivity depends on coordinate units.

A phrase's score is the sum over its global/local channels; both are retained or
removed together. Individual paraphrases can compete for retention: this is a
compression decision, not an assertion that their underlying construct is absent.
Optional role bundles remove all declared items together, addressing a different
compression granularity. Correlated proxies can redistribute gradient sensitivity;
a low score is a candidate for an empirical pruning trial, never sufficient proof
that its cue is irrelevant. Export both fold stability and exact selected texts.

Each inner training split fits its own relevance model and mask. Validation labels
choose retention fraction and kernel recipe; outer labels choose neither. The
unpruned original bank is always available. A guard accepts a target-specific
change only after a declared relative improvement margin on inner predictions.
This protects model selection but does not guarantee noninferiority on new data.
Never choose fallbacks using the same outer labels on which success is reported.

Report per-target MSE ratio to baseline, macro mean ratio (equal target weight),
maximum ratio, counts improved/worsened, and the strict observed all-target
nondegradation indicator. A favorable mean cannot compensate for a bad target
under the strict objective. Also report the union of retained phrases: large
per-target pruning does not automatically reduce encoder cost when the union
remains large. A globally removed phrase must be unused by every target, and
training-only final-fit masks are deployment artifacts, not new validation.

AGOP can recognize an interaction that a marginal screen misses: for the fitted
function f(z)=z1*z2, each squared partial derivative depends on the other coordinate's
second moment even when each coordinate has zero marginal correlation with f.
This motivates sensitivity-based ranking without proving that the learned pilot
has recovered the true interaction. In strongly correlated phrase banks, gradients
also concern perturbations away from the observed score manifold. Refitting and
validating actual removals is consequently essential; attribution alone is not a
pruning certificate. A surviving proxy may replace another item's predictive role.

### Stable expansion of a target-specific bank

Re-ranking a larger global bank can evict an existing feature under a fractional
budget even when a new phrase is not useful. This confounds testing additions with
changing the incumbent representation. A stable-extension policy first fits the
incumbent target mask and then compares that same mask against its union with
ranked new phrases, including zero additions. Preprocessing always follows column
selection. A target that declines additions receives exactly its incumbent model;
unrelated global bank growth cannot change its distances. Selecting the wrong
addition based on noisy validation remains possible, and an imperfect incumbent
is not repaired by this construction. Report extension versus pruned incumbent
and versus the original full bank separately.

A relevance pilot can itself be misaligned with the final predictor. The companion
`frozen_readout_agop` computes gradients of the actual selected kernel ensemble,
using an additional train-only selection split. For the linear kernel its gradient
is constant, sum_i a_i z_i/p; for `(1+z'z_i/p)^2`, it is
`2 sum_i a_i (1+z'z_i/p) z_i/p`; the Gaussian expression is given above. Ensemble
gradients must be summed before squaring, since mean squared component gradients
omit cross terms and do not describe sensitivity of the actual ensemble. Comparing
this one-step attribution against Laplace-RFM ranking diagnoses pilot mismatch;
it is not a claim to reproduce recursive feature learning with that ensemble.

### All-target pruning findings and adoption boundary

Iteration 03 evaluated all 34 targets, not the five-target subset. Broad per-target
AGOP pruning of the expanded bank reduced macro normalized MSE 2.33%/3.00% across
two fold assignments and retained about 277/273 of 852 phrases per target.
Nevertheless, 11/10 targets worsened, with worst increases 6.26%/7.19%. Attributing
the actual kernel rather than a fixed Laplace pilot improved 26/34 targets, but
still left a worst increase of 4.53%. This supports sensitivity-based compression,
not a claim that feature selection automatically protects every target.

Milder budgets and conservative inner-validation adoption reduced some risks, but
no nontrivial tested policy satisfied strict observed all-target nondegradation.
The strongest mild guard achieved it only by retaining the baseline everywhere.
That is an unchanged control, not evidence of successful selection. Thresholds
chosen with validation are fallible; none of these results establishes population
noninferiority or authorizes describing an outer-label-selected fallback as honest CV.

Stable extension preserved incumbent masks and reduced the worst additional
penalty to 1.56%, compared with 5.40% under re-ranking in one split. It did not repair
incumbent regressions. A bank-growth policy and a bank-compression policy should
therefore be distinct operations: default to the same incumbent coordinates for
non-adopting targets, and evaluate deliberate removals separately.

The union of useful target masks still covered the entire expanded phrase bank.
Per-target sparsity and encoder-wide phrase deletion are different objectives.
Use the exported per-target sensitivity and selection-frequency tables to propose
future compression, not to declare unused constructs from a small gradient score.
[Full all-target report](../../research/clip_psychometry/iteration_03/REPORT.md).


### Promotion into the full-CV pipeline

The user subsequently authorized the actual-readout AGOP expanded-bank model as
`fgclip2_multitarget`'s default. This is a policy decision to adopt the current
candidate, not a retroactive claim of all-target noninferiority. The 1004-image
matched experiment compares the old 772-phrase baseline and the complete new
852-phrase, target-specific procedure. The 400 development images are included,
so these OOF measurements still carry the history of model exploration.

For target t define r_t = MSE_new,t / MSE_old,t. Macro normalized risk is the
unweighted mean of r_t, while raw target-macro RMSE is the square root of the
mean target MSE. The former gives each target equal relative importance; the
latter weights absolute rating errors. Report both and the full vector r, its
maximum, number above 1 and above 1.01. A mean benefit does not imply a Pareto
improvement. Fold-wise signs reveal instability without providing independent
replications: training sets overlap. The existing paired cluster bootstrap of
image losses conditions on fitted OOF predictions and does not include refitting
or exploratory selection. It cannot certify simultaneous population noninferiority.
`cv_report.compare_run` implements this comparison only after verifying identical
rows, outcomes, masks and fold assignments, and complete image coverage.

Reusable implementation: `target_kernel.fit_target_kernel` separates the estimator
from the historical research runner. `fgclip2_multitarget.agop` groups identical
label masks, preserves phrase identities, saves fitted per-target readouts, and
passes them to the common checkpoint inference path. Streaming frozen feature
extraction preserves the original half-cache arithmetic without retaining dense
patch tensors for the entire dataset. See [current model](../../fgclip2_multitarget/CURRENT_MODEL.md).

### A targeted next step: supervised geometry without discarding coordinates

The current diagonal AGOP ranking becomes a binary mask. A richer, still testable
intervention retains continuous relevance and cross-coordinate directions. Write
G_t for the matrix of target-t gradients in training-standardized coordinates,
A_t=G_t'G_t/n, Q_t=p A_t/tr(A_t), and H_t=(1-b)I+b Q_t. H_t is positive semidefinite
for b in [0,1]. Diagonal Q changes individual distance contributions; full Q also
changes which *combinations* constitute large distances. Trace normalization fixes
the average coordinate mass but not the empirical distribution of pairwise
distances; bandwidth selection must remain available. Keeping I protects directions
that the finite-sample pilot missed. Do not recenter gradients indiscriminately:
a useful linear predictor has constant gradients and zero gradient covariance.

For scalar predictions, a full metric can be evaluated without a p×p factorization:
`z'H_t w/p = (1-b)z'w/p + b (G_t z)'(G_t w)/||G_t||_F²`.
This identity preserves all gradient directions and the ensemble cross terms.
`learned_geometry.metric_dots` implements it. No second coordinate standardization
follows the metric: that would erase diagonal weighting. Ordinary kernel centering
and an intercept still follow, as they address a different issue.

For heterogeneous targets, a shared candidate metric can average trace-normalized
A_t rather than averaging gradients. Averaging gradients first can cancel opposite
rating directions; summing raw AGOPs can let high-variance targets dominate. Neither
normalization establishes reliability or causal relevance. Treat shared geometry
as a candidate beside independent target geometry, not an assumption imposed by
sharing an encoder. `leading_direction` uses a linear operator for their averaged
outer product, avoiding a dense multi-target covariance allocation.

A split learned from this direction is a *model partition*, not a CV split.
Evaluation groups remain shared and untouched across all targets. Target-specific
partitions may learn different local representations; they also discard support
across a threshold. Soft routing trades discontinuities against evaluating both
leaves. The proposed one-level study learns the split direction, median and IQR
from training data, and relearns the local pilot and normalization inside each leaf.
A high-gradient direction is not necessarily a moderator that changes which cues
matter; splitting on it can merely split target intensity and reduce sample size.
The experiment must distinguish these possibilities through prediction results.

Source protocol and all departures from [xRFM](https://arxiv.org/html/2508.10053v3)
are recorded in [the exploratory design](../../research/clip_psychometry/xrfm_geometry/PROTOCOL.md).
This is a concrete probe of our kernel geometry, not a claim to reproduce the
paper's complete algorithm. Extensive theory about transport or causal regimes
should await evidence that this intervention is useful on our data.


The first geometry screen is now empirical: continuous weighting improved over
nested hard pruning by 3.10% in mean relative MSE on matched development folds,
with 25/34 targets improving. The tested shared and target-specific partitions
instead lost accuracy. [Results](../../research/clip_psychometry/xrfm_geometry/REPORT.md)
support retaining the metric-learning derivation and implementation, while keeping
partitioning as a concise failed probe rather than an adopted theory of visual
regimes. These results do not identify detector validity or causal effects.

### Bank policies, final adoption and honest production evaluation

The completed original/expanded × pruning/weighting factorial comparison did not
support a useful seven-target exclusion rule. The adopted B2 default uses the full
expanded bank and independently selected continuous AGOP metrics for every target.
The historical policy and runners have been retired; the study's report and OOF
predictions remain in `research/clip_psychometry/target_bank_factorial`.

Primary comparisons use the 32-target set S excluding looks-like-you and memorable.
Report mean_t∈S(MSE_new,t/MSE_control,t − 1), mean_t∈S(MSE_new,t − MSE_control,t),
win/loss counts, worst regressions and trustworthy individually. These omissions
are reporting choices, not omitted model outputs or phrase-bank exclusions.

The production protocol and ceiling derivation are in
[PRODUCTION.md](../../fgclip2_multitarget/PRODUCTION.md); the implementation is
`fgclip2_multitarget/production.py` and `ceiling_report.py`. An inner validation
row can be reused when refitting the selected model on all outer-training rows.
An outer test row cannot be predicted by the other fold models for evaluation,
because those models trained on it. Ten-fold averaging is deployment fusion,
not ten independent held-out predictions per image.

For Y_i = μ_i + ε_i with independent unbiased errors, estimate Var(ε_i) by
s_i²/n_i. Centering gives E[ε'Hε]/(N−1) = mean_i(s_i²/n_i), so the signal
variance estimator is Var(Y, ddof=1) minus that mean. Its ratio to Var(Y) is
mean-rating reliability; **the square root** is the attainable Pearson ceiling
against noisy image means. This remains valid under unequal rater counts and
heteroskedastic variances, but shared-rater covariance needs the general correction
tr(HΣ)/(N−1). It cannot be recovered from lists without rater IDs. This is an
estimated measurement-noise ceiling, not proof that every stable rating component
is visually learnable. Do not truncate observed r to an estimated ceiling.
