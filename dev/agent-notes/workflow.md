# Change and context upkeep

## Before editing

Read the relevant current contract and source. Preserve unrelated working-tree
changes and keep edits scoped to the requested feature. For API changes, update
Python, the matching Dart transport/entities/decoding, and
`docs/API_COMMUNICATION.md` together. Extend the existing feature boundaries;
do not undertake a framework rewrite to implement one behavior.

Keep research intent, observed behavior and proposals distinguishable. Never
silently replace the manuscript's method with a convenient approximation while
retaining its name. Document a partial implementation and its deviations.

## Close substantial work

1. Record the delivered behavior and actual validation, including untested model,
   device and empirical aspects. Do not imply a run from the existence of a test.
2. Rewrite affected sections of `implementation.md` as current behavior; update
   the API guide for transport changes. Keep detailed schemas in the API guide.
3. Remove completed backlog actions and obsolete workarounds. Add only concrete
   remaining work, clearly labelling proposals and unverified suspicions.
4. Update validation selectors and source-map caveats when their targets change.
   Preserve the scientific intent in `vision.md` unless the author changes it.
5. Hand off a concise account of what changed, why, checks actually performed and
   material remaining limits. Do not build or run tests just for note upkeep.

This app does not inherit LangTorch's release cadence, subsystem bug IDs or
benchmark archive. Do not create empty ledgers or fabricate historical evidence.
For a requested app release, `pubspec.yaml` holds the Flutter version; document
the release's capabilities and compatibility limits and change the version as
part of that release. Documentation upkeep alone needs no version bump, commit,
publication or deployment.

If recurring defects eventually justify a ledger, give each reproduced defect
one stable ID, reproducer and fix reference. Until then, keep evidence with the
relevant issue/change rather than duplicating it across these notes.
