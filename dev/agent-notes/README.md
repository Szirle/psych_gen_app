# Agent context

These notes cover the face-stimuli generator launched by `app.py`, not adjacent
research projects. Snapshot: 2026-09-17, based on source inspection, without a
build, model execution, or test run.

The manuscript defines scientific intent; application code establishes current
behavior. Tests record regression expectations, not proof that this checkout
passes them. Proposed work is not an implemented capability.

| Stage | Read |
| --- | --- |
| Before changing behavior | [vision.md](vision.md), then the relevant section of [implementation.md](implementation.md). |
| Choosing work | [backlog.md](backlog.md): open work and proposed priorities. |
| Resolving a scientific or API ambiguity | [documentation-map.md](documentation-map.md), then the original source. |
| Selecting necessary checks | [validation.md](validation.md). |
| Closing substantial work | [workflow.md](workflow.md). |

Read only the relevant files. Do not preload generated assets, model weights,
data pickles, vendor trees, or research archives. Files whose names contain
`traversal` are temporary unrelated experiments; FG-CLIP/`CLIP/` is outside this
application's scope. Neither supplies requirements for this app.

Keep intent in vision, current contracts in implementation, and next actions in
backlog. Link detailed API documentation rather than duplicating payload schemas.
