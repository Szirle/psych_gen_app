# Development context

The application-specific guidance in this file and `dev/agent-notes/` applies
only to the face-stimuli generator served by `app.py` and its Flutter client.
For that application, start with [dev/agent-notes/README.md](dev/agent-notes/README.md).

Requests concerning separate projects in this folder, such as [cgs-gan/](cgs-gan/),
[CLIP/](CLIP/), or traversal experiments, are outside that scope. Work on those
projects when requested, using their own relevant instructions. Do not load the
face-generator notes, update its documentation, or apply its upkeep workflow
merely because another project shares this repository.

**Do not rebuild Flutter or stage `build/web` for changes to these separate
projects.** The broad build/staging rules in `.cursor/rules/` are scoped to the
face-generator application, not every file in this folder. Apply them to a
cross-project change only when it actually changes the served application or its
Flutter client, or when the user explicitly requests a build.

Use `/opt/anaconda3/envs/manip311/bin/python` locally. Python importing PyTorch
must run outside the sandbox so MPS is available; remote VMs use their own environment.
Prefer established CLI tools, targeted reads, and quiet output. Preserve unrelated
working-tree changes. For ordinary research or documentation work, do not run unit
tests, visual verification, or superfluous checks unless requested. Only launch
the webapp when asked to test it or when diagnosing an error requires the UI.
Stop any temporary verification server before handoff. If a simple command fails
and continuing would require a costly investigation the user can resolve quickly,
ask instead.

After substantial face-generator application changes, follow the
[upkeep workflow](dev/agent-notes/workflow.md).
