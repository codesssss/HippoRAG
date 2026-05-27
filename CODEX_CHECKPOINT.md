# Codex Checkpoint

Date: 2026-05-25 09:56 CST
Workspace: `/mnt/nvme/code/HippoRAG`
Recovered context: `/tmp/codex_019e5968_recovered_tail.md`

## Current State

- The recovered context was read only as context; the old session should not be replayed.
- The preceding work had already completed a small appendix patch in `paper/sections/A_appendix.tex`:
  - Added run-log mapping rows for mechanism controls and FCRG.
  - Clarified top-100 readout pool vs traversal candidate limit `L=120`.
  - Removed `Support gain` from the general evaluation-protocol metric list.
  - Reworded case-study `Evidence-need binding` as a binding step within evidence-need mining.
  - Clarified the `graph-main` row label as HippoRAG 2 / PropRAG.
- The paper PDF was then regenerated successfully as `paper/current_full_preview_acl.pdf`.
- The user decided not to further adjust the appendix table-only page.

## Latest Thread

The latest meaningful unfinished thread is about EMNLP/ACL end-matter requirements:

- `Limitations` is the required section title.
- `Ethical Considerations` is the recommended optional ethics-section title when adding one.
- The next useful action is to inspect the current manuscript end matter and, if missing and appropriate, add a short `\section*{Ethical Considerations}` before the bibliography without compiling unless asked.

## Constraints

- Keep paper edits surgical.
- Do not run broad audits.
- Do not compile unless explicitly requested.
- Do not stage or commit anything.
