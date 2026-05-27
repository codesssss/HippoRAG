# Project Notes for Codex

## Paper Build Environment

- The local LaTeX engine available on this machine is `tectonic` (`Tectonic 0.16.9`).
- Build the current ACL paper from the `paper/` directory with:

```sh
tectonic -X compile current_full_preview_acl.tex
```

- Do not try `latexmk` or `pdflatex` first in this workspace; they are not installed in the current environment.
- The main draft is `paper/current_full_preview_acl.tex`, which includes section files from `paper/sections/`.
