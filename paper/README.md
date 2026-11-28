# Papers

| File | Status |
|---|---|
| [tcl-final/](tcl-final/) | **Final paper, first draft** (November 28, 2026): *Token Confidence Is Hard to Beat*, covering Phases 1-4a, in TMLR format: [PDF](tcl-final/tcl-final.pdf), `tcl-final.tex`, `references.bib` (`tcl-final-draft.docx` is a pandoc conversion for reading). For arXiv, then TMLR. Built from [../docs/final/](../docs/final/). |
| [TCL-Theory-Paper-v2.md](TCL-Theory-Paper-v2.md) / `.docx` | **Theory paper** (October 2026), kept unchanged as the record of the predictions. Separates the jointly trained proposal (TCL-J) from its frozen baseline (TCL-F), defines an operational trust vector, and states hypotheses H1-H6. References verified against their sources. A status note at the top records how every hypothesis turned out. |
| [acl-phase2/](acl-phase2/) | **Phase 2 negative-result paper** (November 2026), ACL format: [PDF](acl-phase2/tcl-phase2-negative-result.pdf), `tcl-phase2-negative-result.tex`, and `references.bib`. Joint training does not make trust heads transfer, and every trained head transfers below raw token confidence. |
| [TCL-Phase2-Negative-Result.md](TCL-Phase2-Negative-Result.md) | The same paper as Markdown, for reading on GitHub. The `.tex` is the version of record. |
| `TCL-Theory-Paper-v1.docx` | First theory draft. Superseded by v2, which fixes its frozen-versus-joint contradiction and its unobservable trust dimensions. |
| [TCL-v0-research-writeup.md](TCL-v0-research-writeup.md) / `.docx` / `.pdf` | Archived June 2026 empirical note on TCL-v0. Superseded by the Phase 1 results in [../docs/phase1/](../docs/phase1/). |

The `.docx` of v2 is generated from the Markdown source with pandoc:

```bash
pandoc paper/TCL-Theory-Paper-v2.md -o paper/TCL-Theory-Paper-v2.docx
```

## ACL Format

The Phase 2 paper uses the official ACL LaTeX style ([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)). The style files are not copied into this repository. To build a PDF, open the ACL template on Overleaf, replace its `.tex` with the paper's `.tex`, upload the `references.bib`, and compile. Locally, put `acl.sty` and `acl_natbib.bst` from the official repository next to the `.tex` and run `tectonic -X compile tcl-phase2-negative-result.tex` (how the committed PDF was built), or `pdflatex`, `bibtex`, then `pdflatex` twice.

## TMLR Format

The final paper uses the official TMLR style ([JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file)), also not copied into this repository. Put `tmlr.sty`, `tmlr.bst`, and `fancyhdr.sty` next to `tcl-final.tex` and run `tectonic -X compile tcl-final.tex` (Tectonic 0.17.0 built the committed PDF). `\usepackage[preprint]{tmlr}` gives the arXiv version with my name; `\usepackage{tmlr}` gives the anonymous submission version (the author block and repository links must then be anonymized too).
