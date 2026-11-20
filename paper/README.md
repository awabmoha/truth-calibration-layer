# Papers

| File | Status |
|---|---|
| [TCL-Theory-Paper-v2.md](TCL-Theory-Paper-v2.md) / `.docx` | **Current theory paper** (October 2026). Separates the jointly trained proposal (TCL-J) from its frozen baseline (TCL-F), defines an operational trust vector, and states hypotheses H1-H6. References verified against their sources. A status note at the top records how H1-H3 turned out. |
| [acl-phase2/](acl-phase2/) | **Phase 2 negative-result paper** (November 2026), ACL format: `tcl-phase2-negative-result.tex` and `references.bib`. Joint training does not make trust heads transfer, and every trained head transfers below raw token confidence. |
| [TCL-Phase2-Negative-Result.md](TCL-Phase2-Negative-Result.md) | The same paper as Markdown, for reading on GitHub. The `.tex` is the version of record. |
| `TCL-Theory-Paper-v1.docx` | First theory draft. Superseded by v2, which fixes its frozen-versus-joint contradiction and its unobservable trust dimensions. |
| [TCL-v0-research-writeup.md](TCL-v0-research-writeup.md) / `.docx` / `.pdf` | Archived June 2026 empirical note on TCL-v0. Superseded by the Phase 1 results in [../docs/phase1/](../docs/phase1/). |

The `.docx` of v2 is generated from the Markdown source with pandoc:

```bash
pandoc paper/TCL-Theory-Paper-v2.md -o paper/TCL-Theory-Paper-v2.docx
```

## ACL Format

Papers from Phase 2 on use the official ACL LaTeX style ([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)). The style files are not copied into this repository. To build a PDF, open the ACL template on Overleaf, replace its `.tex` with the paper's `.tex`, upload the `references.bib`, and compile. Locally, put `acl.sty` and `acl_natbib.bst` from the official repository next to the `.tex` and run `pdflatex`, `bibtex`, then `pdflatex` twice.
