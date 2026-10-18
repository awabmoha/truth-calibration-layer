# Papers

| File | Status |
|---|---|
| [TCL-Theory-Paper-v2.md](TCL-Theory-Paper-v2.md) / `.docx` | **Current theory paper** (October 2026). Separates the jointly trained proposal (TCL-J) from its frozen baseline (TCL-F), defines an operational trust vector, and states hypotheses H1-H6. References verified against their sources. |
| `TCL-Theory-Paper-v1.docx` | First theory draft. Superseded by v2, which fixes its frozen-versus-joint contradiction and its unobservable trust dimensions. |
| [TCL-v0-research-writeup.md](TCL-v0-research-writeup.md) / `.docx` / `.pdf` | Archived June 2026 empirical note on TCL-v0. Superseded by the Phase 1 results in [../docs/phase1/](../docs/phase1/). |

The `.docx` of v2 is generated from the Markdown source with pandoc:

```bash
pandoc paper/TCL-Theory-Paper-v2.md -o paper/TCL-Theory-Paper-v2.docx
```
