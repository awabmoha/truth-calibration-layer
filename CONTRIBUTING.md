# Contributing

TCL is a closed research program (November 28, 2026): I am not running new experiments, and the final paper is being written from what is here. Contributions that make the existing record more correct or easier to check are very welcome.

## Most Useful

- **Reproductions.** Rerunning any analysis from the committed records (see the README's "Reproducing" sections and [tcl_experiments/README.md](tcl_experiments/README.md)) and reporting whether you get the same numbers.
- **Errors.** A wrong number in a checkpoint, a bug in a script, a mislabeled answer in an audit, or a broken link. Point to the file and line.
- **Missing prior work.** A paper that already reports one of the findings, or contradicts one. The literature review is in [docs/final/](docs/final/).
- **Questions about a result.** If something in a protocol or checkpoint is unclear, ask.

## Not Accepted

- Changes to a locked protocol's text, a recorded decision, or a run's records. These are preregistration records; corrections are added as dated notes or amendments, never as silent edits.
- New experiments presented as part of this program. If you build on it, please do so in your own repository and cite this one ([CITATION.cff](CITATION.cff)).

## How

- **Issues:** use the templates (bug or error, question about a result, missing prior work).
- **Pull requests:** keep them small, one fix per pull request, and say what you checked. For anything beyond a typo, open an issue first.
- Code is Python 3.10+; analysis scripts run on a CPU from committed files. Do not commit hidden-state arrays (`*.npz`) or model weights.

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
