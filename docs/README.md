# docs

**[alpha_chess.pdf](alpha_chess.pdf)** — *AlphaChess: How the model works, and
how it is trained.* An 18-page technical description of the method as it is
actually implemented here: the board and move encodings, the network, PUCT
search, the self-play pipeline, the training objective, and how strength is
measured. It cites the files and functions it describes, and calls out where
the implementation departs from the AlphaZero paper and why.

The PDF is committed, so nothing needs building to read it.

## Rebuilding

`alpha_chess.typ` is the source, written in [Typst](https://typst.app). The
compiler is pip-installable and self-contained (no LaTeX, no system packages):

```bash
python3 -m venv /tmp/docvenv
/tmp/docvenv/bin/pip install typst
/tmp/docvenv/bin/python -c "import typst; typst.compile('alpha_chess.typ', output='alpha_chess.pdf')"
```

Run it from this directory. It is deliberately kept out of the project venv and
out of `requirements.txt`: building the documentation is not a prerequisite for
running or training anything.

If you have the `typst` CLI installed instead, `typst compile alpha_chess.typ`
does the same job.

## Keeping it honest

The document quotes parameter counts, plane layouts, default hyperparameters
and measured throughput. When those change in the code, they need changing
here too — there is no mechanism that checks it. The figures were last
verified against the tree at the commit that introduced this directory.
