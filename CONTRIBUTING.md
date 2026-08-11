# Contributing

## Environment

Use the locked uv environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --all-groups
```

Do not hand-edit `uv.lock`. Change `pyproject.toml`, run `uv lock`, inspect the diff, and commit both
files. Raw UCI sentences, pretrained caches, and model binaries must remain ignored.

## Before a pull request

```bash
make check
```

Tests must remain offline. Transformer tests should construct a miniature local model rather than
depend on a mutable remote download. New evaluation code must keep the test set outside model,
hyperparameter, epoch, and threshold selection. JSON writers must reject NaN and Infinity.

The manual `Full benchmark` workflow is the networked integration path for the pinned UCI archive
and DistilBERT revision. If results change intentionally, regenerate all report CSV/JSON/Markdown/PNG
files together and explain the protocol change.

## Pull requests

Keep each change focused. Describe the leakage boundary, tests run, expected result changes, and any
new data/model license obligations. Never commit secrets, raw review text, or downloaded weights.
