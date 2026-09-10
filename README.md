# harness-engineering-code

Companion code for *Harness Engineering for Production AI Systems: Sensors, Gates, and Bounds for
LLM Applications That Have to Work When the Model Doesn't*, by Ranjan Kumar.

One Python package, `harness`, that grows across the book. There is a git tag per chapter, so
checking out a tag gives you the package exactly as the book describes it at the end of that
chapter, with the tests that were passing then.

```bash
git tag --list
git checkout ch01-demo-cliff
```

## Running it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

**The test suite runs with no API key, no network, and no model.** Every tag. That is the central
technique of Chapter 15 arriving early: if the model is a seam you can cut, most of what you need
to know about a harness is knowable without it.

## Layout

| Path | What it is | Chapter |
|---|---|---|
| `harness/state.py` | `RunState`, the object every component reads, and `RunContext`, the read-only view | 1 |
| `harness/errors.py` | Every failure the harness raises | 1 |
| `harness/audit.py` | The one-page harness audit | 1 |
| `harness/components/` | The controls each chapter adds | 4 onward |
| `harness/graph/` | The only framework-aware code in the package | Part II |

Nothing below `harness/graph/` imports a framework. Delete `graph/` and you still have a working
harness.

## Versions

Python 3.12 or newer. The book was written against 3.13. Library pins are in `pyproject.toml` and
were verified against PyPI on 10 September 2026.
