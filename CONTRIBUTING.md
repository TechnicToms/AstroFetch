# Contributing to AstroFetch

Thanks for your interest in AstroFetch — a PyTorch-friendly, ML-ready gateway to
planetary science data. Contributions are welcome, and **new body modules (Mars
first) and new instrument datasets are especially wanted**.

This file is the quick-start for contributors. The design rules, architecture,
and phased roadmap live in [`AGENTS.md`](AGENTS.md) — read it before a
non-trivial change; it is the source of truth for how the codebase is meant to
fit together.

## Ways to contribute

- **Report a bug** or **request a feature** with the [issue templates](https://github.com/TechnicToms/AstroFetch/issues/new/choose).
- **Add an instrument dataset** (a new STAC-backed sensor) or a **new body module**.
- **Improve docs, examples, or tests.**
- Not sure where to start? Open a discussion issue first — for anything
  substantial, please float the idea before writing a large PR.

## Development setup

AstroFetch uses [uv](https://docs.astral.sh/uv/) for everything. You do not need
to manage a virtualenv by hand.

```bash
git clone https://github.com/TechnicToms/AstroFetch.git
cd AstroFetch
uv sync              # create the venv and install the package + dev/test groups
```

Common commands:

```bash
uv run pytest                 # unit tests only — no network (the CI default)
uv run ruff check             # lint
uv run ruff format            # format (the single source of truth for layout)
uv run ty check               # type check
uv run --group docs mkdocs serve   # preview the docs site at localhost:8000
```

Live endpoint tests hit real government servers and are **deselected by
default**. Run them deliberately, one at a time, when verifying an endpoint —
never in a loop:

```bash
uv run pytest tests/live -m live
```

## Making a change

1. **Branch** off `main` (`git switch -c my-change`).
2. **Write the code and its tests together.** Every new public function needs a
   NumPy-style docstring with a runnable example and at least one unit test.
3. **Keep unit tests network-free.** Mock STAC responses and read small local
   GeoTIFFs; never point a unit test at a live service. If you add or change an
   endpoint or query, add or update the corresponding fixture.
4. **Run the full local gate before pushing:**
   ```bash
   uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
   ```
5. **Open a PR** into `main` with a clear description. If you reimplemented
   anything instead of wrapping existing archive tooling, justify it there
   (see design rule 1 in `AGENTS.md`).

## Coding conventions

These are enforced in CI; the short version:

- **Formatting is automated.** `ruff format` owns layout, quotes, and line
  length — do not hand-align or fight it. Run it before every commit.
- **Imports** are sorted and grouped (stdlib, third-party, first-party) by
  ruff's isort rules; no wildcard or unused imports.
- **Type hints** on every public function; keep `ty check` clean.
- **Naming** follows PEP 8 (`snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`).
- **Conventional commits** (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `chore:`).
- **No print statements** in library code; use the `astrofetch` logger. Raise
  `EndpointError` for remote failures and `ValueError` for bad user input.

Non-negotiable architecture rules (endpoint URLs live only in
`data/endpoints.py`, the quantitative COG path and rendered tile path never mix,
the cache is disposable, and so on) are spelled out in `AGENTS.md` — please skim
them so review can focus on the substance of your change.

## Adding a new instrument dataset

The typical shape of such a change:

1. Add the dataset class in `src/astrofetch/moon/datasets.py` (subclass
   `InstrumentDataset`, declare its `probe`, `instrument`, STAC `collection`,
   and `all_products` map of product → `Product(layer, asset)`).
2. Register its layers and wire it into the `MOON` catalog in
   `src/astrofetch/moon/layers.py`.
3. Export it from `src/astrofetch/moon/__init__.py` and `astrofetch/__init__.py`.
4. Add unit tests (mocked reads) and, if useful, a single live check.

A new **body** (e.g. Mars) is a new sibling module under `src/astrofetch/`, not
edits scattered through the Moon code — the body-agnostic `data/` layer is
reused as-is.

## Code of conduct

Be kind, be constructive, assume good faith. Planetary science is a small,
collaborative community; let's keep it welcoming.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache 2.0](LICENSE) license.
