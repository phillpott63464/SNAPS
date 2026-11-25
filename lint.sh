cp .python-version .python-version.old
echo 3.13 > .python-version
uv sync
uv run ruff check --fix python/refactor notebooks/ tests/
uv run blue python/refactor notebooks/ tests/
mv .python-version.old .python-version
uv sync
