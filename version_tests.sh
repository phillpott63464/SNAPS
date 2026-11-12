#!/usr/bin/env bash

backup=$(mktemp)
cp .python-version "$backup"

restore_version() {
    cp "$backup" .python-version
    rm -f "$backup"
}
trap restore_version EXIT

versions=(3.8 3.9 3.10 3.11 3.12 3.13 3.14 3.15)
failed=()

for version in "${versions[@]}"; do
    echo "Testing Python $version"
    printf '%s\n' "$version" > .python-version

    if ! uv run pytest -q; then
        failed+=("$version")
    fi
done

if ((${#failed[@]})); then
    printf '\nFailed versions: %s\n' "${failed[*]}"
    exit 1
else
    echo
    echo "All versions passed."
fi
