#!/bin/sh
# Note: we need this run in a file because currently the 'language: system'
# type in pre-commit-config doesn't seem to support multiple commands (even on using &&)
uv run coverage run --source src -m pytest -rfpsxEX --disable-warnings --verbose --durations=3

uv run coverage html --ignore-errors
