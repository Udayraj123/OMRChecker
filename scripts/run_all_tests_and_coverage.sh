#!/bin/sh
# Note: we need this run in a file as the 'language: system' type in pre-commit-config
# doesn't seem to support multiple commands (even on using &&)
uv run coverage run --source src -m pytest --cov-fail-under=50 -rfpsxEX --verbose --durations=3

uv run coverage html --ignore-errors