#!/usr/bin/env sh
# Note: we need this run in a file because currently the 'language: system'
# type in pre-commit-config doesn't seem to support multiple commands (even on using &&)
coverage run --source src -m pytest -rfpsxEX --disable-warnings --verbose --durations=3

coverage html --ignore-errors
