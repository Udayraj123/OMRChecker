#!/bin/sh
echo "PYTHONPATH=$PWD:$PYTHONPATH uv run python3 $@"
PYTHONPATH="$PWD:$PYTHONPATH" uv run python3 "$@"
