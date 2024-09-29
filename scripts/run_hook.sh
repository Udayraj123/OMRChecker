#!/bin/sh
SCRIPT_DIR="$(dirname "$0")"
export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"
# echo "$SCRIPT_DIR/$@"
# echo "PYTHONPATH=$PYTHONPATH"
python3 "$@"