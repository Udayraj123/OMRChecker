#!/bin/sh
echo "PYTHONPATH=$PWD:$PYTHONPATH python3 $@"
PYTHONPATH="$PWD:$PYTHONPATH" python3 "$@"
