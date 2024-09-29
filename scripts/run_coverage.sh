#!/bin/sh
coverage run --source src -m pytest -rfpsxEX --disable-warnings --verbose --durations=3 && coverage html --ignore-errors