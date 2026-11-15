#!/bin/bash
# DEV set runs: used ONLY to choose the personal-mode prompt design.
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
$PY -u run_transcribe.py --set dev --model small --mode plain
$PY -u run_transcribe.py --set dev --model small --mode personal --style list --nwords 40
$PY -u run_transcribe.py --set dev --model small --mode personal --style list --nwords 15
$PY -u run_transcribe.py --set dev --model small --mode personal --style sentence --nwords 40
$PY -u run_transcribe.py --set dev --model small --mode personal --style sentence --nwords 15
echo DEV_DONE
