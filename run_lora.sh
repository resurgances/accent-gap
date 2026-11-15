#!/bin/bash
# E4 (stretch): LoRA fine-tune on DEV speakers only, then transcribe the eval clips with it.
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
[ -f ../lora_out/adapter_config.json ] || $PY -u lora_finetune.py || { echo FAILED_TRAIN; exit 1; }
$PY -u run_transcribe.py --model small-lora --mode plain || echo FAILED_TRANSCRIBE
echo LORA_DONE
