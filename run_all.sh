#!/bin/bash
# Runs every Whisper transcription job in priority order. Resumable: re-run to continue.
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
for job in "small plain" "small personal" "small control" "base plain" "base personal" "medium plain" "medium personal"; do
  set -- $job
  $PY -u run_transcribe.py --model $1 --mode $2 || echo "FAILED $1 $2"
done
echo ALL_DONE
