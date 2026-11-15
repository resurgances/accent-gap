#!/bin/bash
# Runs the FINAL personal-mode design (chosen on DEV speakers, see results/chosen_design.json)
# on the evaluation clips: personal + control, for small first, then base and medium.
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
STYLE=$($PY -c "import json;print(json.load(open('../results/chosen_design.json'))['style'])")
N=$($PY -c "import json;print(json.load(open('../results/chosen_design.json'))['nwords'])")
echo "final design: style=$STYLE nwords=$N"
for m in small base medium; do
  for mode in personal control; do
    $PY -u run_transcribe.py --model $m --mode $mode --style $STYLE --nwords $N || echo "FAILED $m $mode"
  done
done
echo FINAL_DONE
