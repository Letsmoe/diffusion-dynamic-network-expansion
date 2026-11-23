#!/bin/bash
cd /home/moritz/Documents/personal/scratch
PY=/home/moritz/.venvs/hobby/bin/python
run() {
  name=$1; shift
  env "$@" EVAL_OUT=sweep_$name.json $PY -I eval_refine.py 164 > sweep_$name.log 2>&1
}
run steps32      STEPS=32  GEN_LEN=256 REMASK=0.3
run steps256     STEPS=256 GEN_LEN=256 REMASK=0.3
run remask01     STEPS=128 GEN_LEN=256 REMASK=0.1
run gen512_s256  STEPS=256 GEN_LEN=512 REMASK=0.3
echo ALLDONE > sweep_done.txt
