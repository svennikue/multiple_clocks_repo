#!/usr/bin/env bash

BASE="~/scratch/data/derivatives"

for n in $(seq -w 01 35); do
  subj="sub-$n"
  func="$BASE/$subj/func"
  [ -d "$func" ] || continue

  for d in "$func"/glm*; do
    [ -d "$d" ] || continue

    # count non-recursive files in that folder
    c=$(find "$d" -maxdepth 1 -type f | wc -l)

    if [ "$c" -lt 10 ]; then
      echo "$subj  -  <10 files  -  $c  -  $d"
    fi

    case "$d" in
      *glm_fut-steps*|*glm_all*)
        if [ "$c" -gt 10 ]; then
          echo "$subj  -  >10 special  -  $c  -  $d"
        fi
        ;;
    esac
  done
done
