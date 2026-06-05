# #!/usr/bin/env bash
# # quick check for FEAT runs: classifies glm_all*/glm_fut* as worked/failed by file count

root="/home/fs0/xpsy1114/scratch/data/derivatives"
first=1
last=35

# thresh=10
# prefix="feat_scan"

# summary="${prefix}_summary.csv"
# worked="${prefix}_worked_dirs.txt"
# failed="${prefix}_failed_dirs.txt"
# delsh="${prefix}_delete_failed_dirs.sh"

# echo "subject,path,file_count,status" > "$summary"
# : > "$worked"
# : > "$failed"

# for s in $(seq -w $first $last); do
#   func="$root/sub-$s/func"
#   [ -d "$func" ] || continue
#   for d in "$func"/glm_all*.feat "$func"/glm_fut*.feat; do
#     [ -d "$d" ] || continue
#     n=$(find "$d" -type f | wc -l)
#     if [ "$n" -ge "$thresh" ]; then
#       status="worked"; echo "$d" >> "$worked"
#     else
#       status="failed"; echo "$d" >> "$failed"
#     fi
#     echo "sub-$s,$d,$n,$status" >> "$summary"
#   done
# done

# # make a review-first deletion script
# awk '{print "# rm -rf -- \""$0"\""}' "$failed" > "$delsh"
# chmod +x "$delsh"

# echo "Wrote: $summary, $worked, $failed, $delsh"



out="feat_presence_with_counts.txt"

shopt -s nullglob

# Build subject list
subs=()
for s in $(seq -w $first $last); do subs+=("sub-$s"); done

# Gather unique folder base names seen anywhere
tmp_names=$(mktemp)
for s in "${subs[@]}"; do
  func="$root/$s/func"
  [[ -d "$func" ]] || continue
  for d in "$func"/glm_all*.feat "$func"/glm_fut*.feat; do
    [[ -d "$d" ]] || continue
    basename "${d}" >> "$tmp_names"
  done
done

# Sort unique names
mapfile -t names < <(sort -u "$tmp_names")
rm -f "$tmp_names"

: > "$out"

# For each unique name, check every subject
for name in "${names[@]}"; do
  missing=()
  present=()
  for s in "${subs[@]}"; do
    d="$root/$s/func/$name"
    if [[ -d "$d" ]]; then
      n=$(find "$d" -type f | wc -l)
      present+=("$s($n)")
    else
      missing+=("$s")
    fi
  done

  if ((${#missing[@]}==0)); then
    printf "%s: present: %s\n" \
      "$name" "$(IFS=', '; printf '%s' "${present[*]}")" | tee -a "$out"
  else
    printf "%s: missing: %s ; present: %s\n" \
      "$name" "$(IFS=', '; printf '%s' "${missing[*]}")" \
      "$(IFS=', '; printf '%s' "${present[*]}")" | tee -a "$out"
  fi
done

echo
echo "Saved to: $out"
