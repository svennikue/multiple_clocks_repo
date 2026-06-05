#!/usr/bin/env bash
# helper_prep_delete_useless_feat_glm_dirs.sh — build a safe plan from feat_scan_summary.csv
set -euo pipefail

CSV="${1:-feat_scan_summary.csv}"
PLAN="feat_tidy_plan.sh"
: > "$PLAN"

declare -A best_count best_src worked_family failed_set

# parse CSV (skip header)
while IFS=, read -r subject path count status; do
  [[ "$subject" == "subject" ]] && continue
  base="${path##*/}"
  nobase="${base//+/}"                               # strip all +
  family="${path%/*}/$nobase"                        # canonical path (no +)

  if [[ "$status" == "failed" ]]; then
    failed_set["$path"]=1
  elif [[ "$status" == "worked" ]]; then
    worked_family["$path"]="$family"
    if [[ -z "${best_count[$family]+x}" || "$count" -gt "${best_count[$family]}" ]]; then
      best_count["$family"]="$count"
      best_src["$family"]="$path"
    fi
  fi
done < <(tail -n +2 "$CSV")

# 1) delete all failed
for p in "${!failed_set[@]}"; do
  printf "rm -rf -- %q\n" "$p" >> "$PLAN"
done

# 2) delete non-winning worked duplicates
for p in "${!worked_family[@]}"; do
  fam="${worked_family[$p]}"
  if [[ "${best_src[$fam]}" != "$p" ]]; then
    printf "rm -rf -- %q\n" "$p" >> "$PLAN"
  fi
done

# 3) rename each winning worked to canonical no-plus name
for fam in "${!best_src[@]}"; do
  src="${best_src[$fam]}"
  if [[ "$src" != "$fam" ]]; then
    printf "rm -rf -- %q\n" "$fam" >> "$PLAN"   # ensure target is clear
    printf "mv -- %q %q\n" "$src" "$fam" >> "$PLAN"
  fi
done

chmod +x "$PLAN"
echo "Plan written to: $PLAN"
echo "Review first (e.g., head -n 30 $PLAN), then run it when happy:"
echo "  ./$PLAN"
