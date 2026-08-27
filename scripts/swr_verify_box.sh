#!/bin/bash
# =============================================================================
# Verify the Box download: per session, does what is on disk match the remote?
#
# Answers "did I download everything correctly", which `rclone copy` alone does
# not tell you -- it exits 0 whether it transferred everything or nothing.
#
# For each Box folder it compares the remote file list against the local
# destination and reports MATCH / MISSING n / EXTRA n. `rclone check` is used
# for the authoritative comparison (size + hash where Box provides one).
#
# Usage:
#   bash scripts/swr_verify_box.sh
#   ONLY="s18 s19" bash scripts/swr_verify_box.sh
#   DEEP=1 bash scripts/swr_verify_box.sh      # full rclone check (slower)
#
# @author: Svenja Kuchenhoff
# =============================================================================
set -uo pipefail

RC="${RC:-./rclone-v1.70.1-linux-amd64/rclone}"
REMOTE="${REMOTE:-baylor_box:}"
FID="${FID:---box-root-folder-id 284057243982}"
ROOT="${ROOT:-/ceph/behrens/svenja/human_ABCD_ephys}"
ONLY="${ONLY:-}"
DEEP="${DEEP:-0}"
SKIP_FOLDERS="${SKIP_FOLDERS:-s47_UT202311}"

[ -x "$RC" ] || { echo "rclone not found at: $RC"; exit 1; }

printf "%-30s %-6s %8s %8s   %s\n" "BOX FOLDER" "SESS" "REMOTE" "LOCAL" "STATUS"
printf "%.0s-" {1..86}; echo

# portable alternative to `mapfile` (absent on bash 3.2, e.g. macOS)
FOLDERS=()
while IFS= read -r _line; do
    [ -n "$_line" ] && FOLDERS+=("$_line")
done < <($RC lsf --dirs-only $FID "$REMOTE" 2>/dev/null | sed 's:/*$::')
tot_ok=0; tot_bad=0; tot_none=0

for f in "${FOLDERS[@]}"; do
    [[ "$f" =~ ^s([0-9]+)([A-Za-z]*)_ ]] || continue
    sess=$(printf "%02d" "${BASH_REMATCH[1]}")
    [[ " $SKIP_FOLDERS " == *" $f "* ]] && continue
    [ -n "$ONLY" ] && [[ " $ONLY " != *" s${BASH_REMATCH[1]} "* ]] && continue

    inc=(--include "*.ns2" --include "*.ns3" --include "*.ns5" --include "*.nev"
         --include "*.ncs" --include "*.mat")

    n_remote=$($RC lsf -R --files-only $FID "${inc[@]}" "$REMOTE$f" 2>/dev/null | wc -l)
    [ "$n_remote" -eq 0 ] && continue

    n_local=0
    for sub in LFP micros_and_macros electrodes; do
        d="$ROOT/s$sess/$sub"
        [ -d "$d" ] && n_local=$((n_local + $(find "$d" -maxdepth 1 -type f \
            \( -name '*.ns2' -o -name '*.ns3' -o -name '*.ns5' -o -name '*.nev' \
               -o -name '*.ncs' -o -name '*.mat' \) 2>/dev/null | wc -l)))
    done

    if [ "$n_local" -eq 0 ]; then
        status="NOT DOWNLOADED"; tot_none=$((tot_none+1))
    elif [ "$n_local" -eq "$n_remote" ]; then
        status="match"; tot_ok=$((tot_ok+1))
    elif [ "$n_local" -lt "$n_remote" ]; then
        status="INCOMPLETE  missing $((n_remote-n_local))"; tot_bad=$((tot_bad+1))
    else
        status="extra $((n_local-n_remote)) local file(s)"; tot_ok=$((tot_ok+1))
    fi

    printf "%-30s %-6s %8s %8s   %s\n" "$f" "s$sess" "$n_remote" "$n_local" "$status"

    # authoritative comparison, size + hash
    if [ "$DEEP" = "1" ] && [ "$n_local" -gt 0 ]; then
        $RC check $FID --include "*.ns2" --include "*.ns3" --include "*.ncs" \
            --include "*.mat" --one-way \
            "$REMOTE$f" "$ROOT/s$sess/LFP" 2>&1 | grep -E "ERROR|NOTICE" | head -5
    fi
done

echo
printf "%.0s-" {1..86}; echo
echo "  complete       : $tot_ok"
echo "  incomplete     : $tot_bad     <- re-run the download; rclone fetches only what is missing"
echo "  not downloaded : $tot_none"
echo
echo "  file counts only. For a byte-level check of one session:"
echo "    $RC check $FID $REMOTE<folder> $ROOT/s<NN>/LFP --one-way"
echo "  or run this script with DEEP=1"
