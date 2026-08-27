#!/bin/bash
# =============================================================================
# Download LFP data from Box straight into the pipeline's folder layout.
#
# Box folders are named `s{NN}_{SITE}_{CODE}_{day}` (s10_BCM_YEL_day1,
# s17_UT202306), so the session number is already in the folder name and each
# one can be copied directly to its destination -- no flat download, no
# sorting step.
#
#   Box  s18_BCM_YER_day1/*.ns3   ->   <ROOT>/s18/LFP/
#        s03_.../*.ncs            ->   <ROOT>/s03/micros_and_macros/
#        s..._/*.mat              ->   <ROOT>/s..../electrodes/
#
# rclone copy SKIPS files already present and is resumable, so re-running
# after a partial download only fetches what is missing. Safe to re-run.
#
# DRY RUN BY DEFAULT.  Set EXECUTE=1 to actually transfer.
#
# Usage:
#   bash scripts/swr_download_box.sh                 # dry run, see the plan
#   EXECUTE=1 bash scripts/swr_download_box.sh       # do it
#   EXECUTE=1 ONLY="s18 s19" bash scripts/swr_download_box.sh
#   REMOTE=ucla_box: FID="" EXECUTE=1 bash scripts/swr_download_box.sh   # UCLA
#
# @author: Svenja Kuchenhoff
# =============================================================================
set -uo pipefail

# ---- configure ---------------------------------------------------------------
RC="${RC:-./rclone-v1.70.1-linux-amd64/rclone}"
REMOTE="${REMOTE:-baylor_box:}"
FID="${FID:---box-root-folder-id 284057243982}"      # set FID="" if not needed
ROOT="${ROOT:-/ceph/behrens/svenja/human_ABCD_ephys}"
EXECUTE="${EXECUTE:-0}"
ONLY="${ONLY:-}"                                     # e.g. ONLY="s18 s19"
LOG="${LOG:-$ROOT/box_download_$(date +%Y-%m-%d_%H-%M-%S).log}"

# Folders to skip, with the reason.
#   s47_UT202311 : patient 202311, but the manifest says s47 is UT202302.
#                  `s47new_UT202302` is the correct s47; this folder looks like
#                  a duplicate of s48. Verify before using it.
#   s6?_...      : literal '?' in the name. Handled by ALIAS below rather than
#                  skipped, so no data is lost -- the recording-length vs
#                  behaviour gate in swr_extract_continuous will reject it
#                  loudly if the assignment is wrong.
SKIP_FOLDERS="${SKIP_FOLDERS:-s47_UT202311}"

# folder name -> session number, for folders whose name does not parse.
# Set ALIAS_SESSION to the intended session before running with EXECUTE=1.
ALIAS_FOLDER="${ALIAS_FOLDER:-s6?_BCM_YFP_day2?}"
ALIAS_SESSION="${ALIAS_SESSION:-}"        # e.g. ALIAS_SESSION=61

# ---- checks ------------------------------------------------------------------
[ -x "$RC" ] || { echo "rclone not found/executable at: $RC"; exit 1; }
mkdir -p "$(dirname "$LOG")" 2>/dev/null

echo "remote : $REMOTE $FID"
echo "root   : $ROOT"
echo "mode   : $([ "$EXECUTE" = "1" ] && echo 'EXECUTE' || echo 'DRY RUN (set EXECUTE=1 to transfer)')"
echo "log    : $LOG"
echo "skip   : $SKIP_FOLDERS"
echo "==============================================================================="

# ---- enumerate Box folders ---------------------------------------------------
# portable alternative to `mapfile` (absent on bash 3.2, e.g. macOS)
FOLDERS=()
while IFS= read -r _line; do
    [ -n "$_line" ] && FOLDERS+=("$_line")
done < <($RC lsf --dirs-only $FID "$REMOTE" 2>/dev/null | sed 's:/*$::')
[ "${#FOLDERS[@]}" -gt 0 ] || { echo "no folders returned -- check remote/FID"; exit 1; }
echo "found ${#FOLDERS[@]} folders at the remote"
echo

n_ok=0; n_skip=0; n_bad=0
for f in "${FOLDERS[@]}"; do
    # session number must be the leading s<digits> of the folder name
    if [ -n "$ALIAS_SESSION" ] && [ "$f" = "$ALIAS_FOLDER" ]; then
        sess=$(printf "%02d" "$ALIAS_SESSION")
        echo "  [s$sess ] $f   (ALIAS -- verify with the alignment gate)"
    elif [[ "$f" =~ ^s([0-9]+)([A-Za-z]*)_ ]]; then
        sess=$(printf "%02d" "${BASH_REMATCH[1]}")
    else
        echo "  [SKIP ] $f   (no parseable session number)"
        n_bad=$((n_bad+1)); continue
    fi

    if [[ " $SKIP_FOLDERS " == *" $f "* ]]; then
        echo "  [SKIP ] $f   (on the skip list)"
        n_skip=$((n_skip+1)); continue
    fi
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" s${BASH_REMATCH[1]} "* ]]; then
        continue
    fi

    echo "  [s$sess ] $f"
    n_ok=$((n_ok+1))

    # extension group -> destination subdirectory
    for spec in "LFP:*.ns2 *.ns3 *.ns5 *.nev" \
                "micros_and_macros:*.ncs" \
                "electrodes:*.mat"; do
        sub="${spec%%:*}"; pats="${spec#*:}"
        inc=(); for pat in $pats; do inc+=(--include "$pat"); done
        dest="$ROOT/s$sess/$sub"

        # is there anything of this kind in the folder? (cheap, avoids empty dirs)
        cnt=$($RC lsf -R --files-only $FID "${inc[@]}" "$REMOTE$f" 2>/dev/null | wc -l)
        [ "$cnt" -eq 0 ] && continue
        echo "        $cnt file(s) -> $dest"

        if [ "$EXECUTE" = "1" ]; then
            mkdir -p "$dest"
            $RC copy --progress --transfers 8 --checkers 16 \
                 --log-file "$LOG" --log-level INFO \
                 $FID "${inc[@]}" "$REMOTE$f" "$dest"
        fi
    done
done

echo
echo "==============================================================================="
echo "  folders to download : $n_ok"
echo "  skipped (list)      : $n_skip"
echo "  unparseable         : $n_bad"
if [ "$EXECUTE" != "1" ]; then
    echo
    echo "  DRY RUN — nothing transferred. Re-run with:  EXECUTE=1 bash $0"
else
    echo
    echo "  done. Now verify and check the pipeline can see it:"
    echo "    bash scripts/swr_verify_box.sh"
    echo "    python scripts/swr_check_inputs.py"
fi
