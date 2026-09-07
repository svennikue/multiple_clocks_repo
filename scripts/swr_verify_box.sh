#!/bin/bash
# =============================================================================
# Verify the Box download BY SIZE, and re-fetch what is wrong.
#
# Answers "did I download everything correctly", which `rclone copy` alone does
# not tell you -- it exits 0 whether it transferred everything or nothing.
#
# WHY THIS COMPARES BYTES, NOT FILE COUNTS
# ----------------------------------------
# The previous version of this script counted files. That misses the failure
# mode that actually happened: an interrupted transfer leaves the right NUMBER
# of files, each truncated. s40 read as `match` with 505 .ncs files present
# while holding 12 GB of an expected ~21 GB, and nothing downstream noticed --
# `session_block_table` reads the duration from the FIRST file of a group only,
# so a short tail across the other 168 is invisible to the audit.
#
# It also only looked at `$ROOT/s{NN}/{LFP,micros_and_macros,electrodes}` with
# `-maxdepth 1`, so sessions downloaded with the remote's own subfolder layout
# (s61-s63, which nest as `LFP/EMU-.../`) read as NOT DOWNLOADED.
#
# Files are matched by BASENAME anywhere under `$ROOT/s{NN}`, so both layouts
# verify correctly.
#
# Usage:
#   bash scripts/swr_verify_box.sh                     # report only
#   ONLY="s16 s40 s63" bash scripts/swr_verify_box.sh  # just these
#   FIX=1 ONLY="s16 s40 s63" bash scripts/swr_verify_box.sh   # re-fetch the bad ones
#
# FIX=1 deletes every local file whose size disagrees with Box and re-fetches
# it together with anything missing. It never touches a file that matches.
#
# @author: Svenja Kuchenhoff
# =============================================================================
set -uo pipefail

RC="${RC:-./rclone-v1.70.1-linux-amd64/rclone}"
REMOTE="${REMOTE:-baylor_box:}"
FID="${FID:---box-root-folder-id 284057243982}"
ROOT="${ROOT:-/ceph/behrens/svenja/human_ABCD_ephys}"
ONLY="${ONLY:-}"
FIX="${FIX:-0}"
SKIP_FOLDERS="${SKIP_FOLDERS:-s47_UT202311}"
LOG="${LOG:-$ROOT/box_refetch_$(date +%Y-%m-%d_%H-%M-%S).log}"
WORK="${WORK:-$(mktemp -d)}"

# rclone spawns many OS threads; the SWC login nodes cap them and it aborts with
# `pthread_create failed`. Harmless on a compute node, necessary on hpc-gw*.
export GOMAXPROCS="${GOMAXPROCS:-2}"

[ -x "$RC" ] || { echo "rclone not found at: $RC"; exit 1; }

EXTS=(ns2 ns3 ns5 nev ncs mat)
inc=(); for e in "${EXTS[@]}"; do inc+=(--include "*.$e"); done
find_expr=(); for e in "${EXTS[@]}"; do find_expr+=(-o -name "*.$e"); done
find_expr=("${find_expr[@]:1}")          # drop the leading -o

echo "remote : $REMOTE $FID"
echo "root   : $ROOT"
echo "mode   : $([ "$FIX" = "1" ] && echo 'FIX (will delete bad files and re-fetch)' || echo 'REPORT ONLY (set FIX=1 to re-fetch)')"
echo "work   : $WORK"
echo
printf "%-30s %-6s %8s %8s %8s %8s   %s\n" "BOX FOLDER" "SESS" "REMOTE" "OK" "MISSING" "TRUNC" "STATUS"
printf "%.0s-" {1..104}; echo

FOLDERS=()
while IFS= read -r _line; do
    [ -n "$_line" ] && FOLDERS+=("$_line")
done < <($RC lsf --dirs-only $FID "$REMOTE" 2>/dev/null | sed 's:/*$::')
[ "${#FOLDERS[@]}" -gt 0 ] || { echo "no folders returned -- check remote/FID"; exit 1; }

tot_ok=0; tot_bad=0; tot_none=0; BAD_SESSIONS=(); UNPARSED=()

for f in "${FOLDERS[@]}"; do
    # Never skip silently: `s6?_BCM_YFP_day2?` does not parse, and quietly
    # dropping it is how a folder goes unchecked without anyone noticing.
    if ! [[ "$f" =~ ^s([0-9]+)([A-Za-z]*)_ ]]; then
        printf "%-30s %-6s %8s %8s %8s %8s   %s\n" \
               "$f" "-" "-" "-" "-" "-" "NOT CHECKED (no session number in name)"
        UNPARSED+=("$f"); continue
    fi
    snum="${BASH_REMATCH[1]}"
    sess=$(printf "%02d" "$snum")
    [[ " $SKIP_FOLDERS " == *" $f "* ]] && continue
    [ -n "$ONLY" ] && [[ " $ONLY " != *" s${snum} "* ]] && [[ " $ONLY " != *" s${sess} "* ]] && continue

    # ---- remote: "size|relative/path" ---------------------------------------
    $RC lsf -R --files-only --format "sp" --separator "|" $FID "${inc[@]}" \
        "$REMOTE$f" > "$WORK/remote_$sess.txt" 2>/dev/null
    n_remote=$(wc -l < "$WORK/remote_$sess.txt")
    [ "$n_remote" -eq 0 ] && continue

    # ---- local: "size|absolute/path", anywhere under the session dir --------
    # The sentinel line is load-bearing: awk's NR==FNR two-file idiom breaks if
    # the first file is empty (FNR restarts, so every REMOTE line is read as if
    # it were local and the session reports a clean match while holding nothing
    # at all). Seeding one unmatchable line keeps the first file non-empty.
    printf '0|/__no_local_files__\n' > "$WORK/local_$sess.txt"
    [ -d "$ROOT/s$sess" ] && find "$ROOT/s$sess" -type f \( "${find_expr[@]}" \) \
        -printf '%s|%p\n' 2>/dev/null >> "$WORK/local_$sess.txt"

    # ---- compare by basename + size ----------------------------------------
    awk -F'|' -v miss="$WORK/missing_$sess.txt" -v trunc="$WORK/trunc_$sess.txt" \
              -v del="$WORK/delete_$sess.txt" '
        NR == FNR {
            n = split($2, parts, "/"); bn = parts[n]
            if (!(bn in lsize) || $1 > lsize[bn]) { lsize[bn] = $1; lpath[bn] = $2 }
            next
        }
        {
            n = split($2, parts, "/"); bn = parts[n]
            if (!(bn in lsize))       { print $2 > miss;  m++ }
            else if (lsize[bn] != $1) { print $2 > trunc; print lpath[bn] > del; t++ }
            else                      { k++ }
        }
        END { printf "%d %d %d\n", k+0, m+0, t+0 }
    ' "$WORK/local_$sess.txt" "$WORK/remote_$sess.txt" > "$WORK/counts_$sess.txt"
    n_ok=0; n_missing=0; n_trunc=0
    read -r n_ok n_missing n_trunc < "$WORK/counts_$sess.txt"

    if   [ "$n_ok" -eq 0 ] && [ "$n_missing" -eq "$n_remote" ]; then
        status="NOT DOWNLOADED"; tot_none=$((tot_none+1)); BAD_SESSIONS+=("$sess:$f")
    elif [ "$n_missing" -eq 0 ] && [ "$n_trunc" -eq 0 ]; then
        status="match"; tot_ok=$((tot_ok+1))
    else
        status="INCOMPLETE"; tot_bad=$((tot_bad+1)); BAD_SESSIONS+=("$sess:$f")
    fi
    printf "%-30s %-6s %8s %8s %8s %8s   %s\n" \
           "$f" "s$sess" "$n_remote" "$n_ok" "$n_missing" "$n_trunc" "$status"
done

echo
printf "%.0s-" {1..104}; echo
echo "  complete       : $tot_ok"
echo "  incomplete     : $tot_bad"
echo "  not downloaded : $tot_none"
[ "${#UNPARSED[@]}" -gt 0 ] && {
    echo "  NOT CHECKED    : ${#UNPARSED[@]}  -> ${UNPARSED[*]}"
    echo "                   (folder name has no leading s<NN>_; check these by hand)"
}
echo
echo "  NOTE: only folders under this remote/FID were seen. UCLA sessions live on"
echo "        a different Box share -- re-run with REMOTE=... FID=... to cover them."

[ "${#BAD_SESSIONS[@]}" -eq 0 ] && { echo; echo "  nothing to re-fetch."; exit 0; }

echo
echo "  sessions needing a re-fetch: ${BAD_SESSIONS[*]%%:*}"
for entry in "${BAD_SESSIONS[@]}"; do
    sess="${entry%%:*}"; f="${entry#*:}"
    echo
    echo "  --- s$sess ($f)"
    [ -s "$WORK/missing_$sess.txt" ] && { echo "      missing:"; head -3 "$WORK/missing_$sess.txt" | sed 's/^/        /'; }
    [ -s "$WORK/trunc_$sess.txt" ]   && { echo "      truncated:"; head -3 "$WORK/trunc_$sess.txt" | sed 's/^/        /'; }
done

if [ "$FIX" != "1" ]; then
    echo
    echo "  REPORT ONLY -- nothing changed. Re-fetch with:"
    echo "    FIX=1 ONLY=\"$(for e in "${BAD_SESSIONS[@]}"; do printf 's%s ' "${e%%:*}"; done)\" bash $0"
    exit 0
fi

# =============================================================================
# FIX: delete the truncated files, then re-fetch missing + truncated.
# =============================================================================
echo
echo "==============================================================================="
echo "  RE-FETCHING   log: $LOG"
echo "==============================================================================="

for entry in "${BAD_SESSIONS[@]}"; do
    sess="${entry%%:*}"; f="${entry#*:}"
    echo
    echo "  --- s$sess ($f)"

    if [ -s "$WORK/delete_$sess.txt" ]; then
        n_del=$(wc -l < "$WORK/delete_$sess.txt")
        echo "      deleting $n_del truncated file(s)"
        while IFS= read -r p; do [ -f "$p" ] && rm -f "$p"; done < "$WORK/delete_$sess.txt"
    fi

    cat "$WORK/missing_$sess.txt" "$WORK/trunc_$sess.txt" 2>/dev/null \
        | sort -u > "$WORK/fetch_$sess.txt"
    n_fetch=$(wc -l < "$WORK/fetch_$sess.txt")
    [ "$n_fetch" -eq 0 ] && { echo "      nothing to fetch"; continue; }

    # Destination. If the Box folder has subdirectories, mirror them under the
    # session root -- that is the layout s61-s63 already have, and duplicating
    # a .ns3 into a second path would make the extractor see a phantom extra
    # recording block. If the remote is flat, route by extension the way
    # swr_download_box.sh does.
    has_subdirs=$($RC lsf --dirs-only $FID "$REMOTE$f" 2>/dev/null | head -1)
    if [ -n "$has_subdirs" ]; then
        dest="$ROOT/s$sess"
        echo "      $n_fetch file(s) -> $dest  (mirroring remote layout)"
        mkdir -p "$dest"
        $RC copy --progress --transfers 4 --checkers 8 --multi-thread-streams 0 \
             --log-file "$LOG" --log-level INFO \
             $FID --files-from "$WORK/fetch_$sess.txt" "$REMOTE$f" "$dest"
    else
        for spec in "LFP:ns2 ns3 ns5 nev" "micros_and_macros:ncs" "electrodes:mat"; do
            sub="${spec%%:*}"; exts="${spec#*:}"
            pat=$(echo "$exts" | tr ' ' '|')
            grep -E "\.($pat)$" "$WORK/fetch_$sess.txt" > "$WORK/fetch_${sess}_$sub.txt" || true
            n_sub=$(wc -l < "$WORK/fetch_${sess}_$sub.txt")
            [ "$n_sub" -eq 0 ] && continue
            dest="$ROOT/s$sess/$sub"
            echo "      $n_sub file(s) -> $dest"
            mkdir -p "$dest"
            $RC copy --progress --transfers 4 --checkers 8 --multi-thread-streams 0 \
                 --log-file "$LOG" --log-level INFO \
                 $FID --files-from "$WORK/fetch_${sess}_$sub.txt" "$REMOTE$f" "$dest"
        done
    fi
done

echo
echo "==============================================================================="
echo "  done. Re-verify, then re-run the audit:"
echo "    ONLY=\"$(for e in "${BAD_SESSIONS[@]}"; do printf 's%s ' "${e%%:*}"; done)\" bash $0"
echo "    python scripts/swr_audit_sessions.py"
