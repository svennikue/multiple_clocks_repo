#!/bin/bash
# =============================================================================
# Copy the UCLA sessions out of all_box_data into the pipeline layout.
#
# These six folders are NOT homogeneous, so this cannot be a one-liner:
#
#   folder  session  task dir                 format      note
#   559     s03      EXP3_Baylor_ABCD_v3      Neuralynx
#   573     s56      EXP12_ABCD               Neuralynx   also holds p579_* (not ours)
#   576     s50      EXP3_ABCD                BLACKROCK   config says ncs -- wrong
#   577     s40      EXP3_ABCD                Neuralynx   4 recording blocks
#   578     s51      EXP3_ABCD                BLACKROCK   also has EXP2_*_practice
#   582     s60      EXP5_ABCD                Neuralynx   also has EXP4_*_Practice
#
# The task directory is NOT always EXP3. The only reliable rule is: the EXP*
# directory that is not a practice one. Practice data must never land in a
# session folder -- it would be analysed as task data.
#
#   *.ncs          -> <ROOT>/s{NN}/micros_and_macros/
#   *.ns3 *.ns5 *.nev -> <ROOT>/s{NN}/LFP/
#   *localizations*.xlsx, *Localizations*.xlsx -> <ROOT>/s{NN}/electrodes/
#
# DRY RUN BY DEFAULT. Set EXECUTE=1 to copy.
#
# Usage:
#   bash scripts/swr_copy_ucla.sh
#   EXECUTE=1 bash scripts/swr_copy_ucla.sh
#
# @author: Svenja Kuchenhoff
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/ceph/behrens/svenja/human_ABCD_ephys}"
BOX="${BOX:-$ROOT/all_box_data/Baylor_ABCD}"
EXECUTE="${EXECUTE:-0}"

# patient -> session (from session_manifest.csv subject_label UC*-0NNN)
# Portable lookup: `declare -A` needs bash 4, and macOS ships 3.2.
PATIENTS=(559 573 576 577 578 582)
session_for() {
    case "$1" in
        559) echo "03" ;;  573) echo "56" ;;  576) echo "50" ;;
        577) echo "40" ;;  578) echo "51" ;;  582) echo "60" ;;
        *)   echo "" ;;
    esac
}

# Directories inside a patient folder that must never be copied.
#   *practice*        : EXP2_ABCD_practice (578), EXP4_ABCD_Practice (582)
#   p*_manual_sorting : p579_manual_sorting_JS sits inside 573_ABCD and belongs
#                       to a different patient
EXCLUDE_RE='practice|manual_sorting'

echo "box  : $BOX"
echo "root : $ROOT"
echo "mode : $([ "$EXECUTE" = "1" ] && echo EXECUTE || echo 'DRY RUN (EXECUTE=1 to copy)')"
echo "==============================================================================="

for pat in "${PATIENTS[@]}"; do
    s="$(session_for "$pat")"
    [ -n "$s" ] || { echo "  [$pat] no session mapping -- skipped"; continue; }
    src="$BOX/${pat}_ABCD"
    if [ ! -d "$src" ]; then
        printf "  [%s] s%s  MISSING: %s\n" "$pat" "$s" "$src"; continue
    fi

    # the task directory: an EXP* dir that is not practice / not another patient
    task=""
    while IFS= read -r d; do
        base=$(basename "$d")
        echo "$base" | grep -Eqi "$EXCLUDE_RE" && continue
        task="$d"; break
    done < <(find "$src" -maxdepth 1 -mindepth 1 -type d -name 'EXP*' | sort)

    if [ -z "$task" ]; then
        printf "  [%s] s%s  no non-practice EXP* directory found\n" "$pat" "$s"; continue
    fi

    n_ncs=$(find "$task" -type f -name '*.ncs' 2>/dev/null | wc -l)
    n_ncs_data=$(find "$task" -type f -name '*.ncs' -size +17k 2>/dev/null | wc -l)
    n_nsx=$(find "$task" -type f \( -name '*.ns3' -o -name '*.ns5' \) 2>/dev/null | wc -l)
    fmt="unknown"
    [ "$n_ncs" -gt 0 ] && fmt="neuralynx"
    [ "$n_nsx" -gt 0 ] && fmt="BLACKROCK"

    printf "  [%s] s%s  %-24s %-10s ncs=%d (data %d)  nsx=%d\n" \
           "$pat" "$s" "$(basename "$task")" "$fmt" "$n_ncs" "$n_ncs_data" "$n_nsx"

    # excluded siblings, listed so nothing is dropped silently
    while IFS= read -r d; do
        b=$(basename "$d")
        echo "$b" | grep -Eqi "$EXCLUDE_RE" && printf "         skipping: %s\n" "$b"
    done < <(find "$src" -maxdepth 1 -mindepth 1 -type d | sort)

    if [ "$EXECUTE" = "1" ]; then
        if [ "$n_ncs" -gt 0 ]; then
            mkdir -p "$ROOT/s$s/micros_and_macros"
            find "$task" -type f -name '*.ncs' -exec cp -n {} "$ROOT/s$s/micros_and_macros/" \;
        fi
        if [ "$n_nsx" -gt 0 ]; then
            mkdir -p "$ROOT/s$s/LFP"
            find "$task" -type f \( -name '*.ns3' -o -name '*.ns5' -o -name '*.nev' \) \
                 -exec cp -n {} "$ROOT/s$s/LFP/" \;
        fi
        # localisation spreadsheets live at the patient level, not in EXP*
        find "$src" -maxdepth 1 -type f -iname '*localizations*.xlsx' | while read -r x; do
            mkdir -p "$ROOT/s$s/electrodes"; cp -n "$x" "$ROOT/s$s/electrodes/"
        done
    fi
done

echo
echo "==============================================================================="
if [ "$EXECUTE" != "1" ]; then
    echo "  DRY RUN — nothing copied. Re-run with:  EXECUTE=1 bash $0"
else
    echo "  done. Verify:"
    echo "    python scripts/swr_check_inputs.py"
    echo
    echo "  NOTE s50 and s51 are Blackrock, but config_human_ABCD_iEEG.yaml says"
    echo "  'LFP_file_format: ncs'. swr_io.discover_raw_files now detects the format"
    echo "  from disk and warns, so the pipeline copes -- but fixing the YAML is tidier."
fi
