import os
import mc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

# -----------------------------------------------------------------------
# Build neuron.csv
#   One row per (session × neuron × config).
#   Timecourse = mean over correct repeats, averaged across all runs of
#   that config within the session.
#   Columns: session, neuron_label, roi, config, bin_000 … bin_359
# -----------------------------------------------------------------------

DATA_FOLDER  = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
NEURON_CSV   = os.path.join(DATA_FOLDER, "all_neurons_avg_per_gridrun.csv")

BIN_COLS = [f"bin_{b:03d}" for b in range(360)]

neuron_rows = []

for sub in range(1, 64):
    sub_str  = f"{sub:02d}"
    sub_key  = f"sub-{sub_str}"

    # -- configs file (maps grid_no → config tuple) ----------------------
    configs_path = os.path.join(DATA_FOLDER, f"s{sub_str}", "cells_and_beh",
                                f"all_configs_sub{sub_str}.csv")
    if not os.path.exists(configs_path):
        print(f"s{sub_str}: configs file missing, skipping.")
        continue

    configs_arr = np.genfromtxt(configs_path, delimiter=',').astype(int)
    if configs_arr.ndim == 1:
        configs_arr = configs_arr.reshape(1, -1)
    # grid_no is 1-indexed
    grid_to_config = {i + 1: tuple(configs_arr[i]) for i in range(len(configs_arr))}

    # -- load full session data ------------------------------------------
    data_dict, source_path = mc.analyse.helpers_human_cells.get_data(sub)
    if sub_key not in data_dict:
        print(f"s{sub_str}: data not found, skipping.")
        continue

    beh         = data_dict[sub_key]['beh']          # (n_trials, 14)
    cell_labels = data_dict[sub_key]['cell_labels']  # list of ROI strings
    norm        = data_dict[sub_key]['normalised_neurons']  # {key: DataFrame (n_trials, 360)}

    # sanity: n_neurons should match n_keys
    neuron_keys = list(norm.keys())
    if len(neuron_keys) != len(cell_labels):
        print(f"s{sub_str}: WARNING neuron keys ({len(neuron_keys)}) ≠ cell_labels ({len(cell_labels)})")

    


    # group trial indices by grid_no (one run = one grid_no block)
    # average only correct repeats within each grid_no run
    gridno_to_trials = defaultdict(list)
    for trial_idx, row in beh.iterrows():
        gno = int(row['grid_no'])
        if gno not in grid_to_config:
            continue
        if row['correct'] == 1:
            gridno_to_trials[gno].append(trial_idx)

    if not gridno_to_trials:
        print(f"s{sub_str}: no correct trials found, skipping.")
        continue

    
    import pdb; pdb.set_trace()
    # build one row per (neuron × grid_no run)
    for n_idx, nkey in enumerate(neuron_keys):
        roi = cell_labels[n_idx] if n_idx < len(cell_labels) else 'unknown'
        arr = norm[nkey].values   # (n_trials, 360) ndarray

        for gno, trial_indices in gridno_to_trials.items():
            cfg = grid_to_config[gno]
            mean_tc = arr[trial_indices, :].mean(axis=0)  # (360,)
            config_str = f"{cfg[0]}-{cfg[1]}-{cfg[2]}-{cfg[3]}"
            row_dict = {
                'session':      f"s{sub_str}",
                'neuron_label': nkey,
                'roi':          roi,
                'config':       config_str,
                'grid_no':      gno,
            }
            for b, v in enumerate(mean_tc):
                row_dict[BIN_COLS[b]] = v
            neuron_rows.append(row_dict)

    print(f"s{sub_str}: {len(neuron_keys)} neurons × {len(gridno_to_trials)} grid_no runs "
          f"→ {len(neuron_keys) * len(gridno_to_trials)} rows added")

df_neurons = pd.DataFrame(neuron_rows, columns=['session', 'neuron_label', 'roi', 'config', 'grid_no'] + BIN_COLS)
os.makedirs(os.path.dirname(NEURON_CSV), exist_ok=True)
df_neurons.to_csv(NEURON_CSV, index=False)
print(f"\nneuron.csv saved to: {NEURON_CSV}")
print(f"Shape: {df_neurons.shape}")
print(df_neurons[['session', 'neuron_label', 'roi', 'config']].head(10).to_string(index=False))



# -----------------------------------------------------------------------
# Part 1 – Heatmap: location frequency as A/B/C/D across top configs
# -----------------------------------------------------------------------
PIVOT_CSV   = "/Users/xpsy1114/Documents/projects/multiple_clocks/results/config_pivot_table.csv"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"

df_pivot = pd.read_csv(PIVOT_CSV)

# Keep configs with at least 21 total runs
top = df_pivot[df_pivot['total_runs'] >= 21].copy()
top[['A', 'B', 'C', 'D']] = top['sequence'].str.split('-', expand=True).astype(int)

# Build 9x4 frequency matrix (locations 1-9 × rewards A/B/C/D),
# weighted by total_runs so more-repeated configs count more.
freq = np.zeros((9, 4))
for _, row in top.iterrows():
    for j, rew in enumerate(['A', 'B', 'C', 'D']):
        loc_idx = row[rew] - 1          # location 1-9 → index 0-8
        freq[loc_idx, j] += row['total_runs']

fig, ax = plt.subplots(figsize=(4, 6))
im = ax.imshow(freq, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(4));  ax.set_xticklabels(['A', 'B', 'C', 'D'], fontsize=12)
ax.set_yticks(range(9));  ax.set_yticklabels(range(1, 10))
ax.set_xlabel('Reward position');  ax.set_ylabel('Location (1–9)')
ax.set_title(f'Location frequency per reward\n(top {len(top)} configs, ≥21 runs)')
plt.colorbar(im, ax=ax, label='Total runs (weighted)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'top_config_location_heatmap.png'), dpi=150)
plt.close()
print(f"Heatmap saved.  Top configs: {len(top)},  total runs covered: {top['total_runs'].sum()}")

# -----------------------------------------------------------------------
# Part 2 – Per-session reward and step tables
# -----------------------------------------------------------------------


def runs_for_row(row):
    arr = np.asarray(row)
    change = np.flatnonzero(arr[1:] != arr[:-1]) + 1

    starts = np.r_[0, change]
    ends   = np.r_[change - 1, len(arr) - 1]
    locs   = arr[starts]

    runs = pd.DataFrame({"start": starts, "end": ends, "loc": locs})

    # merge first and last run if they have the same value
    if len(runs) > 1 and runs.loc[0, "loc"] == runs.loc[len(runs) - 1, "loc"]:
        runs.loc[0, "start"] = runs.loc[len(runs) - 1, "start"]
        runs = runs.iloc[:-1].reset_index(drop=True)

    runs["wrap"] = runs["start"] > runs["end"]
    runs["step"] = np.arange(len(runs))
    return runs


def find_run(runs, idx):
    hit = runs[(runs.start <= idx) & (idx <= runs.end)]
    if len(hit):
        return hit.iloc[0]

    hit = runs[(runs.start > runs.end) & ((idx >= runs.start) | (idx <= runs.end))]
    if len(hit):
        return hit.iloc[0]

    raise ValueError(f"No run found for idx={idx}")


def reward_label_for_run(run, n_bins):
    targets = {
        "reward_A": 90,
        "reward_B": 180,
        "reward_C": 270,
        "reward_D": n_bins - 1,   # 359 for a 360-bin row
    }

    for label, t in targets.items():
        idx = t % n_bins
        if run.start <= run.end:
            if run.start <= idx <= run.end:
                return label
        else:
            if idx >= run.start or idx <= run.end:
                return label

    return np.nan


def detect_bins_matrix(df_locs, df_beh):
    needed = ["grid_no", "loc_A", "loc_B", "loc_C", "loc_D"]
    missing = [c for c in needed if c not in df_beh.columns]
    if missing:
        raise KeyError(f"Missing columns in df_beh: {missing}")

    if len(df_locs) != len(df_beh):
        raise ValueError("df_locs and df_beh must have the same number of rows")

    beh = df_beh.copy()
    beh["_grid_block"] = beh["grid_no"].ne(beh["grid_no"].shift()).cumsum()

    n_bins = df_locs.shape[1]
    targets = {"A": 90, "B": 180, "C": 270, "D": n_bins - 1}

    block_step_tables = []
    rew_rows = []
    block_modal_steps = {}

    # first pass: compute modal step count per block
    for block_id, beh_block in beh.groupby("_grid_block", sort=False):
        step_counts = [len(runs_for_row(df_locs.loc[trial_idx])) for trial_idx in beh_block.index]
        block_modal_steps[block_id] = int(pd.Series(step_counts).mode().iat[0])

    max_horizon = max(block_modal_steps.values()) - 1  # largest next+k needed anywhere

    # second pass: build tables
    for block_id, beh_block in beh.groupby("_grid_block", sort=False):
        block_rows = []

        # flatten this block into one step sequence, but do not cross block boundary
        for trial_in_block, trial_idx in enumerate(beh_block.index):
            row = df_locs.loc[trial_idx]
            if row.isna().all():
                print(f"  [flag] trial {trial_idx} has all-NaN locations (recording dropout), skipping")
                continue
            runs = runs_for_row(row).copy()

            runs["trial_index"] = trial_idx
            runs["trial_in_block"] = trial_in_block
            runs["grid_block"] = block_id
            runs["grid_no"] = beh.loc[trial_idx, "grid_no"]
            runs["trial_step_count"] = len(runs)
            runs["block_modal_step_count"] = block_modal_steps[block_id]

            # reward label for each row in the step table
            runs["reward"] = runs.apply(lambda r: reward_label_for_run(r, n_bins), axis=1)

            block_rows.append(runs)

        if not block_rows:
            continue   # entire block was NaN dropout trials
        block_df = pd.concat(block_rows, ignore_index=True)

        # add lookahead columns, only within this block
        for k in range(1, max_horizon + 1):
            block_df[f"next+{k}"] = np.nan

        for i in range(len(block_df)):
            for k in range(1, block_modal_steps[block_id]):
                if i + k < len(block_df):
                    block_df.loc[i, f"next+{k}"] = block_df.loc[i + k, "loc"]

        block_step_tables.append(block_df)

        # trial summary table
        for trial_in_block, trial_idx in enumerate(beh_block.index):
            row = df_locs.loc[trial_idx]
            if row.isna().all():
                continue   # already flagged in the step loop above
            runs = runs_for_row(row).copy()

            rec = {
                "trial_index": trial_idx,
                "trial_in_block": trial_in_block,
                "grid_block": block_id,
                "grid_no": beh.loc[trial_idx, "grid_no"],
            }

            is_correct = beh.loc[trial_idx, "correct"] == 1
            for name, t in targets.items():
                run = find_run(runs, t)
                
                loc = run['loc']
                expected = beh.loc[trial_idx, f"loc_{name}"]

                if pd.notna(expected) and int(expected) != loc:
                    if is_correct:
                        # Mismatch on a correct trial is a real data problem — raise
                        raise ValueError(
                            f"Mismatch on CORRECT trial {trial_idx} "
                            f"(grid_block={block_id}, grid_no={beh.loc[trial_idx, 'grid_no']}), "
                            f"{name}: expected {expected}, got {loc}"
                        )
                    else:
                        # On incorrect trials the subject didn't reach the reward,
                        # so the location at the target bin won't match — expected behaviour.
                        print(f"  [flag] incorrect trial {trial_idx}, {name}: "
                              f"locations bin says {loc}, beh expected {int(expected)}")
                        import pdb; pdb.set_trace()
                rec[f"{name}_start"] = int(run.start)
                rec[f"{name}_end"]   = int(run.end)
                rec[f"{name}_loc"]   = loc

            rec["D_wrap"] = rec["D_start"] > rec["D_end"]
            rew_rows.append(rec)

    all_steps_df = pd.concat(block_step_tables, ignore_index=True) if block_step_tables else pd.DataFrame()

    rew_df = pd.DataFrame(rew_rows)[[
        "trial_index", "trial_in_block", "grid_block", "grid_no",
        "A_start", "A_end", "A_loc",
        "B_start", "B_end", "B_loc",
        "C_start", "C_end", "C_loc",
        "D_start", "D_end", "D_loc", "D_wrap"
    ]]

    return all_steps_df, rew_df


skipped = []
for sub in range(1, 64):
    data_dict, source_path = mc.analyse.helpers_human_cells.get_data(sub)
    if f"sub-{sub:02}" not in data_dict:
        print(f"s{sub:02}: data not found, skipping.")
        skipped.append(sub)
        continue

    curr_sub_beh  = data_dict[f"sub-{sub:02}"]["beh"]
    curr_sub_locs = data_dict[f"sub-{sub:02}"]["locations"]

    try:
        all_steps_df, rew_df = detect_bins_matrix(curr_sub_locs, curr_sub_beh)
    except ValueError as e:
        print(f"s{sub:02}: ERROR (correct-trial mismatch) — {e}")
        skipped.append(sub)
        continue

    # Save both tables next to the existing cells_and_beh files
    out_dir = os.path.join(source_path, f"s{sub:02}", "cells_and_beh")
    
    # import pdb; pdb.set_trace()
    # careful! I want that len(data_dict[f"sub-{sub:02}"]['normalised_neurons][neuron]) = len(rew_df)
    # also, think about what I need for the RSA and create it in this script.
    
    # for these sessions: 
    # ['s27', 's28', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's40', 's43', 's44', 's45', 's46', 's49', 's50', 's51', 's53', 's55', 's56', 's57', 's58', 's59', 's60', 's61', 's62', 's63']
    # config_summary = pd.read_csv(CONFIG_CSV)
    # sessions_to_load = config_summary[config_summary['n_sessions'] == config_summary['n_sessions'].max()]['sessions'].apply(ast.literal_eval).iloc[0]
    # create whatever you need for the control model RDMs here, as well as for the decoding.
    # also create a function that determiens how to average neurons and map sessions together.
    
    import pdb; pdb.set_trace()
    all_steps_df.to_csv(os.path.join(out_dir, f"all_steps_sub{sub:02}.csv"),    index=False)
    rew_df.to_csv(       os.path.join(out_dir, f"reward_bins_sub{sub:02}.csv"), index=False)
    print(f"s{sub:02}: saved all_steps ({len(all_steps_df)} rows) and reward_bins ({len(rew_df)} rows)")

print(f"\nDone.  Flagged+skipped sessions: {skipped if skipped else 'none'}")