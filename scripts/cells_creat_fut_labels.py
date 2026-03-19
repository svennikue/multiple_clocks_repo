import mc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def runs_for_row(row):
    arr = np.asarray(row)
    change = np.flatnonzero(arr[1:] != arr[:-1]) + 1

    starts = np.r_[0, change]
    ends   = np.r_[change - 1, len(arr) - 1]
    vals   = arr[starts]

    runs = pd.DataFrame({"start": starts, "end": ends, "val": vals})

    # merge first and last run if they have the same value
    if len(runs) > 1 and runs.loc[0, "val"] == runs.loc[len(runs) - 1, "val"]:
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

        block_df = pd.concat(block_rows, ignore_index=True)

        # add lookahead columns, only within this block
        for k in range(1, max_horizon + 1):
            block_df[f"next+{k}"] = np.nan

        for i in range(len(block_df)):
            for k in range(1, block_modal_steps[block_id]):
                if i + k < len(block_df):
                    block_df.loc[i, f"next+{k}"] = block_df.loc[i + k, "val"]

        block_step_tables.append(block_df)

        # trial summary table
        for trial_in_block, trial_idx in enumerate(beh_block.index):
            row = df_locs.loc[trial_idx]
            runs = runs_for_row(row).copy()

            rec = {
                "trial_index": trial_idx,
                "trial_in_block": trial_in_block,
                "grid_block": block_id,
                "grid_no": beh.loc[trial_idx, "grid_no"],
            }

            for name, t in targets.items():
                run = find_run(runs, t)
                val = int(run.val)
                expected = beh.loc[trial_idx, f"loc_{name}"]

                if pd.notna(expected) and int(expected) != val:
                    raise ValueError(
                        f"Mismatch at trial {trial_idx} "
                        f"(grid_block={block_id}, grid_no={beh.loc[trial_idx, 'grid_no']}), "
                        f"{name}: expected {expected}, got {val}"
                    )

                rec[f"{name}_start"] = int(run.start)
                rec[f"{name}_end"] = int(run.end)
                rec[f"{name}_val"] = val

            rec["D_wrap"] = rec["D_start"] > rec["D_end"]
            rew_rows.append(rec)

    all_steps_df = pd.concat(block_step_tables, ignore_index=True) if block_step_tables else pd.DataFrame()

    rew_df = pd.DataFrame(rew_rows)[[
        "trial_index", "trial_in_block", "grid_block", "grid_no",
        "A_start", "A_end", "A_val",
        "B_start", "B_end", "B_val",
        "C_start", "C_end", "C_val",
        "D_start", "D_end", "D_val", "D_wrap"
    ]]

    return all_steps_df, rew_df


for sub in range(20, 22):
    data_dict, source_path = mc.analyse.helpers_human_cells.get_data(sub)
    curr_sub_beh = data_dict[f"sub-{sub:02}"]["beh"]
    curr_sub_locs = data_dict[f"sub-{sub:02}"]["locations"]

    plt.figure()
    plt.imshow(curr_sub_locs, aspect="auto")
    for i in [90, 180, 270]:
        plt.axvline(i, color="white")
    plt.title(f"locations for subject {sub}")

    all_steps_df, rew_df = detect_bins_matrix(curr_sub_locs, curr_sub_beh)

    print(rew_df.head())
    print(all_steps_df.head())