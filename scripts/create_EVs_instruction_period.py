#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 2026

Creates EV files for FEAT, plus the .fsf file that feeds them in, for the
instruction phase. One GLM per instruction epoch, each holding 10 EVs per task
half (one per task) — same shape as the '01-TR{k}' per-TR instruction GLMs in
create_EVs_for_RDMs.py, but the epochs are defined by what is on screen rather
than by TR.

Which epochs are made is set in the config file, e.g. EV_config_instruction.json.
Output goes to  EVs_{version}-TR{k}_pt0{th}/  with EVs named
{task}_{direction}_instruction_onset, so that
mc.analyse.my_RSA.load_data_EVs_instr_TRwise and pair_correct_tasks in
scripts/fMRI_run_RSA_instruction.py read it unchanged: set
"regression_version" to the config's name and "TR" to the epoch index k.

--- where the timings come from ---
The instruction routine is "show_rewards" in
mc/latest_experiment/3x3_fMRI_part1.py. It is non-slip timed to exactly 12 s
and the coins are drawn on this schedule (t from routine onset, lines 697-712):

    0.0 - 1.5   A          6.0 - 7.0   A
    1.5 - 3.0   B          7.0 - 8.0   B
    3.0 - 4.5   C          8.0 - 9.0   C
    4.5 - 6.0   D          9.0 - 10.0  D
                          10.0 - 12.0  background only (all coins and the
                                       'backwards' warning stop at 10 s)

--- where the onset comes from ---
The routine ends immediately before the first repeat of the task, so the
instruction onset is (first 'start_ABCD_screen' of that task) - 12 s. Checked
against the on-flip timestamps ('sand_pirate.started') in the *_all.csv: the
interval is constant to within ~12 ms (one frame) across all 10 tasks of a
session.

USAGE
    python create_EVs_instruction_period.py <subj_no> [config.json]
    (no subj_no -> all subjects)

@author: Svenja Kuechenhoff
"""

import numpy as np
import os
import pandas as pd
import mc
import sys
import json
import shutil

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    data_dir_deriv = f"{source_dir}/data/derivatives"
    data_dir = f"{source_dir}/data/pilot"
    analysis_dir = f"{source_dir}/multiple_clocks_repo/mc/fmri_analysis"
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    data_dir_deriv = f"{source_dir}/data/derivatives"
    data_dir = f"{source_dir}/data/pilot"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    analysis_dir = f"{source_dir}/analysis"
    print(f"Running on Cluster, setting {source_dir} as data directory")


# --- Load configuration ---
config_file = sys.argv[2] if len(sys.argv) > 2 else "EV_config_instruction.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
version = config.get("name")
segments = config.get("segments")
regress_buttons = config.get("regress_buttons", True)

# the instruction routine lasts 12 s in the experiment script
instruction_duration = 12
# nominal run length, only used to warn about epochs that would fall outside the
# acquired data. An EV past the end of the run makes feat_model abort.
nominal_run_duration = config.get("nominal_run_duration", 1670 * 1.078)

# Subjects
if len(sys.argv) > 1 and sys.argv[1]:
    subjects = [f"sub-{sys.argv[1]}"]
else:
    subjects = [f'sub-{i:02}' for i in range(1, 36)]
    subjects.remove('sub-29')
    subjects.remove('sub-21')


def load_beh(sub, th, all_file=False):
    # a couple of subjects have typos in their filenames, so try the variants.
    tail = "_all.csv" if all_file else ".csv"
    options = [f"{sub}_fmri_pt{th}{tail}", f"{sub}_fmri-pt{th}{tail}",
               f"{sub.replace('-', '_')}_fmri_pt{th}{tail}"]
    for option in options:
        if os.path.exists(f"{data_dir}/{sub}/beh/{option}"):
            return pd.read_csv(f"{data_dir}/{sub}/beh/{option}")
    raise FileNotFoundError(f"no behavioural file for {sub} pt{th}, tried {options}")


print(f"\ncreating {len(segments)} GLMs per task half:")
for segment in segments:
    print(f"   {version}_{segment['name'].replace('_', '-'):<30} "
          f"{segment['onset']:>4.1f} - {segment['onset'] + segment['duration']:.1f} s")
print("\nglm_names for the FEAT runner:")
print("   " + " ".join(s['name'].replace('_', '-') for s in segments))

for sub in subjects:
    for th in [1, 2]:
        print(f"\nNow creating instruction EVs for fmri file {th} and {sub}")

        df = load_beh(sub, th)
        df_all = load_beh(sub, th, all_file=True)

        # everything is timed relative to the first TR of this task half.
        first_TR_at = df_all['TR_received_no0'].dropna().unique().tolist()[0]

        # column that allows to differentiate all 10 tasks of this half
        df['config_type'] = df['task_config'] + '_' + df['type']
        df['config_type'] = df['config_type'].fillna(method='ffill')

        # Button press onsets -> nuisance regressor covering the execution periods.
        # nav_key_task.rt is relative to the start of the repeat it belongs to,
        # so add the start_ABCD_screen of that same repeat.
        on_press = []
        if regress_buttons == True:
            new_task = df[(~df['start_ABCD_screen'].isna())].reset_index(drop=True)
            end_task = df[(~df['nav_key_task.rt'].isna())].reset_index(drop=True)
            # in case a repeat was started but not finished (scanner stopped)
            while len(new_task) > len(end_task):
                new_task = new_task.drop(new_task.index[-1]).reset_index(drop=True)
            for i, row in end_task.iterrows():
                presses_curr_task = row['nav_key_task.rt'].strip('[]').split(', ')
                presses_curr_task = [(float(time) + new_task.at[i, 'start_ABCD_screen'])
                                     for time in presses_curr_task]
                on_press = on_press + presses_curr_task

        # Instruction onset per task = first start_ABCD_screen of that task - 12 s.
        instruction_onsets = df[~df['start_ABCD_screen'].isna()].groupby(
            'config_type', sort=False)['start_ABCD_screen'].first() - instruction_duration
        if len(instruction_onsets) < 10:
            print(f"careful! only {len(instruction_onsets)} instead of 10 instruction periods for {sub} in task half {th}")
        if (instruction_onsets - first_TR_at).min() < 0:
            print(f"careful! an instruction period starts before the first TR for {sub} in task half {th}")
        # An EV past the end of the run makes feat_model abort with "No valid
        # [onset duration strength] triplets found" -- it builds no design at
        # all rather than fitting an empty regressor. The real run length is
        # only known where the data is, so warn against the nominal length.
        last_ev_ends = (instruction_onsets - first_TR_at).max() + instruction_duration
        if last_ev_ends > nominal_run_duration:
            late = instruction_onsets[(instruction_onsets - first_TR_at + instruction_duration) > nominal_run_duration]
            print(f"careful! {sub} pt{th}: instruction period(s) {', '.join(late.index)} end after the "
                  f"nominal {nominal_run_duration:.0f} s run (last ends {last_ev_ends:.0f} s). "
                  f"If the acquired run is that short, feat_model will ABORT on these GLMs.")

        if sub in ['sub-04', 'sub-06', 'sub-30', 'sub-31', 'sub-34']:
            template_name = 'new_fsf_file_84.fsf'
        elif sub in ['sub-05', 'sub-35'] and th == 1:
            template_name = 'new_fsf_file_84.fsf'
        else:
            template_name = 'new_fsf_file_pnm_84.fsf'

        # one GLM per instruction epoch
        for seg_i, segment in enumerate(segments):
            # named after what it measures, e.g. 'instr_see-A-first', so the
            # folder reads EVs_instr_see-A-first_pt01 rather than a bare index.
            glm_name = f"{version}_{segment['name'].replace('_', '-')}"
            EV_folder = f'{data_dir_deriv}/{sub}/func/EVs_{glm_name}_pt0{th}/'
            if os.path.exists(EV_folder):
                shutil.rmtree(EV_folder)
            os.makedirs(EV_folder)

            for task, instruction_onset in instruction_onsets.items():
                EV = mc.analyse.analyse_MRI_behav.create_EV(
                    [instruction_onset + segment['onset']], [segment['duration']],
                    np.ones(1), f"{task}_instruction_onset", EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cut EV {task} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{task}_instruction_onset" + '.txt', array, delimiter="    ", fmt='%f')

            if regress_buttons == True:
                dur_press = np.ones(len(on_press)) * 0.02
                mag_press = np.ones(len(on_press))
                button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(
                    on_press, dur_press, mag_press, 'press_EV', EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted EV button press file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + 'press_EV' + '.txt', array, delimiter="    ", fmt='%f')

            # collect all filepaths I just created.
            EV_paths = [os.path.join(EV_folder, EV) for EV in os.listdir(EV_folder)
                        if EV.startswith("ev_") and EV.endswith(".txt")]
            sorted_EVs = sorted(EV_paths)

            with open(f"{EV_folder}task-to-EV.txt", 'w') as file:
                for i, EV_path in enumerate(sorted_EVs):
                    EV_file_name = EV_path.split('/')[-1].replace('.txt', '')
                    file.write(f'{i} {EV_file_name}\n')

            # then, adjust the .fsf file I will use for the regression.
            text_to_write = []
            with open(f"{analysis_dir}/templates/{template_name}", "r") as fin:
                for line in fin:
                    for i, EV_path in enumerate(sorted_EVs):
                        # the count in the EV file starts from 1, not 0 -> so do +1
                        if line.startswith(f"set fmri(custom{i+1})"):
                            line = f'set fmri(custom{i+1}) "{EV_path}"\n'
                        if line.startswith(f"set fmri(evtitle{i+1})"):
                            EV_name = os.path.basename(EV_path).rsplit('.', 1)[0]
                            line = f'set fmri(evtitle{i+1}) "{EV_name}"\n'
                        if line.startswith("set fmri(evs_orig)"):
                            line = f"set fmri(evs_orig) {len(sorted_EVs)}\n"
                        if line.startswith("set fmri(evs_real)"):
                            line = f"set fmri(evs_real) {len(sorted_EVs)}\n"
                    text_to_write.append(line)

            n_EVs = len(sorted_EVs)

            # then, in the next round, delete all the EVs that I don't actually include.
            # first, do this for the orthogonalisation of the EVs + contrasts you want with the ones you don't.
            skip = 0
            text_to_write_half_cleaned = []
            for line in text_to_write:
                if skip > 0:
                    # if the counter is increased, skip next line and decrease counter
                    skip -= 1
                    continue
                if (line.startswith("# Orthogonalise EV") and int(line.split()[-1]) > n_EVs) or (line.startswith("# Real contrast_orig") and int(line.split()[-1]) > n_EVs) or (line.startswith("# Real contrast_real vector") and int(line.split()[-1]) > n_EVs):
                    skip = 2
                else:
                    text_to_write_half_cleaned.append(line)

            # then, delete all the configurations of the actual EVs I don't want.
            skip_until_marker = False
            marker_line = "# Contrast & F-tests mode"
            text_to_write_cleaned = []
            for line in text_to_write_half_cleaned:
                if skip_until_marker:
                    if line.strip() == marker_line:
                        # add marker line to text and stop skipping
                        text_to_write_cleaned.append(line)
                        skip_until_marker = False
                    continue
                if line.startswith("# EV ") and int(line.split()[2]) > n_EVs:
                    skip_until_marker = True
                else:
                    text_to_write_cleaned.append(line)

            fsf_path = f"{data_dir_deriv}/{sub}/func/{sub}_draft_GLM_0{th}_{glm_name}.fsf"
            with open(fsf_path, "w") as fout:
                for line in text_to_write_cleaned:
                    fout.write(line)
            print(f"  {glm_name} ({segment['name']}): {n_EVs} EVs -> {fsf_path}")

            # Save the settings alongside the EVs for provenance, so the mapping
            # from epoch index to what was on screen is never lost.
            with open(os.path.join(EV_folder, f"{sub}_th-{th}_settings_summary.json"), "w") as f:
                json.dump({
                    "subject": sub,
                    "task_half": th,
                    "name": version,
                    "config_file": config_file,
                    "glm_name": glm_name,
                    "epoch_index": seg_i,
                    "epoch": segment,
                    "instruction_duration": instruction_duration,
                    "instruction_onset_defined_as": "first start_ABCD_screen of the task minus 12 s",
                    "regress_buttons": regress_buttons,
                    "n_tasks": len(instruction_onsets),
                    "n_EVs": n_EVs,
                    "first_TR_at": first_TR_at,
                    "OG_template_used": f"{analysis_dir}/templates/{template_name}",
                    "fsf_stored_as": fsf_path,
                    "EVs_stored_in": EV_folder,
                }, f, indent=2)
