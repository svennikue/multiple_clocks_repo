#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:14:06 2025
plotting ripples

@author: xpsy1114
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from scipy import stats
import numpy as np
import math
import pingouin as pg



# loading existing datasets 
# (didnt check, copied from chat)
# Session list
session_list = list(range(60))
result_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"

# Check if on server
if not os.path.isdir(result_path):
    print("running on ceph")
    result_path = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"

# Data containers
power_dict = {}
ripple_events = {}
beh = {}
loaded_sessions = []

column_names = [
    'rep_correct', 't_A', 't_B', 't_C', 't_D',
    'loc_A', 'loc_B', 'loc_C', 'loc_D',
    'rep_overall', 'new_grid_onset', 'session_no', 'grid_no'
]

# Loop through sessions and only load those with all files present
for s_idx in session_list:
    sesh = f"s{s_idx:02}"

    power_path = f"{result_path}/{sesh}/LFP-ripples/ripple_power_dict_{sesh}"
    beh_path = f"{result_path}/{sesh}/cells_and_beh/all_trial_times_{s_idx:02}.csv"
    ripple_path = f"{result_path}/{sesh}/LFP-ripples/ripples_{sesh}.csv"

    if os.path.isfile(power_path) and os.path.isfile(beh_path) and os.path.isfile(ripple_path):
        # Load power
        with open(power_path, 'rb') as f:
            power_dict[sesh] = pickle.load(f)
        # Load behavior
        beh[sesh] = pd.read_csv(beh_path, header=None)
        beh[sesh].columns = column_names
        # Load ripple events
        ripple_events[sesh] = pd.read_csv(ripple_path)

        loaded_sessions.append(sesh)
    else:
        print(f"Missing files for {sesh}, skipping.")

# ✅ Summary
print(f"\nLoaded sessions ({len(loaded_sessions)}): {loaded_sessions}")

sesh = loaded_sessions[-1]

print(f"overview of ripple event table for sessions {sesh}")
print(len(ripple_events[sesh]))
ripple_events[sesh].sort_values('onset_in_secs').head()


print(f"overview of behavioural table for sessions {sesh}")
print(len(beh[sesh]))
beh[sesh].head()


print(f"overview of power dictionary structure {sesh}")
print()
def print_dict_structure(d, indent=0):
    prefix = "  " * indent
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/ (dict)")
                print_dict_structure(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key} - list of length {len(value)}")
                if value and isinstance(value[0], dict):
                    print(f"{prefix}  [0]/ (dict in list)")
                    print_dict_structure(value[0], indent + 2)
            else:
                print(f"{prefix}{key} - {type(value).__name__}")
    else:
        print(f"{prefix}- {type(d).__name__}: {d}")


# sort timings in 'plan', 'explore' and 'repeat' for each grid.
beh_phases = {}
for sesh in beh:
    phases = []

    for grid_id, group in beh[sesh].groupby("grid_no"):
        group = group.sort_values(by="rep_overall")  # Ensure order

        # Find 'explore' phase
        first_row = group[group['rep_correct'] == 0].iloc[0]
        explore_start = first_row['new_grid_onset']
        explore_end = first_row['t_D']

        # Find 'plan' phase
        last_plan_row = group[group['rep_correct'] == 0].iloc[-1]
        plan_start = explore_end
        plan_end = last_plan_row['t_D']

        # Find 'repeat' phase
        max_rep_row = group.loc[group['rep_overall'].idxmax()]
        repeat_start = plan_end
        repeat_end = max_rep_row['t_D']

        # Add to list
        phases.append({
            'grid_no': grid_id,
            'phase': 'explore',
            'start': explore_start,
            'end': explore_end
        })
        phases.append({
            'grid_no': grid_id,
            'phase': 'plan',
            'start': plan_start,
            'end': plan_end
        })
        phases.append({
            'grid_no': grid_id,
            'phase': 'repeat',
            'start': repeat_start,
            'end': repeat_end
        })
    
    # Create result DataFrame
    beh_phases[sesh] = pd.DataFrame(phases)
    beh_phases[sesh]['duration'] = beh_phases[sesh]['end'] - beh_phases[sesh]['start']

beh_phases[sesh].head()




#compute the ripple rate per grid. 
ripple_rates = {}
for sesh in beh: 
    # Ensure grid numbers are consistently typed
    beh_phases[sesh]['grid_no'] = beh_phases[sesh]['grid_no'].astype(str)
    ripple_events[sesh]['grid_no'] = ripple_events[sesh]['grid_no'].str.extract(r'(\d+)')  # Extract numeric part from 'grid1'
    ripple_events[sesh]['grid_no'] = ripple_events[sesh]['grid_no'].astype(str)

    # Prepare an empty list to collect results
    results = []

    # Loop through each phase
    for _, phase_row in beh_phases[sesh].iterrows():
        grid = phase_row['grid_no']
        phase = phase_row['phase']
        start = phase_row['start']
        end = phase_row['end']
        duration = phase_row['duration']

        # Get matching ripples
        ripples_in_phase = ripple_events[sesh][
            (ripple_events[sesh]['grid_no'] == grid) &
            (ripple_events[sesh]['onset_in_secs'] >= start) &
            (ripple_events[sesh]['onset_in_secs'] < end)
        ]

        # Calculate ripple rate
        ripple_count = len(ripples_in_phase)
        ripple_rate = ripple_count / duration if duration > 0 else 0

        results.append({
            'grid_no': grid,
            'phase': phase,
            'ripple_count': ripple_count,
            'duration': duration,
            'ripple_rate': ripple_rate
        })
    
    ripple_rates[sesh]= pd.DataFrame(results)
ripple_rates[sesh].head()




all_results = []  # Collect all per-session dataframes here

for sesh in beh:
    # Ensure grid numbers are consistently typed
    beh_phases[sesh]['grid_no'] = beh_phases[sesh]['grid_no'].astype(str)
    ripple_events[sesh]['grid_no'] = ripple_events[sesh]['grid_no'].str.extract(r'(\d+)')
    ripple_events[sesh]['grid_no'] = ripple_events[sesh]['grid_no'].astype(str)

    results = []

    for _, phase_row in beh_phases[sesh].iterrows():
        grid = phase_row['grid_no']
        phase = phase_row['phase']
        start = phase_row['start']
        end = phase_row['end']
        duration = phase_row['duration']

        ripples_in_phase = ripple_events[sesh][
            (ripple_events[sesh]['grid_no'] == grid) &
            (ripple_events[sesh]['onset_in_secs'] >= start) &
            (ripple_events[sesh]['onset_in_secs'] < end)
        ]

        ripple_count = len(ripples_in_phase)
        rate = ripple_count / duration if duration > 0 else 0

        results.append({
            'session': sesh,
            'grid_no': grid,
            'phase': phase,
            'ripple_count': ripple_count,
            'duration': duration,
            'ripple_rate': rate
        })

    all_results.append(pd.DataFrame(results))

# Combine all into a single DataFrame
ripple_rates_df = pd.concat(all_results, ignore_index=True)
ripple_rates_df.head()



# compute stats per session.
# Store session-level stats
session_stats = []

for sesh, group in ripple_rates_df.groupby('session'):
    summary = group.groupby('phase')['ripple_rate'].agg(['mean', 'sem']).reset_index()
    summary['session'] = sesh

    # Pairwise t-tests between phases
    def get_rates(phase): return group[group['phase'] == phase]['ripple_rate']
    try:
        t_explore_plan = stats.ttest_rel(get_rates('explore'), get_rates('plan'))
        t_plan_repeat = stats.ttest_rel(get_rates('plan'), get_rates('repeat'))
        t_explore_repeat = stats.ttest_rel(get_rates('explore'), get_rates('repeat'))
    except Exception as e:
        print(f"Error in t-tests for session {sesh}: {e}")
        t_explore_plan = t_plan_repeat = t_explore_repeat = [np.nan, np.nan]

    session_stats.append({
        'session': sesh,
        'means': summary.set_index('phase')['mean'].to_dict(),
        'sems': summary.set_index('phase')['sem'].to_dict(),
        't_explore_vs_plan_p': t_explore_plan.pvalue,
        't_plan_vs_repeat_p': t_plan_repeat.pvalue,
        't_explore_vs_repeat_p': t_explore_repeat.pvalue
    })
    
stats_df = pd.DataFrame(session_stats)
stats_df.head()


# Utility: turn p-value into stars
def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'
    
    
    


n_sessions = len(stats_df)
colors = ['lightblue','goldenrod', 'salmon']
plt.rcParams.update({'font.size': 20})

# dynamically compute rows and columns of subplots
n_rows, n_cols = math.ceil(n_sessions / math.ceil(n_sessions**0.5)), math.ceil(n_sessions**0.5)
# Create figure and axis
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True)
axes = axes.flatten()
    
for idx, row in stats_df.iterrows():
    ax = axes[idx]
    # Plot the bars
    ax.bar(row['means'].keys(), row['means'].values(), yerr=row['sems'].values(), capsize=5, color=colors, alpha=0.5)
    
    # Prepare plotting the scatter points for individual data points (i.e. ripple rates per task)
    data_curr_sesh = ripple_rates_df[ripple_rates_df['session']==row['session']]
    ripple_rate_curr_sesh_phase= []
    for idx_p, phase in enumerate(row['means'].keys()):
        ripple_rate_curr_sesh_phase.append(data_curr_sesh[data_curr_sesh['phase']==phase]['ripple_rate'].values)
        ax.scatter(np.ones(len(ripple_rate_curr_sesh_phase[idx_p]))*idx_p, ripple_rate_curr_sesh_phase[idx_p], color=colors[idx_p], alpha=0.6, zorder=3)
    
    # connect the ripple rate datapoint of each task across phases to demonstrate trajectory
    for i_task, ripple_rate_in_task in enumerate(ripple_rate_curr_sesh_phase[idx]):
        ax.plot([0,1], [ripple_rate_curr_sesh_phase[0][i_task], ripple_rate_curr_sesh_phase[1][i_task]], color='gray', alpha=0.5, zorder=2)  # Connecting lines
        ax.plot([1,2], [ripple_rate_curr_sesh_phase[1][i_task], ripple_rate_curr_sesh_phase[2][i_task]], color='gray', alpha=0.5, zorder=2)  # Connecting lines
        
    

    # Compute max y for star placement
    all_vals = np.concatenate(ripple_rate_curr_sesh_phase)
    y_max = all_vals.max()
    
    # Draw line and star between plan and first_correct solve (plan)
    ax.plot([0, 1], [y_max]*2, color='black')
    ax.text(0.5, y_max + 0.02, p_to_star(row['t_explore_vs_plan_p']), ha='center')

    # Draw line and star between first_correct_rate and all_reps_rate
    ax.plot([1, 2], [y_max + 0.1]*2, color='black')
    ax.text(1.5, y_max + 0.12, p_to_star(row['t_plan_vs_repeat_p']), ha='center')

    # Draw line and star between find_ABCD_rate and all_reps_rate
    ax.plot([0, 2], [y_max + 0.2]*2, color='black')
    ax.text(1.0, y_max + 0.22, p_to_star(row['t_explore_vs_repeat_p']), ha='center')

    
    # final plot adjustments
    # Adjust y-limits to accommodate lines and stars
    ax.set_ylim(0, y_max + 0.3)

    # Adding labels
    ax.set_ylabel('Ripple Rate', labelpad=20)
    ax.set_xticks(ticks=[0, 1, 2], rotation = 45)
    ax.set_title(f"Ripple rate per grid \n for {row['session']}", pad=20)
    #ax.tight_layout()
    
# Turn off unused axes
for j in range(idx+1, len(axes)):
    axes[j].axis('off')

# Show the plot
plt.show()

    
# Prepare data in long format
anova_df = ripple_rates_df.pivot_table(index='session', columns='phase', values='ripple_rate').reset_index()

# Run repeated-measures ANOVA
anova = pg.rm_anova(data=anova_df.melt(id_vars='session', var_name='phase', value_name='ripple_rate'),
                    dv='ripple_rate', within='phase', subject='session', detailed=True)
print(anova)

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

p1 = ttest_rel(anova_df['explore'], anova_df['plan']).pvalue
p2 = ttest_rel(anova_df['plan'], anova_df['repeat']).pvalue
p3 = ttest_rel(anova_df['explore'], anova_df['repeat']).pvalue

# Apply Bonferroni correction
corrected = multipletests([p1, p2, p3], method='bonferroni')
print(f"Corrected p-values: {corrected[1]}")





