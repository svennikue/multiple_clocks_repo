#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:12:02 2024

Functions for Ripples.
Human LFPs.


@author: Svenja Kuchenhoff
"""

import pdb
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import neo
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.stats import norm
import math
import mc 


# import pdb; pdb.set_trace() 

def plot_ripples_per_channel(data_dict, channel_list, subject):
    # import pdb; pdb.set_trace()

    # Group channels by family based on prefixes
    wires = defaultdict(list)
    for channel in channel_list:
        prefix = channel[:6]  # First 6 characters
        wires[prefix].append(channel)
    
    # Prepare data for plotting
    ripple_freq_per_channel = defaultdict(list)
    
    for family, channels in wires.items():
        for channel in channels:
            total_sum = 0
            for task, ripple_count in data_dict.items():
                if channel in ripple_count:
                    total_sum += len(ripple_count[channel])
            ripple_freq_per_channel[family].append(total_sum)
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Number of Ripples per Microwire')
    
    # Flatten axs for easy indexing
    axs = axs.flatten()
    # import pdb; pdb.set_trace()
    # Plot each family in its own subplot
    for i, (wire, sums) in enumerate(ripple_freq_per_channel.items()):
        axs[i].bar(range(len(sums)), sums, align='center')
        axs[i].set_title(f'Microwire: {wire} for subject {subject}')
        axs[i].set_xlabel('Channel Index')
        axs[i].set_xticks(range(len(sums)))  # Set positions
        axs[i].set_xticklabels(wires[wire], rotation=45, ha='right')  # Set labels and rotate for readability
        axs[i].set_ylabel('Ripple Amount')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    

def plot_ripple_count_three_bars(ripple_info_dict_across_sesh_tasks, sub):
    # first step: go through dicitonary and sort ripples by section:
    # finding all 4 rewards, solving it correctyl once, all other solves
    # then divide by average duration per grid.
    # then plot across grids.
    ripples_per_section = {}
    durations = {}
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        ripples_per_section_task = {}
        durations_per_section_task = {}
        for task_i in task_numbers:
            ripples_per_section_task[task_i], durations_per_section_task[task_i] = mc.analyse.ripple_helpers.sort_in_three_sections(session_sub_dict, task_i)
        
        ripples_per_section[session] = ripples_per_section_task.copy()
        durations[session] = durations_per_section_task.copy()
    
    
    find_ABCD, first_correct, all_reps = [],[],[]
    find_ABCD_rate, first_correct_rate, all_reps_rate = [],[],[]
    for session in sorted(ripples_per_section.keys()):
        for task in sorted(ripples_per_section[session].keys()):
            find_ABCD_rate.append(len(ripples_per_section[session][task]['find_ABCD'])/durations[session][task][0])
            find_ABCD.append(len(ripples_per_section[session][task]['find_ABCD']))
            first_correct_rate.append(len(ripples_per_section[session][task]['first_solve_correctly'])/durations[session][task][1])
            first_correct.append(len(ripples_per_section[session][task]['first_solve_correctly']))
            all_reps_rate.append(len(ripples_per_section[session][task]['all_repeats'])/durations[session][task][2])
            all_reps.append(len(ripples_per_section[session][task]['all_repeats']))
            
    colors = ['lightblue','goldenrod', 'salmon']

    # Calculate mean and standard error for bars
    means = [np.nanmean(find_ABCD_rate), np.nanmean(first_correct_rate), np.nanmean(all_reps_rate)]
    errors = [[0,0,0],[np.nanstd(find_ABCD_rate), np.nanstd(first_correct_rate), np.nanmean(all_reps_rate)]]
    labels = ['find_ABCD_locs', 'until first correct solve', 'later repeats']
    
    # import pdb; pdb.set_trace() 
    # Create figure and axis
    plt.figure(figsize=(5, 6))
    # Plot the bars
    bars = plt.bar(labels, means, yerr=errors, capsize=5, color=colors, alpha=0.5)
    
    # Plot scatter points for individual data points
    #for i in range(len(group_one)):
    plt.scatter(np.zeros(len(find_ABCD_rate)), find_ABCD_rate, color=colors[0], alpha=0.6, zorder=3)
    plt.scatter(np.ones(len(first_correct_rate)), first_correct_rate, color=colors[1], alpha=0.6, zorder=3) 
    plt.scatter(np.ones(len(all_reps_rate))*2, all_reps_rate, color=colors[2], alpha=0.6, zorder=3) 
    
    for i, task in enumerate(first_correct_rate):
        plt.plot([0, 1], [find_ABCD_rate[i], first_correct_rate[i]], color='gray', alpha=0.5, zorder=2)  # Connecting lines
        plt.plot([1,2], [first_correct_rate[i], all_reps_rate[i]], color='gray', alpha=0.5, zorder=2)  # Connecting lines
        
    # Adding the significance indicator (***)
    # Perform a t-test between the two lists
    # import pdb; pdb.set_trace() 
    t_stat_one, p_value_one = stats.ttest_rel(find_ABCD_rate, first_correct_rate, nan_policy='omit')
    t_stat_two, p_value_two = stats.ttest_rel(first_correct_rate, all_reps_rate, nan_policy='omit')
    t_stat_three, p_value_three = stats.ttest_rel(find_ABCD_rate, all_reps_rate, nan_policy='omit')
    
    stars_one = mc.analyse.plotting_ripples.significance_stars(p_value_one)
    stars_two = mc.analyse.plotting_ripples.significance_stars(p_value_two)
    stars_three = mc.analyse.plotting_ripples.significance_stars(p_value_three)
    
    positions = [0, 1, 2]  # x-positions for bars
    
    # Draw lines and stars for significance between pairs of bars
    y_max = max(find_ABCD_rate) + 0.05  # Set a base height for the lines
    if max(first_correct_rate) > y_max:
        y_max = max(first_correct_rate)
    if max(all_reps_rate) > y_max:
        y_max = max(all_reps_rate)
    
    # Draw line and star between find_ABCD_rate and first_correct_rate
    plt.plot([0, 1], [y_max, y_max], color='black')
    plt.text(0.5, y_max + 0.02, stars_one, ha='center', fontsize=14)
    
    # Draw line and star between first_correct_rate and all_reps_rate
    plt.plot([1, 2], [y_max + 0.1, y_max + 0.1], color='black')
    plt.text(1.5, y_max + 0.12, stars_two, ha='center', fontsize=14)
    
    # Draw line and star between find_ABCD_rate and all_reps_rate
    plt.plot([0, 2], [y_max + 0.2, y_max + 0.2], color='black')
    plt.text(1, y_max + 0.22, stars_three, ha='center', fontsize=14)
    
    # Adjust y-limits to accommodate lines and stars
    plt.ylim(0, y_max + 0.3)


    # Adding labels
    plt.ylabel('ripple_rate per section')
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.title(f"Ripple count/duration per grid for {sub}", fontsize=14)
    
    # Remove top and right spines
    # sns.despine()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # no_session = len(normalised_ripples)
        
        
    
# Determine number of stars based on p-value
def significance_stars(p_value):
    if p_value > 0.05:
        stars = 'n.s.'  # No significance
    elif p_value > 0.01:
        stars = '*'
    elif p_value > 0.005:
        stars = '**'
    else:
        stars = '***'
    return stars



def events_by_ripples(ripple_info_dict_across_sesh_tasks, feedback_across_sessions_tasks, sub):
    # first, put a gaussian normal distribution over every ripple
    # then add up these gaussians on the normalised scale
    # this makes it clear when I have a lot of ripples
    
    sigma = 0.01  # Standard deviation for the Gaussian kernel
    
    normalised_ripples, durations = {}, {}
    normalised_fb = {}
    
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        feedback_session_sub_dict = feedback_across_sessions_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        normalised_ripples_task = {}
        durations_per_section_task = {}
        norm_feedback_per_section_task = {}
        for task_i in task_numbers:
            normalised_ripples_task[task_i], durations_per_section_task[task_i] = mc.analyse.ripple_helpers.normalise_task(session_sub_dict, task_i)
            norm_feedback_per_section_task[f"{task_i}_pos"], norm_feedback_per_section_task[f"{task_i}_neg"] = mc.analyse.ripple_helpers.normalise_feedback(session_sub_dict, feedback_session_sub_dict, task_i)
    
        normalised_ripples[session] = normalised_ripples_task.copy()
        durations[session] = durations_per_section_task.copy()
        normalised_fb[session] = norm_feedback_per_section_task.copy()
        
    # import pdb; pdb.set_trace()
    # then, after I normalised this, collapse across sessions and tasks and do the    
    time_range = np.linspace(0,1,1000)
    
    
    for sesh_i, session in enumerate(sorted(normalised_ripples.keys())):
        all_ripples_across_tasks, all_fb_pos_across_tasks, all_fb_neg_across_tasks = [], [], []
        # import pdb; pdb.set_trace() 
        for task in normalised_ripples[session]:
            all_ripples_across_tasks.extend(normalised_ripples[session][task])
            all_fb_pos_across_tasks.extend(normalised_fb[session][f"{task}_pos"])
            all_fb_neg_across_tasks.extend(normalised_fb[session][f"{task}_neg"])
        

        # Plot the ripple likelihood curve
        plt.figure()
        ripples = sorted(all_ripples_across_tasks)
        for section in [0,1,2]:
            time_range_section = np.linspace(section*(1/3),(section+1)*(1/3),10000)
            ripple_likelihood = np.zeros_like(time_range_section)
            pos_likelihood = np.zeros_like(time_range_section)
            neg_likelihood = np.zeros_like(time_range_section)
            
            # find where ripples of first section are done (1/3)
            if section == 2:
                index_end_section = len(ripples)
            else:
                index_end_section = np.where(np.array(ripples) >= 1/3*(section+1))[0][0]
            if section == 0:
                index_start_section = 0
            else:
                index_start_section  = np.where(np.array(ripples) <= 1/3*section)[0][-1]
            
            for ripple_time in ripples[index_start_section:index_end_section]:
                ripple_likelihood += norm.pdf(time_range_section, loc=ripple_time, scale=sigma)
            
            
            # import pdb; pdb.set_trace() 
            for p_feedback_time in all_fb_pos_across_tasks:
                pos_likelihood += norm.pdf(time_range_section, loc=p_feedback_time, scale=sigma)
            for n_feedback_time in all_fb_neg_across_tasks:
                neg_likelihood += norm.pdf(time_range_section, loc=n_feedback_time, scale=sigma)
            
            # Get duration of the section for each task
            total_section_duration = 0
            for task in normalised_ripples[session]:
                total_section_duration += durations[session][task][section]  # Get duration for the section of the task
            
            # Normalize ripple likelihood by total duration
            ripple_likelihood /= total_section_duration
            plt.plot(time_range_section, ripple_likelihood, color='teal' , linewidth=0.8)
            plt.fill_between(time_range_section, ripple_likelihood, color='teal', alpha=0.3)
            
            pos_likelihood /= (np.max(pos_likelihood)/2)
            plt.plot(time_range_section, pos_likelihood, color= 'darkgreen', linewidth = 0.8)
            
            neg_likelihood /= (np.max(neg_likelihood)/2)
            plt.plot(time_range_section, neg_likelihood, color= 'maroon', linewidth = 0.8)
        
        # # Plot each ripple as a Gaussian
        # for ripple_time in ripples:
        #     plt.plot(time_range, norm.pdf(time_range, loc=ripple_time, scale=sigma), linestyle='--', color='purple', alpha=0.1, linewidth=0.2)
        
        # # Plot feedback events as vertical lines
        # for p_feedback_time in all_fb_pos_across_tasks:
        #     plt.axvline(p_feedback_time, color='darkgreen', linestyle='-', alpha=0.4, linewidth=0.2)
        # for n_feedback_time in all_fb_neg_across_tasks:
        #     plt.axvline(n_feedback_time, color='maroon', linestyle='-', alpha=0.4, linewidth=0.2)
        
        
        plt.axvline(1/3, color='grey', linestyle='--', linewidth=1)
        plt.axvline(2/3, color='grey', linestyle='--', linewidth=1)

    
        # Add labels and legend
        plt.title(f"Gaussian Kernels on Ripple Times and Feedback Events session {sesh_i}, {sub}")
        plt.xlabel("Time (normalised)")
        plt.ylabel("Ripple Likelihood / Duration per section")
        plt.legend()
        plt.show()
        # import pdb; pdb.set_trace()


def gaussian_ripples_HFB_aligned(ripple_info_dict_across_sesh_tasks, HFB_info_dict_across_sesh_tasks, sub, ROI_dict):
    # align all ripples with High frequency broadband events, keep split in 3 sections.
    # this will be a figure with 3 sub-figures: 
    # first row is finding ABCD for first time; second is solving it correctly for first time, third all other repeats.
    # always plot the 3 seconds before and after.
    
    
    # SOMETHING IS OFF HERE-
    # many timings are super similar but also the plots across regions look
    # pretty much the same...
    # why???
    # although at the same time, the probability curves are actually not the same...
    # weird. ask tim about this!
    
    
    # import pdb; pdb.set_trace()
    
    sigma = 0.02  # Standard deviation for the Gaussian kernel
    time_window = 3

    HFB_per_task_section_all_sessions, ripple_count_per_task_section_all_sessions= {}, {}

       
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        if ROI_dict =='empty':
            region_list = ['mPFC']
        else:
            region_list = [key for key in ROI_dict[session].keys()]

        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        # import pdb; pdb.set_trace()
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        ripples_per_section_task, HFB_per_section_task= {}, {}
        for task_i in task_numbers:
            ripples_per_section_task[task_i], durations = mc.analyse.ripple_helpers.sort_in_three_sections(session_sub_dict, task_i)
            HFB_per_section_task[task_i] = mc.analyse.ripple_helpers.sort_HFB_event_in_three_sections(session_sub_dict, HFB_info_dict_across_sesh_tasks[session], task_i, region_list)

        HFB_per_task_section_all_sessions[session] = HFB_per_section_task.copy()
        ripple_count_per_task_section_all_sessions[session] = ripples_per_section_task.copy()

    # Initialize dictionaries to store aligned ripples for each section
    aligned_ripples = {
        'find_ABCD': [], 
        'first_solve_correctly': [], 
        'all_repeats': []
    }
    
    no_HFB_events = {
        'find_ABCD': [], 
        'first_solve_correctly': [], 
        'all_repeats': []
    }
    
    
    # Now, per region
    # Collapse across all sessions and tasks
    HFB_event_ripple_likelihood = {}
    for ROI in region_list:
        for session in sorted(ripple_count_per_task_section_all_sessions.keys()):
            for task in sorted(ripple_count_per_task_section_all_sessions[session].keys()):
                for condition in sorted(ripple_count_per_task_section_all_sessions[session][task].keys()):
                    ripple_times = ripple_count_per_task_section_all_sessions[session][task][condition]
                    HFB_event = HFB_per_task_section_all_sessions[session][task][ROI][condition]
                    
                    for event in HFB_event:
                        aligned_ripples[condition].extend([ripple - event for ripple in ripple_times if abs(ripple - event) <= time_window])
                    no_HFB_events[condition].append(len(HFB_event))
                
        # Define section labels and their corresponding keys
        sections = [
            ('Finding ABCD', 'find_ABCD'),
            ('First Solve', 'first_solve_correctly'),
            ('All Repeats', 'all_repeats')
        ]
        
        # Set up the figure with 6 subplots (3 rows × 2 columns)
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)
        time_range = np.linspace(-time_window, time_window, 1000)
        
        # Iterate over sections and plot the data
        for row_idx, (section_label, section_name) in enumerate(sections):
            # HFB event subplot
            HFB_event_ripple_likelihood[ROI] = np.zeros_like(time_range)
            for ripple_time in aligned_ripples[section_name]:
                HFB_event_ripple_likelihood[ROI] += norm.pdf(time_range, loc=ripple_time, scale=sigma)
        
            HFB_event_ripple_likelihood[ROI] /= np.sum(no_HFB_events[section_name])
            
            axs[row_idx].plot(time_range, HFB_event_ripple_likelihood[ROI], color='teal', linewidth=0.8)
            axs[row_idx].fill_between(time_range, HFB_event_ripple_likelihood[ROI], color='teal', alpha=0.3)
            axs[row_idx].axvline(0, color='black', linestyle='--', linewidth=0.8)
            axs[row_idx].set_title(f'HFB event: {section_label}')
            axs[row_idx].set_ylabel('Ripple Likelihood')

        # Add common x-axis label
        axs[-1].set_xlabel('Time (s)')
        
        plt.suptitle(f"Gaussian Kernels on Ripples aligned to HFB events in {ROI};\n" 
                     f"by section; across grids and sessions for {sub}", 
                     fontsize=14, y=0.97)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.show()
        

def plot_ripple_rate_gaussian_exploration_phase(ripple_info_dict_across_sesh_tasks, feedback_across_sessions_tasks, sub):
    sigma = 0.5
    # plotting ripple rate for the exploration period.
    # do it once with subplots per task, including the feedback plot.
    # and then collapsed across tasks:
        # align/normalise by correct feedback, and divide the gaussian by the time each section takes!

    # for the first plot, literally plot in real time.
    # just 'cut out' the exploration trials.
    explore_trials_fb_pos,  explore_trials_fb_neg = {}, {}
    explore_ripples = {}
    durations = {}
    duration_ABCD = {}
    normalised_ripples = {}
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        explore_ripples_curr_session, explore_trials_fb_curr_session, durations_curr_session = {}, {}, {}
        explore_ripples_normalised_curr_session, explore_fb_pos_normalised_curr_session, explore_fb_neg_normalised_curr_session, duration_ABCD_curr_session = {}, {}, {}, {}
        fig, axs = plt.subplots(math.ceil(len(task_numbers)/2), 2, figsize=(12, len(task_numbers) * 4), sharey = True)
        
        for task in task_numbers:
            if task > math.ceil(len(task_numbers)):
                continue
                # to make it easy, just skip the last task.
                
            start_explore = ripple_info_dict_across_sesh_tasks[session][f"{task}_start_task"]
            end_explore = ripple_info_dict_across_sesh_tasks[session][f"{task}_first_corr_solve"]
            find_all_rews = ripple_info_dict_across_sesh_tasks[session][f"{task}_found_all_locs"]
            explore_ripples_curr_session[task] = [ripple for ripple in ripple_info_dict_across_sesh_tasks[session][f"{task}_ripples_across_choi"] if start_explore < ripple < end_explore]
            explore_trials_fb_curr_session[f"{task}_pos"] = [pos_fb for pos_fb in feedback_across_sessions_tasks[session][f"{task}_correct"] if start_explore < pos_fb < end_explore]
            explore_trials_fb_curr_session[f"{task}_neg"] = [neg_fb for neg_fb in feedback_across_sessions_tasks[session][f"{task}_error"] if start_explore < neg_fb < end_explore]
            durations_curr_session[task] = end_explore - start_explore
            duration_ABCD_curr_session[task] = find_all_rews+3 - start_explore
            explore_ripples_normalised_curr_session[task], explore_fb_pos_normalised_curr_session[task], explore_fb_neg_normalised_curr_session[task] = mc.analyse.ripple_helpers.normalise_explore_by_pos_fb(start_explore, find_all_rews+3, explore_ripples_curr_session, explore_trials_fb_curr_session, task)
            
            #then set up the real-time figure with each grid seperately.
            if task <= math.ceil(len(task_numbers)/2):
                col_idx = 0
                row_idx = 0
            elif task > math.ceil(len(task_numbers)/2):
                col_idx = 1
                row_idx = math.ceil(len(task_numbers)/2) * -1
            time_range = np.linspace(start_explore, end_explore, 1000)
            ripple_likelihood = np.zeros_like(time_range)
            for ripple in explore_ripples_curr_session[task]:
                ripple_likelihood += norm.pdf(time_range, loc=ripple, scale = sigma)
            # normalise by duration to make it ripple rate
            ripple_likelihood /= durations_curr_session[task]
            axs[task-1 + row_idx, col_idx].plot(time_range, ripple_likelihood, color = 'teal', linewidth = 0.8)
            axs[task-1 + row_idx, col_idx].fill_between(time_range, ripple_likelihood, color = 'teal', alpha = 0.3)
            axs[task-1 + row_idx, col_idx].set_title(f"Ripple Rate for task {task} between {start_explore} and {end_explore}", fontsize = 8)
            axs[task-1 + row_idx, col_idx].set_ylabel("Ripple Rate")
            
            axs[task-1 + row_idx, col_idx].axvline(find_all_rews, color='darkgreen', linestyle='--', linewidth=1.5)

            # axs[task-1 + row_idx, col_idx].yticks()
            for fb in explore_trials_fb_curr_session:
                if fb in [f"{task}_pos"]:
                    for event in explore_trials_fb_curr_session[fb]:
                        axs[task-1 + row_idx, col_idx].axvline(event, color='darkgreen', linestyle='--', linewidth=0.8)
                if fb in [f"{task}_neg"]:
                    for event in explore_trials_fb_curr_session[fb]:
                        axs[task-1 + row_idx, col_idx].axvline(event, color='maroon', linestyle='--', linewidth=0.8, alpha = 0.8)
        
        # Add common x-axis label
        # for ax in axs[-1, :]:
        #     ax.set_xlabel('Time (s)')
        
        plt.suptitle(f"Gaussian Kernels on Ripples in explore period (until first correct solve));\n" 
                     f" ripple count normalised by duration of explore period for subject {sub} session {session}", 
                     y=0.97)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.show()
        
        explore_ripples[session] = explore_ripples_curr_session.copy()
        explore_trials_fb_neg[session] = explore_fb_neg_normalised_curr_session.copy()
        explore_trials_fb_pos[session] = explore_fb_pos_normalised_curr_session.copy()
        durations[session] = durations_curr_session.copy()
        duration_ABCD[session] = duration_ABCD_curr_session.copy()
        normalised_ripples[session] = explore_ripples_normalised_curr_session.copy()
    # import pdb; pdb.set_trace() 
    # next, plot the same but overimposed on each other, and aligned by the correct feedbacks!
    
    plt.figure();
    
    sigma_norm = 0.01
    time_range = np.linspace(0,1, 1000)
    ripple_likelihood = np.zeros_like(time_range)
    counter = 0
    for task in explore_trials_fb_neg[session]:
        for fb_neg in explore_trials_fb_neg[session][task]:
            plt.axvline(fb_neg, color='maroon', linestyle='-', linewidth=0.3, alpha = 0.5)
            
    for session in normalised_ripples:
        for task in normalised_ripples[session]:
            ripple_likelihood_curr_task = np.zeros_like(time_range)
            for ripple in normalised_ripples[session][task]:
                ripple_likelihood_curr_task += norm.pdf(time_range, loc=ripple, scale = sigma_norm)
                counter += 1
            ripple_likelihood_curr_task /= duration_ABCD[session][task]
            ripple_likelihood += ripple_likelihood_curr_task
            # # delete after
            # plt.figure()
            # plt.plot(time_range, ripple_likelihood, color = 'pink', linewidth = 0.8)
    plt.plot(time_range, ripple_likelihood, color = 'teal', linewidth = 0.8)
    plt.fill_between(time_range, ripple_likelihood, color = 'teal', alpha = 0.3)
    plt.ylabel("Ripple Rate")
    for fb in explore_trials_fb_pos[session][1]:
        plt.axvline(fb, color='darkgreen', linestyle='--', linewidth=1.5)

            
    plt.title(f"Ripples in explore phase aligned by finding rewards for the first time: A-B-C-D. \n Ripplerate across grids and sessions for subject {sub}")
    plt.show()
            
       


   
def gaussian_spikes_around_ripple(ripple_info_dict_across_sesh_tasks, sub):
    # first prepare the spike data dictionary. 
    sigma = 0.005  # Standard deviation for the Gaussian kernel
    time_window = 0.8
    
    spiking_dict = mc.analyse.ripple_helpers.load_spiking_data(sub)
    
    # then do the 'reverse' of the gaussian ripple-to-feedback plotting:
        # for each ripple, check for spiking events, and plot as a gaussian curve.
        # then plot: one subplot per channel of this subject.
    # import pdb; pdb.set_trace()
    ripple_count_per_task_all_sessions= {}
    
    
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        ripples_per_task = {}
        for task_i in task_numbers:
            ripples_per_task[task_i] = ripple_info_dict_across_sesh_tasks[session][f"{task_i}_ripples_across_choi"]
        ripple_count_per_task_all_sessions[session] = ripples_per_task.copy()
        

    # Collapse across all sessions and tasks and align spikes to ripples.
     # Now, per region
     # Collapse across all sessions and tasks
    aligned_spikes = {}
    for cell in spiking_dict:
        aligned_spikes[cell] = []
    
    print(f"plotting {len(aligned_spikes)} for subject {sub}")                           
    no_ripples = []
    ripple_spiking_likelihood = {}
    for session in sorted(ripple_count_per_task_all_sessions.keys()):
       for task in sorted(ripple_count_per_task_all_sessions[session].keys()):
           ripple_times = ripple_count_per_task_all_sessions[session][task]
           for cell in spiking_dict:
               for ripple in ripple_times:
                   aligned_spikes[cell].extend([spike - ripple for spike in spiking_dict[cell] if abs(spike - ripple) <= time_window])
               no_ripples.append(len(ripple_times))
           
    num_cells = len(aligned_spikes)
    if num_cells < 8:
        fig, axs = plt.subplots(num_cells, 1, figsize=(12, num_cells * 4), sharex=True, sharey=True)  # Adjust height dynamically
        time_range = np.linspace(-time_window, time_window, 1000)
        
        # Ensure axs is iterable even if there's only one subplot
        if num_cells == 1:
            axs = [axs]
        
        # Iterate over cells and plot the data
        for idx, (cell, spike_times) in enumerate(aligned_spikes.items()):
            # Compute the Gaussian kernel for aligned spikes
            spike_likelihood = np.zeros_like(time_range)
            for spike_time in spike_times:
                spike_likelihood += norm.pdf(time_range, loc=spike_time, scale=sigma)
        
            # Normalize by the number of spikes (optional)
            spike_likelihood /= len(spike_times)
        
            # Plot the data
            axs[idx].plot(time_range, spike_likelihood, color='teal', linewidth=0.8)
            axs[idx].fill_between(time_range, spike_likelihood, color='teal', alpha=0.3)
            axs[idx].axvline(0, color='black', linestyle='--', linewidth=0.8)
            axs[idx].set_title(f'Spike Likelihood for {cell}', fontsize=12)
            axs[idx].set_ylabel('Spike Likelihood')
        
        # Add a common x-axis label
        axs[-1].set_xlabel('Time (s)', fontsize=12)
    
    elif num_cells > 8:
        fig, axs = plt.subplots(math.ceil(num_cells / 2), 2, figsize=(12, num_cells * 4), sharex=True, sharey=True)  # Adjust height dynamically
        time_range = np.linspace(-time_window, time_window, 1000)
        
        # Ensure axs is iterable even if there's only one subplot
        if num_cells == 1:
            axs = [axs]
        
        # Iterate over cells and plot the data
        for idx, (cell, spike_times) in enumerate(aligned_spikes.items()):
            if idx < math.ceil(num_cells / 2):
                # Compute the Gaussian kernel for aligned spikes
                spike_likelihood = np.zeros_like(time_range)
                for spike_time in spike_times:
                    spike_likelihood += norm.pdf(time_range, loc=spike_time, scale=sigma)
            
                # Normalize by the number of spikes (optional)
                spike_likelihood /= len(spike_times)
            
                # Plot the data
                axs[idx,0].plot(time_range, spike_likelihood, color='teal', linewidth=0.8)
                axs[idx,0].fill_between(time_range, spike_likelihood, color='teal', alpha=0.3)
                axs[idx,0].axvline(0, color='black', linestyle='--', linewidth=0.8)
                axs[idx,0].set_title(f'Spike Likelihood for {cell}', fontsize=12)
                axs[idx, 0].set_ylabel('Spike Likelihood')
            if idx > math.ceil(num_cells / 2):
                # Compute the Gaussian kernel for aligned spikes
                spike_likelihood = np.zeros_like(time_range)
                for spike_time in spike_times:
                    spike_likelihood += norm.pdf(time_range, loc=spike_time, scale=sigma)
            
                # Normalize by the number of spikes (optional)
                spike_likelihood /= len(spike_times)
            
                # Plot the data
                axs[idx-math.ceil(num_cells / 2),1].plot(time_range, spike_likelihood, color='teal', linewidth=0.8)
                axs[idx-math.ceil(num_cells / 2),1].fill_between(time_range, spike_likelihood, color='teal', alpha=0.3)
                axs[idx-math.ceil(num_cells / 2),1].axvline(0, color='black', linestyle='--', linewidth=0.8)
                axs[idx-math.ceil(num_cells / 2),1].set_title(f'Spike Likelihood for {cell}', fontsize=12)
                axs[idx-math.ceil(num_cells / 2),1].set_ylabel('Spike Likelihood')
     
    
    # Add a global title
    plt.suptitle(f"Gaussian Kernels on Spikes aligned to hippocampal ripples;\n"
                 f"by Cell; across Grids and Sessions for {sub}",
                 fontsize=14, y=0.97)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()        
         

def gaussian_ripples_feedback_neg_pos_aligned(ripple_info_dict_across_sesh_tasks, feedback_across_sessions_tasks, sub):
    # first detect those feedback combinations where a positive feedback follows a negative one.
    # then align ripples with the positive-negative feedback combination, and normalise; while keep split in 3 sections.
    # this will be a figure with 3 sub-figures: 
    # first row is finding ABCD for first time; second is solving it correctly for first time, third all other repeats.
    # plot in a normalised fashion, but always consider 3 seconds before negative and 3 seconds after the positive feedback.
    
    sigma = 0.05  # Standard deviation for the Gaussian kernel
    time_before_after = 3

    neg_pos_fb_combos_per_section, neg_pos_ripples_per_section = {}, {}
        
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        feedback_session_sub_dict = feedback_across_sessions_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        neg_pos_ripples_per_section_task, neg_pos_fb_combos_per_section_task= {}, {}
        for task_i in task_numbers:
            neg_pos_ripples_per_section_task[task_i], neg_pos_fb_combos_per_section_task[task_i] = mc.analyse.ripple_helpers.neg_pos_fb_combo_in_three_sections_normalised(session_sub_dict, feedback_session_sub_dict, task_i)

        neg_pos_fb_combos_per_section[session] = neg_pos_fb_combos_per_section_task.copy()
        neg_pos_ripples_per_section[session] = neg_pos_ripples_per_section_task.copy()



    # import pdb; pdb.set_trace()
    
    section_labels = ['find_ABCD', 'first_solve_correctly', 'all_repeats']
    # Initialize dictionaries to store aligned ripples for each section
    
    # Set up the figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)
    time_range = np.linspace(0, 3, 1000)

    for row_idx, section in enumerate(section_labels): 
        ripple_rate_likelihood = np.zeros_like(time_range)
        for session in sorted(neg_pos_ripples_per_section.keys()):
            for task in sorted(neg_pos_ripples_per_section[session].keys()):
                for fb_pair in neg_pos_ripples_per_section[session][task][section]:
                    gaussian_ripples = np.zeros_like(time_range)
                    for ripple in fb_pair['normalized_ripples']:
                        gaussian_ripples += norm.pdf(time_range, loc = ripple, scale = sigma)
                    ripple_rate_likelihood += (gaussian_ripples / fb_pair['duration'])
                    
                    
        axs[row_idx].plot(time_range, ripple_rate_likelihood, color='salmon', linewidth=0.8)
        axs[row_idx].fill_between(time_range, ripple_rate_likelihood, color='salmon', alpha=0.3)
        # neg_fb = neg_pos_fb_combos_per_section[session][task][section][0]['normalized_times'][1]
        axs[row_idx].axvline(1, color='maroon', linestyle='--', linewidth=1.5)
        # pos_fb = neg_pos_fb_combos_per_section[session][task][section][0]['normalized_times'][2]
        axs[row_idx].axvline(2, color='darkgreen', linestyle='--', linewidth=1.5)
   
        axs[row_idx].set_title(f'Positive-Neg Feedback pair: {section}')
        axs[row_idx].set_ylabel('Ripple Rate')
    
    
    # for session in sorted(neg_pos_ripples_per_section.keys()):
    #     for task in sorted(neg_pos_ripples_per_section[session].keys()):
    #         for row_idx, section in enumerate(section_labels):
    #             ripple_rate_likelihood = np.zeros_like(time_range)
    #             for fb_pair in neg_pos_ripples_per_section[session][task][section]:
    #                 gaussian_ripples = np.zeros_like(time_range)
    #                 for ripple in fb_pair['normalized_ripples']:
    #                     gaussian_ripples += norm.pdf(time_range, loc = ripple, scale = sigma)
    #                 ripple_rate_likelihood += (gaussian_ripples / fb_pair['duration'])
                    
    #     axs[row_idx, 0].plot(time_range, ripple_rate_likelihood, color='salmon', linewidth=0.8)
    #     axs[row_idx, 0].fill_between(time_range, ripple_rate_likelihood, color='salmon', alpha=0.3)
    #     neg_fb = neg_pos_fb_combos_per_section[session][task][section]['normalized_times'][1]
    #     axs[row_idx, 0].axvline(neg_fb, color='maroon', linestyle='--', linewidth=0.8)
    #     pos_fb = neg_pos_fb_combos_per_section[session][task][section]['normalized_times'][2]
    #     axs[row_idx, 0].axvline(pos_fb, color='darkgreen', linestyle='--', linewidth=0.8)
   
    #     axs[row_idx, 0].set_title(f'Positive-Neg Feedback pair: {section}')
    #     axs[row_idx, 0].set_ylabel('Ripple Rate')

    
    # Add common x-axis label

    plt.xlabel('Time (6 secs)', fontsize = 10)
    
    plt.suptitle(f"Gaussian Kernels on Ripples aligned to neg-pos feedback pairs;\n" 
                 f"3secs before and 3 secs after; normalised by ripple count, plotted by section; across grids and sessions for {sub}", 
                 fontsize=10, y=0.98, wrap =True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    
    
    # import pdb; pdb.set_trace()
    

    # # Collapse across all sessions and tasks
    # for session in sorted(ripple_count_per_task_section_all_sessions.keys()):
    #     for task in sorted(ripple_count_per_task_section_all_sessions[session].keys()):
    #         for condition in sorted(ripple_count_per_task_section_all_sessions[session][task].keys()):
    #             ripple_times = ripple_count_per_task_section_all_sessions[session][task][condition]
    #             feedback_pos = feedback_across_sessions[session][task][f"pos_{condition}"]
    #             feedback_neg = feedback_across_sessions[session][task][f"neg_{condition}"]
                
    #             for pos_event in feedback_pos:
    #                 aligned_ripples[f"pos_{condition}"].extend([ripple - pos_event for ripple in ripple_times if abs(ripple - pos_event) <= time_window])
            
    #             # Align ripples to negative feedback
    #             for neg_event in feedback_neg:
    #                 aligned_ripples[f"neg_{condition}"].extend([ripple - neg_event for ripple in ripple_times if abs(ripple - neg_event) <= time_window])
                
    #             no_fb[f"pos_{condition}"].append(len(feedback_across_sessions[session][task][f"pos_{condition}"]))
    #             no_fb[f"neg_{condition}"].append(len(feedback_across_sessions[session][task][f"neg_{condition}"]))
                
                
                
    # # Define section labels and their corresponding keys
    # sections = [
    #     ('Finding ABCD', 'pos_find_ABCD', 'neg_find_ABCD'),
    #     ('First Solve', 'pos_first_solve_correctly', 'neg_first_solve_correctly'),
    #     ('All Repeats', 'pos_all_repeats', 'neg_all_repeats')
    # ]
    
   
    
    # 

def gaussian_ripples_feedback_aligned(ripple_info_dict_across_sesh_tasks, feedback_across_sessions_tasks, sub):
    # align all ripples by positive or negative feedback, keep split in 3 sections.
    # this will be a figure with 6 sub-figures: positives at top, negative feedback at bottom
    # first row is finding ABCD for first time; second is solving it correctly for first time, third all other repeats.
    # always plot the 3 seconds before and after.
    
    sigma = 0.05  # Standard deviation for the Gaussian kernel
    time_window = 3

    feedback_across_sessions, ripple_count_per_task_section_all_sessions= {}, {}
        
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        feedback_session_sub_dict = feedback_across_sessions_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        ripples_per_section_task, feedback_per_section_task= {}, {}
        for task_i in task_numbers:
            ripples_per_section_task[task_i], durations = mc.analyse.ripple_helpers.sort_in_three_sections(session_sub_dict, task_i)
            feedback_per_section_task[task_i] = mc.analyse.ripple_helpers.sort_feedback_in_three_sections(session_sub_dict, feedback_session_sub_dict, task_i)

        feedback_across_sessions[session] = feedback_per_section_task.copy()
        ripple_count_per_task_section_all_sessions[session] = ripples_per_section_task.copy()

    # Initialize dictionaries to store aligned ripples for each section
    aligned_ripples = {
        'pos_find_ABCD': [], 'neg_find_ABCD': [],
        'pos_first_solve_correctly': [], 'neg_first_solve_correctly': [],
        'pos_all_repeats': [], 'neg_all_repeats': []}
    
    no_fb = {
        'pos_find_ABCD': [], 'neg_find_ABCD': [],
        'pos_first_solve_correctly': [], 'neg_first_solve_correctly': [],
        'pos_all_repeats': [], 'neg_all_repeats': []}
    
        
    # Collapse across all sessions and tasks
    for session in sorted(ripple_count_per_task_section_all_sessions.keys()):
        for task in sorted(ripple_count_per_task_section_all_sessions[session].keys()):
            for condition in sorted(ripple_count_per_task_section_all_sessions[session][task].keys()):
                ripple_times = ripple_count_per_task_section_all_sessions[session][task][condition]
                feedback_pos = feedback_across_sessions[session][task][f"pos_{condition}"]
                feedback_neg = feedback_across_sessions[session][task][f"neg_{condition}"]
                
                for pos_event in feedback_pos:
                    aligned_ripples[f"pos_{condition}"].extend([ripple - pos_event for ripple in ripple_times if abs(ripple - pos_event) <= time_window])
            
                # Align ripples to negative feedback
                for neg_event in feedback_neg:
                    aligned_ripples[f"neg_{condition}"].extend([ripple - neg_event for ripple in ripple_times if abs(ripple - neg_event) <= time_window])
                
                no_fb[f"pos_{condition}"].append(len(feedback_across_sessions[session][task][f"pos_{condition}"]))
                no_fb[f"neg_{condition}"].append(len(feedback_across_sessions[session][task][f"neg_{condition}"]))
                
                
                
    # Define section labels and their corresponding keys
    sections = [
        ('Finding ABCD', 'pos_find_ABCD', 'neg_find_ABCD'),
        ('First Solve', 'pos_first_solve_correctly', 'neg_first_solve_correctly'),
        ('All Repeats', 'pos_all_repeats', 'neg_all_repeats')
    ]
    
    # Set up the figure with 6 subplots (3 rows × 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
    time_range = np.linspace(-time_window, time_window, 1000)
    
    # Iterate over sections and plot the data
    for row_idx, (section_label, pos_key, neg_key) in enumerate(sections):
        # Positive feedback subplot
        pos_ripple_likelihood = np.zeros_like(time_range)
        for ripple_time in aligned_ripples[pos_key]:
            pos_ripple_likelihood += norm.pdf(time_range, loc=ripple_time, scale=sigma)
        
        pos_ripple_likelihood /= np.sum(no_fb[pos_key])
        
        axs[row_idx, 0].plot(time_range, pos_ripple_likelihood, color='darkgreen', linewidth=0.8)
        axs[row_idx, 0].fill_between(time_range, pos_ripple_likelihood, color='darkgreen', alpha=0.3)
        axs[row_idx, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)
        axs[row_idx, 0].set_title(f'Positive Feedback: {section_label}')
        axs[row_idx, 0].set_ylabel('Ripple Likelihood')
    
        # Negative feedback subplot
        neg_ripple_likelihood = np.zeros_like(time_range)
        for ripple_time in aligned_ripples[neg_key]:
            neg_ripple_likelihood += norm.pdf(time_range, loc=ripple_time, scale=sigma)
    
        neg_ripple_likelihood /= np.sum(no_fb[neg_key])
    
        axs[row_idx, 1].plot(time_range, neg_ripple_likelihood, color='maroon', linewidth=0.8)
        axs[row_idx, 1].fill_between(time_range, neg_ripple_likelihood, color='maroon', alpha=0.3)
        axs[row_idx, 1].axvline(0, color='black', linestyle='--', linewidth=0.8)
        axs[row_idx, 1].set_title(f'Negative Feedback: {section_label}')
        axs[row_idx, 1].set_ylabel('Ripple Likelihood')
    
    # Add common x-axis label
    for ax in axs[-1, :]:
        ax.set_xlabel('Time (s)')
    
    plt.suptitle(f"Gaussian Kernels on Ripples aligned to positive or negative feedback;\n" 
                 f"by section; nornalised for fb event count, across grids and sessions for {sub}", 
                 fontsize=14, y=0.97)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    
    # import pdb; pdb.set_trace()
   

        
        
        
    
    
    
   

def plot_ripples_before_vs_after_feedback(ripple_info_dict_across_sesh_tasks, feedback_across_sessions_tasks, sub, time_after_feedback):
    # first, define in which section ripples and feedback is.
    offset_feedback_sec = time_after_feedback
    ripples_by_feedback_section, ripple_count_per_task_section_all_sessions, section_duration_sessions = {}, {}, {}
        
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        feedback_session_sub_dict = feedback_across_sessions_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        ripples_per_section_task, feedback_per_section_task, duration_per_feedback_section = {}, {}, {}
        for task_i in task_numbers:
            ripples_per_section_task[task_i], durations = mc.analyse.ripple_helpers.sort_in_three_sections(session_sub_dict, task_i)
            feedback_per_section_task[task_i] = mc.analyse.ripple_helpers.sort_feedback_in_three_sections(session_sub_dict, feedback_session_sub_dict, task_i)
            #duration_per_feedback_section[task_i] = mc.analyse.ripple_helpers.extract_section_durations(session_sub_dict, feedback_per_section_task[task_i], task_i, time_after_feedback)
        
        ripple_count_per_task_section_all_sessions[session] = ripples_per_section_task.copy()
        ripples_by_feedback_section[session] = mc.analyse.ripple_helpers.sort_ripples_by_feedback_before_vs_after(feedback_per_section_task, ripples_per_section_task, offset_feedback_sec)
        #section_duration_sessions[session] = duration_per_feedback_section.copy()
        
    # then, check the RIPPLE CHANGE: ripple anount before vs after a feedback event.
    
    # plot as 6 bars: by pos/neg and 3 sections
    colors = ['darkgreen', 'maroon', 'darkgreen', 'maroon','darkgreen','maroon']

    # Calculate mean and standard error for bars
    pos_find_ABCD, pos_first_correct, pos_all_reps = [],[],[]
    neg_find_ABCD, neg_first_correct, neg_all_reps = [],[],[]
    
    pos_before_find_ABCD, pos_before_first_correct, pos_before_all_reps = [], [], []
    pos_after_find_ABCD, pos_after_first_correct, pos_after_all_reps = [], [], []
    neg_before_find_ABCD, neg_before_first_correct, neg_before_all_reps = [], [], []
    neg_after_find_ABCD, neg_after_first_correct, neg_after_all_reps = [], [], []
    
    
    #not_linked_to_fb_ABCD,not_linked_to_fb_first_solve, not_linked_to_fb_all_reps  = [], [], []
    for session in sorted(ripples_by_feedback_section.keys()):
        for task in sorted(ripples_by_feedback_section[session].keys()):
            # import pdb; pdb.set_trace()  
            if 'pos_find_ABCD_after_event' in ripples_by_feedback_section[session][task]:
                pos_find_ABCD.append(len(ripples_by_feedback_section[session][task]['pos_find_ABCD_after_event']) - 
                                     len(ripples_by_feedback_section[session][task]['pos_find_ABCD_before_event']))
                pos_before_find_ABCD.append(len(ripples_by_feedback_section[session][task]['pos_find_ABCD_before_event']))
                pos_after_find_ABCD.append(len(ripples_by_feedback_section[session][task]['pos_find_ABCD_after_event']))
                
            if 'neg_find_ABCD_after_event' in ripples_by_feedback_section[session][task]:
                neg_find_ABCD.append(len(ripples_by_feedback_section[session][task]['neg_find_ABCD_after_event']) -
                                     len(ripples_by_feedback_section[session][task]['neg_find_ABCD_before_event']))
                neg_before_find_ABCD.append(len(ripples_by_feedback_section[session][task]['neg_find_ABCD_before_event']))
                neg_after_find_ABCD.append(len(ripples_by_feedback_section[session][task]['neg_find_ABCD_after_event']))
                
            if 'pos_first_solve_correctly_after_event' in ripples_by_feedback_section[session][task]:
                pos_first_correct.append(len(ripples_by_feedback_section[session][task]['pos_first_solve_correctly_after_event']) -
                                         len(ripples_by_feedback_section[session][task]['pos_first_solve_correctly_before_event']))
                pos_before_first_correct.append(len(ripples_by_feedback_section[session][task]['pos_first_solve_correctly_before_event']))
                pos_after_first_correct.append(len(ripples_by_feedback_section[session][task]['pos_first_solve_correctly_after_event']))
                
            if 'neg_first_solve_correctly_after_event' in ripples_by_feedback_section[session][task]:
                neg_first_correct.append(len(ripples_by_feedback_section[session][task]['neg_first_solve_correctly_after_event']) -
                                         len(ripples_by_feedback_section[session][task]['neg_first_solve_correctly_before_event']))    
                neg_before_first_correct.append(len(ripples_by_feedback_section[session][task]['neg_first_solve_correctly_before_event']))
                neg_after_first_correct.append(len(ripples_by_feedback_section[session][task]['neg_first_solve_correctly_after_event']))
                
            if 'pos_all_repeats_after_event' in ripples_by_feedback_section[session][task]:
                pos_all_reps.append(len(ripples_by_feedback_section[session][task]['pos_all_repeats_after_event']) - 
                                    len(ripples_by_feedback_section[session][task]['pos_all_repeats_before_event']))
                pos_before_all_reps.append(len(ripples_by_feedback_section[session][task]['pos_all_repeats_before_event']))
                pos_after_all_reps.append(len(ripples_by_feedback_section[session][task]['pos_all_repeats_after_event']))
                
            if 'neg_all_repeats_after_event' in ripples_by_feedback_section[session][task]:
                neg_all_reps.append(len(ripples_by_feedback_section[session][task]['neg_all_repeats_after_event']) - 
                                    len(ripples_by_feedback_section[session][task]['neg_all_repeats_before_event']))
                neg_before_all_reps.append(len(ripples_by_feedback_section[session][task]['neg_all_repeats_before_event']))
                neg_after_all_reps.append(len(ripples_by_feedback_section[session][task]['neg_all_repeats_after_event']))
                
    
    means = [np.nanmean(pos_find_ABCD), np.nanmean(neg_find_ABCD),
             np.nanmean(pos_first_correct), np.nanmean(neg_first_correct), 
             np.nanmean(pos_all_reps), np.nanmean(neg_all_reps)]
    
    std = [[0,0,0,0,0,0], 
           [np.nanstd(pos_find_ABCD), np.nanstd(neg_find_ABCD), 
            np.nanstd(pos_first_correct), np.nanstd(neg_first_correct), 
            np.nanstd(pos_all_reps), np.nanstd(neg_all_reps)]]
    
    labels = ['while finding \n ABCD locs','while finding \n ABCD locs',  
              'until first \n correct solve','until first \n correct solve', 
              'in later \n repeats','in later \n repeats']
    
    colors_bef_aft = ['darkgreen', 'darkgreen', 'maroon', 'maroon', 'darkgreen', 'darkgreen', 'maroon', 'maroon', 'darkgreen', 'darkgreen', 'maroon', 'maroon']
    means_bef_aft = [np.nanmean(pos_before_find_ABCD), np.nanmean(pos_after_find_ABCD), np.nanmean(neg_before_find_ABCD), np.nanmean(neg_after_find_ABCD),
             np.nanmean(pos_before_first_correct), np.nanmean(pos_after_first_correct), np.nanmean(neg_before_first_correct), np.nanmean(neg_after_first_correct), 
             np.nanmean(pos_before_all_reps), np.nanmean(pos_after_all_reps), np.nanmean(neg_before_all_reps), np.nanmean(neg_after_all_reps)]
    
    std_bef_aft = [[0,0,0,0,0,0,0,0,0,0,0,0], 
           [np.nanstd(pos_before_find_ABCD), np.nanstd(pos_after_find_ABCD), np.nanstd(neg_before_find_ABCD), np.nanstd(neg_after_find_ABCD),
                    np.nanstd(pos_before_first_correct), np.nanstd(pos_after_first_correct), np.nanstd(neg_before_first_correct), np.nanstd(neg_after_first_correct), 
                    np.nanstd(pos_before_all_reps), np.nanstd(pos_after_all_reps), np.nanstd(neg_before_all_reps), np.nanstd(neg_after_all_reps)]]
           
    labels_bef_aft = ['while finding \n ABCD locs \n pos, before fb','while finding \n ABCD locs \n pos, after fb', 
                      'while finding \n ABCD locs \n neg, before fb',  'while finding \n ABCD locs \n neg, after fb',
                      'until first \n correct solve \n pos, before fb', 'until first \n correct solve \n pos, after fb',
                      'until first \n correct solve \n neg, before fb', 'until first \n correct solve \n neg, after fb',
                      'in later \n repeats \n pos, before fb', 'in later \n repeats \n pos, after fb', 
                      'in later \n repeats \n neg, before fb', 'in later \n repeats \n neg, after fb']

    
    # First create the direct before vs. after comparison.
    plt.figure(figsize=(5, 6))
    # Plot the bars
    bars = plt.bar(np.linspace(0,11,12), means_bef_aft, yerr=std_bef_aft, capsize=5, color=colors_bef_aft, alpha=0.5)
    # plt.xticks(rotation=45)
    plt.title(f"Ripple count {offset_feedback_sec} sec after vs. {offset_feedback_sec} sec before feedback, looking at feedback in this section {sub}", fontsize=14)
    plt.ylim(min(means_bef_aft), max(means_bef_aft))
    plt.xticks(ticks=np.linspace(0,11,12), labels = labels_bef_aft)
    
    # Plot scatter points for individual data points
    #for i in range(len(group_one)):
    all_means_bef_aft = [pos_before_find_ABCD, pos_after_find_ABCD, neg_before_find_ABCD, neg_after_find_ABCD,
                 pos_before_first_correct, pos_after_first_correct, neg_before_first_correct, neg_after_first_correct,
                 pos_before_all_reps, pos_after_all_reps, neg_before_all_reps, neg_after_all_reps]    
        
    for i, data in enumerate(all_means_bef_aft):
        y_jitter = np.random.uniform(-0.08, 0.08, len(data))
        plt.scatter(np.ones(len(data))*i+y_jitter, data, color=colors_bef_aft[i], alpha=0.6, zorder=3)
    
    # Connect dots within each trial group
    groups_bef_aft = [(0, 1), (2,3), (4,5), (6,7), (8,9), (10,11)]  # Indices of bars within each group
    for group in groups_bef_aft:
        for trial_idx in range(len(all_means_bef_aft[group[0]])):  # Loop through each trial
            x_coords = [group_idx for group_idx in group]  # Bar positions for this group
            y_coords = [all_means_bef_aft[group_idx][trial_idx] for group_idx in group]  # Y-values for this trial
            plt.plot(x_coords, y_coords, color='black', linestyle='-', alpha=0.5, linewidth=0.2, zorder=2)
    
    
    for section in range(0,6): 
        t_stat_one, p_value_one = stats.ttest_rel(all_means_bef_aft[section*2], all_means_bef_aft[section*2+1], nan_policy='omit')
        stars_one = mc.analyse.plotting_ripples.significance_stars(p_value_one)
        positions = [section*3]  # x-positions for bars
        
        # Draw lines and stars for significance between pairs of bars
        #y_max = 1
        # do 1 plus max std just in case
        y_max = np.max(means_bef_aft) + np.max(std_bef_aft)
            
        # Draw line and star between find_ABCD_rate and first_correct_rate
        plt.plot([section*2, section*2+1], [y_max, y_max], color='black')
        plt.text(section*2+0.5, y_max + 0.02, stars_one, ha='center', fontsize=14)
        # Adjust y-limits to accommodate lines and stars
        plt.ylim(0, y_max + 1)
        
        
    # import pdb; pdb.set_trace() 
    # Create figure and axis
    plt.figure(figsize=(5, 6))
    # Plot the bars
    bars = plt.bar(np.linspace(0,5,6), means, yerr=std, capsize=5, color=colors, alpha=0.5)
    # plt.xticks(rotation=45)
    plt.title(f"Difference in ripple count {offset_feedback_sec} sec after- {offset_feedback_sec} sec before feedback, looking at feedback in this section {sub}", fontsize=14)
    plt.ylim(min(means), max(means))
    plt.xticks(ticks=np.linspace(0,5,6), labels = labels)
    
    # Plot scatter points for individual data points
    #for i in range(len(group_one)):
    all_means = [pos_find_ABCD, neg_find_ABCD,
                 pos_first_correct, neg_first_correct, 
                 pos_all_reps, neg_all_reps]    
        
    for i, data in enumerate(all_means):
        y_jitter = np.random.uniform(-0.08, 0.08, len(data))
        plt.scatter(np.ones(len(data))*i+y_jitter, data, color=colors[i], alpha=0.6, zorder=3)
     
      
    # Adding stats
    for section in range(0,3): 
        # import pdb; pdb.set_trace() 
        t_stat_one, p_value_one = stats.ttest_rel(all_means[section*2], all_means[section*2+1], nan_policy='omit')
        stars_one = mc.analyse.plotting_ripples.significance_stars(p_value_one)
        positions = [section*3]  # x-positions for bars
        
        # Draw lines and stars for significance between pairs of bars
        #y_max = 1
        # do 1 plus max std just in case
        y_max = 1 + np.max(std)
        y_min = min(means) - max(std[1])
            
        # Draw line and star between find_ABCD_rate and first_correct_rate
        plt.plot([section*2, section*2+1], [y_max, y_max], color='black')
        plt.text(section*2+0.5, y_max + 0.02, stars_one, ha='center', fontsize=14)
        # Adjust y-limits to accommodate lines and stars
        plt.ylim(y_min, y_max + 0.3)


    # Show the plot
    # import pdb; pdb.set_trace() 
    plt.tight_layout()
    plt.show()
    # import pdb; pdb.set_trace() 
    
    
    
    
    

def plot_ripples_by_feedback(ripple_info_dict_across_sesh_tasks, feedback_across_sessions_tasks, sub, time_after_feedback):
    # first, define in which section ripples and feedback is.
    offset_feedback_sec = time_after_feedback
    ripples_by_feedback_section, ripple_count_per_task_section_all_sessions, section_duration_sessions = {}, {}, {}
        
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        feedback_session_sub_dict = feedback_across_sessions_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        ripples_per_section_task, feedback_per_section_task, duration_per_feedback_section = {}, {}, {}
        for task_i in task_numbers:
            ripples_per_section_task[task_i], durations = mc.analyse.ripple_helpers.sort_in_three_sections(session_sub_dict, task_i)
            feedback_per_section_task[task_i] = mc.analyse.ripple_helpers.sort_feedback_in_three_sections(session_sub_dict, feedback_session_sub_dict, task_i)
            # CONTINUE HERE!!
            # for some reason, for sub-2 s8 and task 10, I have ripples but no feedback. WHY?
            
            
            duration_per_feedback_section[task_i] = mc.analyse.ripple_helpers.extract_section_durations(session_sub_dict, feedback_per_section_task[task_i], task_i, time_after_feedback)
        
        ripple_count_per_task_section_all_sessions[session] = ripples_per_section_task.copy()
        ripples_by_feedback_section[session] = mc.analyse.ripple_helpers.sort_ripples_by_feedback(feedback_per_section_task, ripples_per_section_task, offset_feedback_sec)
        section_duration_sessions[session] = duration_per_feedback_section.copy()
        
    # then, check how many ripples per grid are within 1.5 after positiv/negative feedback.
    # plot as 6 bars: by pos/neg and 3 sections.
    colors = ['darkgreen', 'maroon', 'lightgrey', 'darkgreen', 'maroon', 'lightgrey', 'darkgreen','maroon', 'lightgrey']

    # Calculate mean and standard error for bars
    pos_find_ABCD, pos_first_correct, pos_all_reps = [],[],[]
    neg_find_ABCD, neg_first_correct, neg_all_reps = [],[],[]
    not_linked_to_fb_ABCD,not_linked_to_fb_first_solve, not_linked_to_fb_all_reps  = [], [], []
    for session in sorted(ripples_by_feedback_section.keys()):
        for task in sorted(ripples_by_feedback_section[session].keys()):
            
            # delete this bit
            ripple_count_find_ABCD = len(ripple_count_per_task_section_all_sessions[session][task]['find_ABCD'])
            ripple_count_first_solve = len(ripple_count_per_task_section_all_sessions[session][task]['first_solve_correctly'])
            ripple_count_all_reps = len(ripple_count_per_task_section_all_sessions[session][task]['all_repeats'])
            
            sum_of_ABCD, sum_of_correct_solve, sum_of_all_reps = [], [], []
            
            # divide ripple count by time spent in this section. 
            
            if section_duration_sessions[session][task]['pos_find_ABCD'] > 0:
                if 'pos_find_ABCD' in ripples_by_feedback_section[session][task]:
                    pos_find_ABCD.append(len(ripples_by_feedback_section[session][task]['pos_find_ABCD'])/section_duration_sessions[session][task]['pos_find_ABCD'])
            else:
                pos_find_ABCD.append(0)
            if section_duration_sessions[session][task]['neg_find_ABCD'] > 0:   
                if 'neg_find_ABCD' in ripples_by_feedback_section[session][task]:
                    neg_find_ABCD.append(len(ripples_by_feedback_section[session][task]['neg_find_ABCD'])/section_duration_sessions[session][task]['neg_find_ABCD'])
            else:
                neg_find_ABCD.append(0)
            if section_duration_sessions[session][task]['pos_first_solve_correctly'] > 0:
                if 'pos_first_solve_correctly' in ripples_by_feedback_section[session][task]:
                    pos_first_correct.append(len(ripples_by_feedback_section[session][task]['pos_first_solve_correctly'])/section_duration_sessions[session][task]['pos_first_solve_correctly'])
            else:
                pos_first_correct.append(0)
            if section_duration_sessions[session][task]['neg_first_solve_correctly'] > 0:    
                if 'neg_first_solve_correctly' in ripples_by_feedback_section[session][task]:
                    neg_first_correct.append(len(ripples_by_feedback_section[session][task]['neg_first_solve_correctly'])/section_duration_sessions[session][task]['neg_first_solve_correctly'])    
            else:
                neg_first_correct.append(0)
            if section_duration_sessions[session][task]['pos_all_repeats'] > 0:
                if 'pos_all_repeats' in ripples_by_feedback_section[session][task]:
                    pos_all_reps.append(len(ripples_by_feedback_section[session][task]['pos_all_repeats'])/section_duration_sessions[session][task]['pos_all_repeats'])
            else:
                pos_all_reps.append(0)
            if section_duration_sessions[session][task]['neg_all_repeats'] > 0:
                if 'neg_all_repeats' in ripples_by_feedback_section[session][task]:
                    neg_all_reps.append(len(ripples_by_feedback_section[session][task]['neg_all_repeats'])/section_duration_sessions[session][task]['neg_all_repeats'])
            else:
                neg_all_reps.append(0)
            
            # and compute how many ripples occured outside of feedback.
            for key, list in ripples_by_feedback_section[session][task].items():
                if key.endswith('find_ABCD'):
                    sum_of_ABCD.append(len(ripples_by_feedback_section[session][task][key]))
                if key.endswith('solve_correctly'):
                    sum_of_correct_solve.append(len(ripples_by_feedback_section[session][task][key])) 
                if key.endswith('all_repeats'):
                    sum_of_all_reps.append(len(ripples_by_feedback_section[session][task][key]))
                
            # divide ripple count by time spent in this section. 
            not_linked_to_fb_ABCD.append((ripple_count_find_ABCD-np.sum(sum_of_ABCD))/section_duration_sessions[session][task]['not_linked_to_fb_find_ABCD'])
            not_linked_to_fb_first_solve.append((ripple_count_first_solve-np.sum(sum_of_correct_solve))/section_duration_sessions[session][task]['not_linked_to_fb_first_solve_correctly'])
            not_linked_to_fb_all_reps.append((ripple_count_all_reps-np.sum(sum_of_all_reps))/section_duration_sessions[session][task]['not_linked_to_fb_all_repeats'])
                
    
    # import pdb; pdb.set_trace() 
    means = [np.nanmean(pos_find_ABCD), np.nanmean(neg_find_ABCD), np.nanmean(not_linked_to_fb_ABCD),
             np.nanmean(pos_first_correct), np.nanmean(neg_first_correct), np.nanmean(not_linked_to_fb_first_solve),
             np.nanmean(pos_all_reps), np.nanmean(neg_all_reps), np.nanmean(not_linked_to_fb_all_reps)]

    
    std = [[0,0,0,0,0,0,0,0,0], 
           [np.nanstd(pos_find_ABCD), np.nanstd(neg_find_ABCD), np.nanstd(not_linked_to_fb_ABCD),
            np.nanstd(pos_first_correct), np.nanstd(neg_first_correct), np.nanstd(not_linked_to_fb_first_solve),
            np.nanstd(pos_all_reps), np.nanstd(neg_all_reps), np.nanstd(not_linked_to_fb_all_reps)]]
    
    labels = ['while finding \n ABCD locs','while finding \n ABCD locs', 'not linked \n to feedback', 
              'until first \n correct solve','until first \n correct solve', 'not linked \n to feedback', 
              'in later \n repeats','in later \n repeats', 'not linked \n to feedback']
    
    # 
    # Create figure and axis
    plt.figure(figsize=(5, 6))
    # Plot the bars
    bars = plt.bar(np.linspace(0,8,9), means, yerr=std, capsize=5, color=colors, alpha=0.5)
    # plt.xticks(rotation=45)
    plt.title(f"Ripples {offset_feedback_sec} sec after feedback, divided by time spent (not) looking at feedback in this section {sub}", fontsize=14)
    plt.ylim(0,1)
    plt.xticks(ticks=np.linspace(0,8,9), labels = labels)
    
    
    # Plot scatter points for individual data points
    #for i in range(len(group_one)):
    all_means = [pos_find_ABCD, neg_find_ABCD, not_linked_to_fb_ABCD, 
                 pos_first_correct, neg_first_correct, not_linked_to_fb_first_solve,
                 pos_all_reps, neg_all_reps, not_linked_to_fb_all_reps]    
        
     
    for i, data in enumerate(all_means):
        plt.scatter(np.ones(len(data))*i, data, color=colors[i], alpha=0.6, zorder=3)
     
    # Connect dots within each trial group
    groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]  # Indices of bars within each group
    for group in groups:
        for trial_idx in range(len(all_means[group[0]])):  # Loop through each trial
            x_coords = [group_idx for group_idx in group]  # Bar positions for this group
            y_coords = [all_means[group_idx][trial_idx] for group_idx in group]  # Y-values for this trial
            plt.plot(x_coords, y_coords, color='black', linestyle='-', alpha=0.5, linewidth=0.2, zorder=2)
    
    # import pdb; pdb.set_trace() 
    
    # Adding stats
    for section in range(0,3): 
        # import pdb; pdb.set_trace() 
        t_stat_one, p_value_one = stats.ttest_rel(all_means[section*3], all_means[section*3+1], nan_policy='omit')
        t_stat_two, p_value_two = stats.ttest_rel(all_means[section*3+1], all_means[section*3+2], nan_policy='omit')
        t_stat_three, p_value_three = stats.ttest_rel(all_means[section*3], all_means[section*3+2], nan_policy='omit')
    
        stars_one = mc.analyse.plotting_ripples.significance_stars(p_value_one)
        stars_two = mc.analyse.plotting_ripples.significance_stars(p_value_two)
        stars_three = mc.analyse.plotting_ripples.significance_stars(p_value_three)
        
        positions = [section*3, section*3+1, section*3+2]  # x-positions for bars
        
        # Draw lines and stars for significance between pairs of bars
        #y_max = 1
        # do 1 plus max std just in case
        y_max = 1 + np.max(std)
            
        # Draw line and star between find_ABCD_rate and first_correct_rate
        plt.plot([section*3, section*3+1], [y_max, y_max], color='black')
        plt.text(section*3+0.5, y_max + 0.02, stars_one, ha='center', fontsize=14)
        
        # Draw line and star between first_correct_rate and all_reps_rate
        plt.plot([section*3+1, section*3+2], [y_max + 0.1, y_max + 0.1], color='black')
        plt.text(section*3+1.5, y_max + 0.12, stars_two, ha='center', fontsize=14)
        
        # Draw line and star between find_ABCD_rate and all_reps_rate
        plt.plot([section*3, section*3+2], [y_max + 0.2, y_max + 0.2], color='black')
        plt.text(section*3+1, y_max + 0.22, stars_three, ha='center', fontsize=14)
        
        # Adjust y-limits to accommodate lines and stars
        plt.ylim(0, y_max + 0.3)


    # # Adding labels
    # plt.ylabel('ripple_rate per section')
    # plt.xticks(ticks=[0, 1, 2], labels=labels)
    # plt.title(f"Ripple count/duration per grid for {sub}", fontsize=14)
    
    # Show the plot
    # import pdb; pdb.set_trace() 
    plt.tight_layout()
    plt.show()
    
    
    
    
    
 

def plot_ripple_distribution_normalised_across_tasks_and_sessions(ripple_info_dict_across_sesh_tasks, sub):
    # first step: loop through the dictionary and normalise the times.
    # one third will be start_task - found_all_locs, 
    # second third will be found_all_locs - first_corr_solve
    # third third will be first_corr_solve - end_task
    # import pdb; pdb.set_trace()
    normalised_ripples = {}
    durations = {}
    for session in sorted(ripple_info_dict_across_sesh_tasks.keys()):
        session_sub_dict = ripple_info_dict_across_sesh_tasks[session]
        # then count how many tasks: split the string after first element and take max
        # Extract the number before the underscore and convert to integers
        task_numbers = sorted([int(key.split('_')[0]) for key in ripple_info_dict_across_sesh_tasks[session].keys()])
        task_numbers = np.unique(task_numbers)
        normalised_ripples_task = {}
        durations_per_section_task = {}
        for task_i in task_numbers:
            normalised_ripples_task[task_i], durations_per_section_task[task_i] = mc.analyse.ripple_helpers.normalise_task(session_sub_dict, task_i)
        
        print(f"mean ripples in task {task_numbers[0]} in {session} is {np.mean(normalised_ripples_task[task_numbers[0]])}")
        
        normalised_ripples[session] = normalised_ripples_task.copy()
        durations[session] = durations_per_section_task.copy()
        
    # import pdb; pdb.set_trace()
    # Define normalized section ranges (e.g., each section is a third of the normalized scale)
    # colors = ['purple', 'lightblue', 'teal']
    # colors_kde = ['lavender','lightblue', 'teal']
    colors = ['maroon','teal', 'darkgreen']
    colors_kde = ['maroon','teal', 'darkgreen']
    plt.figure()
    
    no_session = len(normalised_ripples)
    
    for sesh_i, session in enumerate(sorted(normalised_ripples.keys())):
        all_ripples_across_tasks = []
        for task in normalised_ripples[session]:
            y_jitter = np.random.uniform(0,1, size=len(normalised_ripples[session][task]))
            plt.scatter(normalised_ripples[session][task], y_jitter, color=colors[sesh_i], marker='o', zorder=1, alpha = 0.5)
            all_ripples_across_tasks.extend(normalised_ripples[session][task])
        
        print(f"mean of ripples is {np.mean(all_ripples_across_tasks)} in session {session}")
        kde = gaussian_kde(all_ripples_across_tasks)
        x_values = np.linspace(min(all_ripples_across_tasks), max(all_ripples_across_tasks), 1000)
    
        # Evaluate the KDE on these x-values
        y_values = kde(x_values)
        
        # Normalize the y-values so that the maximum y-value is 1
        y_values_normalized = y_values / max(y_values)
        
        # Plot the normalized KDE
        plt.plot(x_values, y_values_normalized, color=colors_kde[sesh_i], label='Ripple Distribution')
        
        # Fill under the curve if you want to mimic the 'fill=True' from sns.kdeplot
        plt.fill_between(x_values, y_values_normalized, color=colors_kde[sesh_i], alpha=0.3)
        
    
    normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]

    # Add a vertical line for the baseline reference
    plt.axvline(x=normalized_sections[0][1], color='black', linestyle='--', label='Found all 4 rewards')
    plt.axvline(x=normalized_sections[1][1], color='black', linestyle='--', label='First Correct')

    
    # Add titles and labels
    plt.title(f"Ripples across tasks and sessions for subject {sub}")
    
    
    # compute the mean of each section duration.
    average_duration_per_session = {}
    for sessions in durations:
        first_bit = []
        second_bit = []
        third_bit = []
        for task in durations[sessions]:
            first_bit.append(durations[sessions][task][0])
            second_bit.append(durations[sessions][task][1])
            third_bit.append(durations[sessions][task][2])
        average_duration_per_session[f"{sessions}_first"] = first_bit
        average_duration_per_session[f"{sessions}_scnd"] = second_bit
        average_duration_per_session[f"{sessions}_third"] = third_bit
        
    
    
    
    mean_one = round(np.mean(average_duration_per_session[f"{list(durations.keys())[0]}_first"]), 2)
    mean_two = round(np.mean(average_duration_per_session[f"{list(durations.keys())[0]}_scnd"]),2)
    mean_three = round(np.mean(average_duration_per_session[f"{list(durations.keys())[0]}_third"]),2)
    plt.suptitle(f"Mean duration until found all rewards = {mean_one}, until first correct = {mean_two}, all other reps = {mean_three}")
    
    if sesh_i == 0:
        plt.xlabel(f"Normalised into 3 sections")
    elif sesh_i == 1:
        plt.xlabel(f"Normalised into 3 sections, {colors[0]} = session 1, {colors[1]} = session 2")
    elif sesh_i == 2:
        plt.xlabel(f"Normalised into 3 sections, {colors[0]} = session 1, {colors[1]} = session 2, {colors[2]} = session 3")
    plt.xticks(ticks=[0.15, 0.48, 0.79], labels=['finding ABCD for the first time', 'first correct solve', 'all other solves'])
    plt.ylabel('Ripple KDE')
    
    plt.xlim(0,1)
    plt.ylim(0, 1.2)
    
    # Add a legend
    # Get current handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Limit the legend to show only the first 6 items
    # plt.legend(handles[:8], labels[:8], loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    # import pdb; pdb.set_trace()
    
    # Show the plot
    plt.show()
    

    
    
    
    
    



    
    
def plot_ripple_distribution(onset_in_secs_dict, task_to_check, feedback_error_curr_task, feedback_correct_curr_task, ripples_across_channels, found_first_D, seconds_upper, index_upper, index_lower, seconds_lower, sub):
    # if task_to_check == 24:
    #     import pdb; pdb.set_trace()
    
    # y_jitter = {key: np.random.uniform(0, 0.01, len(values)) for key, values in onset_in_secs_dict[task_to_check].items()}
    # colors = plt.cm.get_cmap('tab10', len(onset_in_secs_dict[task_to_check]))
    # Define a function to extract the prefix (first 6 characters or customize as needed)
    def get_prefix(condition):
        return condition[:6]  # Adjust the slice based on how much of the key you want to use as prefix
    
    # Group by prefixes and assign y ranges to each group
    prefixes = set(get_prefix(key) for key in onset_in_secs_dict[task_to_check])
    y_ranges = {prefix: i for i, prefix in enumerate(sorted(prefixes))}
    
    # Create y-jitter based on the prefix group
    y_jitter = {}
    for condition, values in onset_in_secs_dict[task_to_check].items():
        prefix = get_prefix(condition)
        base_y = (y_ranges[prefix]+1)*0.15  # Base y-position for this prefix group
        y_jitter[condition] = base_y + np.random.uniform(0, 0.08, len(values))  # Jitter within a small range
    
    # Define the colors: specific for Ha08, Ha09, Hb08, Hb09; grey otherwise
    color_mapping = {
        'Ha08': 'purple',  # Set the desired color for Ha08
        'HaEa08': 'purple',  # Set the desired color for Ha08
        'Ha09': 'maroon',  # Set the desired color for Ha09
        'HaEa09': 'maroon',  # Set the desired color for Ha09
        'Hb08': 'teal',    # Set the desired color for Hb08
        'HbE08': 'teal',    # Set the desired color for Hb08
        'Hb09': 'lightblue',  # Set the desired color for Hb09
        'HbE09': 'lightblue'  # Set the desired color for Hb09
    }
    default_color = 'grey'
    
    # import pdb; pdb.set_trace() 
    # Define marker types based on key starting letter
    def get_marker(condition):
        if condition.startswith('R'):
            return 'o'  # Dot for 'R'
        elif condition.startswith('L'):
            return 'x'  # Cross for 'L'
        return 'o'  # Default to dot if no match
    
    # Define color function based on condition
    def get_color(condition):
        for key in color_mapping.keys():
            if key in condition:
                return color_mapping[key]
        return default_color
    
    # Create the scatter plot
    plt.figure()
    for condition, values in onset_in_secs_dict[task_to_check].items():
        color = get_color(condition)
        marker = get_marker(condition)
        # import pdb; pdb.set_trace() 
        plt.scatter(values, y_jitter[condition], color=color, marker=marker, label=condition, zorder=1, alpha = 0.5)
    
    # Calculate KDE using scipy's gaussian_kde
    if len(ripples_across_channels) > 2:
        
        kde = gaussian_kde(ripples_across_channels)
        x_values = np.linspace(min(ripples_across_channels), max(ripples_across_channels), 1000)
    
        # Evaluate the KDE on these x-values
        y_values = kde(x_values)
        
        # Normalize the y-values so that the maximum y-value is 1
        y_values_normalized = y_values / max(y_values)
        
        # Plot the normalized KDE
        plt.plot(x_values, y_values_normalized, color='lavender', label='Ripple Distribution')
        
        # Fill under the curve if you want to mimic the 'fill=True' from sns.kdeplot
        plt.fill_between(x_values, y_values_normalized, color='lavender', alpha=0.4)

    # sns.kdeplot(ripples_across_channels, fill=True, color='skyblue', label='Ripple Distribution')
    
    # Add a vertical line for the baseline reference
    plt.axvline(x=(found_first_D[task_to_check-1]), color='black', linestyle='--', label='Found all 4 rewards')
    plt.axvline(x=(seconds_upper[index_lower]), color='black', linestyle='--', label='First Correct')

    
    # Add red rods for feedback: incorrect
    sns.rugplot(feedback_error_curr_task, height=0.1, color='salmon', lw=2)  # Each data point as a 'rug'

    # Add green rods for feedback: correct
    sns.rugplot(feedback_correct_curr_task, height=0.1, color='yellowgreen', lw=2)  # Each data point as a 'rug'

    
    # Add titles and labels
    plt.title(f"Ripple count (clustered by wire) for grid {task_to_check} for subj {sub} [10 correct repeats]")
    plt.xlabel('Seconds in Task')
    plt.ylabel('Ripple Frequency')
    
    plt.xlim(seconds_lower[index_lower], seconds_upper[index_upper])
    plt.ylim(0, 1.2)
    
    # Add a legend
    # Get current handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Limit the legend to show only the first 6 items
    plt.legend(handles[:8], labels[:8], loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
def plot_ripple(freq_to_plot, title, downsampled_data, event, min_length_ripple, filtered_cropped_vhgamma_np, power_dict, repeat, freq_bands_keys, y_label_power):
    # event[0] = onset
    # event[1] = duration
    # event[-1] = channel index
    fig, axs = plt.subplots(4)
    fig.suptitle(title)

    # Create x-values from 5500 to 9500
    # x = np.linspace(event[0]-freq_to_plot, event[0]+freq_to_plot-1, freq_to_plot*2)
    x = np.arange(-freq_to_plot, freq_to_plot) * 2
    
    # import pdb; pdb.set_trace()
    # first subplot is the raw signal:
    # axs[0].plot(x, raw_np_cropped[event[0]-freq_to_plot:event[0]+freq_to_plot, event[-1]], linewidth = 0.2)
    axs[0].plot(x, downsampled_data[event[0]-freq_to_plot:event[0]+freq_to_plot, event[-1]], linewidth = 0.2)
    axs[0].set_title('downsampled raw LFP')
    axs[0].set_xlabel('Time (ms)')
    # set a few x-ticks around the ripple in 20ms resolution. 
    axs[0].set_xticks(np.arange(0-6*min_length_ripple, 7*min_length_ripple, 3*min_length_ripple))  # Set x-ticks from 0 to 10 with a step of 0.5


    # the second subplot will be filtered for high gamma:
    axs[1].plot(x, filtered_cropped_vhgamma_np[event[-1], event[0]-freq_to_plot:event[0]+freq_to_plot], linewidth = 0.2)    
    axs[1].set_title('hgamma filtered signal')   
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_xticks(np.arange(0-6*min_length_ripple, 7*min_length_ripple, 3*min_length_ripple))  # Set x-ticks from 0 to 10 with a step of 0.5

    
    # the third subplot will be the mean power of this frequency: 
    axs[2].plot(x,power_dict[f"{repeat}_mean"]['hgamma'][event[-1], event[0]-freq_to_plot:event[0]+freq_to_plot])
    axs[2].set_title('Mean power hgamma')
    axs[2].set_xticks(np.arange(0-6*min_length_ripple, 7*min_length_ripple, 3*min_length_ripple))  # Set x-ticks from 0 to 10 with a step of 0.5

    ## the fourth subplot is the vhgamma power spectrum
    #power_to_plot_low = power_dict[f"{repeat}_stepwise"]['vhgamma'][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot] # Select first epoch and the specified channel
    #axs[3].imshow(power_to_plot_low, aspect='auto', origin='lower')
    
    # the fifth subplot is the overall power spectrum
    # power_to_plot_all = np.stack(power_all[freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq])
    power_to_plot_all = np.vstack((power_dict[f"{repeat}_stepwise"][freq_bands_keys[0]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot], power_dict[f"{repeat}_stepwise"][freq_bands_keys[1]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
    power_to_plot_all = np.vstack((power_to_plot_all, power_dict[f"{repeat}_stepwise"][freq_bands_keys[2]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
    power_to_plot_all = np.vstack((power_to_plot_all, power_dict[f"{repeat}_stepwise"][freq_bands_keys[3]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
    axs[3].imshow(power_to_plot_all, aspect='auto', origin='lower')
    y_ticks = [5, 25, 45, 65, power_to_plot_all.shape[0] - 1]  # 5, 25, 45, 65 and the max value (y-axis max)

    # Setting the yticks and the labels
    axs[3].set_title('Power spectra of all frequency bands stacked')
    axs[3].set_yticks(y_ticks[:-1])  # Add the desired tick positions except the last one
    axs[3].set_yticks([y_ticks[-1]], minor=True)  # Add the max position as a minor tick
    axs[3].set_yticklabels(y_label_power)  # Add the desired tick labels
    plt.tight_layout()
    
    
    
    
   
# import pdb; pdb.set_trace()     

def ripple_count_barchart(ripples_by_state_per_repeat, n_bars, titlestring, xstring, ystring):
    # Initialize the heights list for 40 bars
    heights = []
    # Loop from bar 1 to n-bars and get the corresponding value from the dictionary
    for i in range(1, n_bars+1):
        if i in ripples_by_state_per_repeat:
            heights.append(ripples_by_state_per_repeat[i])  # Use the value from the dictionary
        else:
            heights.append(0)  # If no key exists for this index, append 0
    
    # Generate labels for the x-axis (1 to n-bars)
    labels = [str(i) for i in range(1, n_bars+1)]
    
    plt.figure();
    # Create the bar plot
    plt.bar(labels, heights)
    
    # Add title and labels
    plt.title(titlestring)
    plt.xlabel(xstring)
    plt.ylabel(ystring)
    
    # Show the plot
    plt.show()
    
    
    
def ripple_amount_violin_scatter(plotting_dict, titlestring, xstring, ystring):
    
    plt.figure(figsize=(8, 6))
    
    # Extract the data for boxplots into a list of lists.
    #data_for_boxplots = [plotting_dict[key] for key in sorted(plotting_dict.keys(), key=lambda x: int(x))]
    data_for_boxplots = [plotting_dict[key] for key in sorted(plotting_dict.keys())]
    nan_filtered_data = []
    for i, sublist in enumerate(data_for_boxplots):
        nan_filtered_data.append([x for x in data_for_boxplots[i] if not math.isnan(x)])
        if nan_filtered_data[i] == []:
            nan_filtered_data[i] = 0

    nan_filtered_data = [list(filter(lambda x: not np.isnan(x), sublist)) for sublist in data_for_boxplots]
    for i, sublist in enumerate(nan_filtered_data):
        if sublist == []:
            nan_filtered_data[i] = 0
    # Plot boxplots
    #plt.boxplot(data_for_boxplots, widths=0.6, patch_artist=True)
    plt.violinplot(nan_filtered_data, showmeans=True, widths=1)
    
    # Add scatter points for each boxplot
    for idx, (key, data_points) in enumerate(sorted(plotting_dict.items())):
        # Add some jitter to avoid overlap of scatter points
        jitter = np.random.uniform(-0.2, 0.2, size=len(data_points))  # Add small random jitter
        # Plot scatter points with jitter
        plt.scatter(np.ones(len(data_points)) * (idx + 1) + jitter, data_points, color='salmon', alpha=0.6, linewidths= 0.5)
    
    max_repeats = int(len(plotting_dict)/4)
    x_tick_labels = max_repeats*['A', 'B', 'C', 'D']
    
    for i in range(0, max_repeats):
        plt.hlines(-0.1, i*4+1, i*4+4, colors='grey', linestyles='solid')
    # Set x-ticks to dictionary keys (boxplot labels)
    plt.xticks(np.arange(1, len(plotting_dict) + 1), x_tick_labels)
    
    # Label and title
    plt.xlabel(xstring)
    
    plt.ylabel(ystring)
    plt.title(titlestring)
    plt.show()        
          
        

   
    
    
    
def ripples_compare_two_bars(data_dict, title_string, y_label_string, ):

    # Convert dictionary to two lists for plotting
    labels = list(data_dict.keys())  # ['overall', 'rest']

    if len(labels) == 2:
        group_one = np.array(data_dict[labels[0]])
        group_two = np.array(data_dict[labels[1]])
    else:
        print('careful! this is written for 2 bars. adjust dictionary!')
        return

    # Calculate mean and standard error for bars
    means = [np.nanmean(group_one), np.nanmean(group_two)]
    errors = [[0,0],[np.nanstd(group_one), np.nanstd(group_two)]]
    
    # import pdb; pdb.set_trace() 
    # Create figure and axis
    plt.figure(figsize=(5, 6))
    
    # Plot the bars
    bars = plt.bar(labels, means, yerr=errors, capsize=5, color=['grey', 'lightblue'], alpha=0.7)
    
    # Plot scatter points for individual data points
    #for i in range(len(group_one)):
    plt.scatter(np.zeros(len(group_one)), group_one, color='gray', alpha=0.6, zorder=3)
    plt.scatter(np.ones(len(group_two)), group_two, color='lightblue', alpha=0.6, zorder=3)   
    #plt.plot([0, 1], [group_one[i], group_two[i]], color='gray', alpha=0.5, zorder=2)  # Connecting lines
    
    # Adding the significance indicator (***)
    # Perform a t-test between the two lists
    # import pdb; pdb.set_trace() 
    t_stat, p_value = stats.ttest_rel(group_one, group_two, nan_policy='omit')
    
    # Determine number of stars based on p-value
    if p_value > 0.05:
        stars = 'n.s.'  # No significance
    elif p_value > 0.01:
        stars = '*'
    elif p_value > 0.005:
        stars = '**'
    else:
        stars = '***'

    plt.text(0.5, 0.55, stars, ha='center', va='bottom', fontsize=20)
    
    # Adding labels
    plt.ylabel(y_label_string)
    plt.xticks(ticks=[0, 1], labels=labels)
    plt.title(f"{title_string}, t = {round(t_stat,4)}, p = {round(p_value, 4)}", fontsize=14)
    
    # Remove top and right spines
    # sns.despine()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

    
