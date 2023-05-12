#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:55:36 2023

@author: Svenja Küchenhoff
this script runs the entire analysis of Mohamadys data
"""

import numpy as np
import mc
import matplotlib.pyplot as plt
import joypy
from matplotlib import cm
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "browser"
import scipy.stats 

#
# Part 1: load data
#
Data_folder='/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/' 

mouse_recday='me11_05122021_06122021' #mouse a
a_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
a_no_task_configs = len(a_rewards_configs)
a_locations = list()
a_neurons = list()
a_timings = list()
for session in range(0, a_no_task_configs):
    a_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    a_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    a_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))


mouse_recday='me11_01122021_02122021' #mouse b 
b_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
b_no_task_configs = len(b_rewards_configs)
b_locations = list()
b_neurons = list()
b_timings = list()
for session in range(0, b_no_task_configs):
    b_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    b_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    b_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))



mouse_recday='me10_09122021_10122021' #mouse c range 0,9
c_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
c_no_task_configs = len(c_rewards_configs)
c_locations = list()
c_neurons = list()
c_timings = list()
for session in range(0, c_no_task_configs):
    c_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    c_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    c_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    

mouse_recday='me08_10092021_11092021' #mouse d range 0,6
d_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
# apparently there is one run less for this day...
d_no_task_configs = len(d_rewards_configs) - 1
d_locations = list()
d_neurons = list()
d_timings = list()
for session in range(0, d_no_task_configs):
    d_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    d_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    d_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))



mouse_recday='ah04_09122021_10122021' #mouse e range 0,8
e_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
e_no_task_configs = len(e_rewards_configs)
e_locations = list()
e_neurons = list()
e_timings = list()
for session in range(0, e_no_task_configs):
    e_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    e_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    e_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
 
mouse_recday='ah04_05122021_06122021' #mouse f range 0,8
f_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
f_no_task_configs = len(f_rewards_configs)
f_locations = list()
f_neurons = list()
f_timings = list()
for session in range(0, f_no_task_configs):
    f_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    f_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    f_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))

mouse_recday='ah04_01122021_02122021' #mouse g range 0,8
g_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
g_no_task_configs = len(g_rewards_configs)
g_locations = list()
g_neurons = list()
g_timings = list()
for session in range(0, g_no_task_configs):
    g_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    g_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    g_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))


# mouse_recday='ah03_18082021_19082021' #mouse h range 0,8
# h_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
# h_no_task_configs = len(h_rewards_configs)
# h_locations = list()
# h_neurons = list()
# h_timings = list()
# for session in range(0, h_no_task_configs):
#     h_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
#     h_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
#     h_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
# # for h, the first timings array is missing
# # > delete the first task completely!
# h_timings = h_timings[1::]
# h_neurons = h_neurons[1::]
# h_locations = h_locations[1::]
# h_rewards_configs = h_rewards_configs[1::, :]


#
# Part 2: subject-level analysis: compute betas and contrasts
#

# try how the regression across tasks looks like
#results_reg_acro_mouse_g, scipy_reg_acro_mouse_g, coefficients_acro_mouse_g = mc.simulation.single_sub_ephys.reg_across_tasks(g_rewards_configs, g_locations, g_neurons, g_timings)
#results_reg_acro_mouse_a, scipy_reg_acro_mouse_a, coefficients_acro_mouse_a = mc.simulation.single_sub_ephys.reg_across_tasks(a_rewards_configs, a_locations, a_neurons, a_timings)
#results_reg_acro_mouse_b, scipy_reg_acro_mouse_b, coefficients_acro_mouse_b, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(b_rewards_configs, b_locations, b_neurons, b_timings)

# results_reg_acro_mouse_c, scipy_reg_acro_mouse_c, coefficients_acro_mouse_c, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(c_rewards_configs, c_locations, c_neurons, c_timings)
# results_reg_acro_mouse_d, scipy_reg_acro_mouse_d, coefficients_acro_mouse_d, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(d_rewards_configs, d_locations, d_neurons, d_timings)

results_reg_acro_mouse_e, scipy_reg_acro_mouse_e, coefficients_acro_mouse_e, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(e_rewards_configs, e_locations, e_neurons, e_timings)



# THIS IS THE INTERESTIGN THING RN
import pdb; pdb.set_trace()



contrast_matrix = ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,-1,0,0), (1, 0,-1,0), (1,0,0,-1), (0,1,-1,0), (0,1,0,-1), (0,0,1,-1))
regression_mouse_a,  contrasts_mouse_a = mc.simulation.single_sub_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix)
# now generate the average beta value for each model
mean_beta_clocks_a = list()
mean_beta_midnight_a = list()
mean_beta_locations_a = list()
mean_beta_phase_a = list()
mean_contrasts_mouse_a = np.zeros((len(contrast_matrix), len(regression_mouse_a)))
for task_no, betas in enumerate(regression_mouse_a):
    mean_beta_clocks_a.append(np.mean(betas[:,0]))
    mean_beta_midnight_a.append(np.mean(betas[:,1]))
    mean_beta_locations_a.append(np.mean(betas[:,2]))
    mean_beta_phase_a.append(np.mean(betas[:,3]))
    for contr in range(len(contrast_matrix)):
        mean_contrasts_mouse_a[contr, task_no] = np.mean(contrasts_mouse_a[task_no][contr])



    
regression_mouse_b,  contrasts_mouse_b = mc.simulation.single_sub_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix)
# now generate the average beta value for each model
mean_beta_clocks_b = list()
mean_beta_midnight_b = list()
mean_beta_locations_b = list()
mean_beta_phase_b = list()
mean_contrasts_mouse_b = np.zeros((len(contrast_matrix), len(regression_mouse_b)))
for task_no, betas in enumerate(regression_mouse_b):
    mean_beta_clocks_b.append(np.mean(betas[:,0]))
    mean_beta_midnight_b.append(np.mean(betas[:,1]))
    mean_beta_locations_b.append(np.mean(betas[:,2]))
    mean_beta_phase_b.append(np.mean(betas[:,3]))
    for contr in range(len(contrast_matrix)):
        mean_contrasts_mouse_b[contr, task_no] = np.mean(contrasts_mouse_b[task_no][contr])

# # SOMETHING GOES WRONG HERE, SUPER LARGE betas
# regression_mouse_c,  contrasts_mouse_c = mc.simulation.single_sub_ephys.reg_per_task_config(c_rewards_configs, c_locations, c_neurons, c_timings, contrast_matrix)
# # for some reason, for the 5th run (timepoints:[741, 776, 794, 811, 818])
# # the mouse just stays at one location (1)
# # now generate the average beta value for each model
# mean_beta_clocks_c = list()
# mean_beta_midnight_c = list()
# mean_beta_locations_c = list()
# mean_contrasts_mouse_c = np.zeros((len(contrast_matrix), len(regression_mouse_c)))
# for task_no, betas in enumerate(regression_mouse_c):
#     mean_beta_clocks_c.append(np.mean(betas[:,0]))
#     mean_beta_midnight_c.append(np.mean(betas[:,1]))
#     mean_beta_locations_c.append(np.mean(betas[:,2]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_c[contr, task_no] = np.mean(contrasts_mouse_c[task_no][contr])

# # SOMETHING GOES WRONG HERE, SUPER SMALLbetas    
# regression_mouse_d,  contrasts_mouse_d = mc.simulation.single_sub_ephys.reg_per_task_config(d_rewards_configs[0:6,:], d_locations, d_neurons, d_timings, contrast_matrix)
# # now generate the average beta value for each model
# mean_beta_clocks_d = list()
# mean_beta_midnight_d = list()
# mean_beta_locations_d = list()
# mean_contrasts_mouse_d = np.zeros((len(contrast_matrix), len(regression_mouse_d)))
# for task_no, betas in enumerate(regression_mouse_d):
#     mean_beta_clocks_d.append(np.mean(betas[:,0]))
#     mean_beta_midnight_d.append(np.mean(betas[:,1]))
#     mean_beta_locations_d.append(np.mean(betas[:,2]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_d[contr, task_no] = np.mean(contrasts_mouse_d[task_no][contr])

# # SOMETHING GOES WRONG HERE, SUPER SMALLbetas    
# regression_mouse_e,  contrasts_mouse_e = mc.simulation.single_sub_ephys.reg_per_task_config(e_rewards_configs, e_locations, e_neurons, e_timings, contrast_matrix)
# # now generate the average beta value for each model
# mean_beta_clocks_e = list()
# mean_beta_midnight_e = list()
# mean_beta_locations_e = list()
# mean_contrasts_mouse_e = np.zeros((len(contrast_matrix), len(regression_mouse_e)))
# for task_no, betas in enumerate(regression_mouse_e):
#     mean_beta_clocks_e.append(np.mean(betas[:,0]))
#     mean_beta_midnight_e.append(np.mean(betas[:,1]))
#     mean_beta_locations_e.append(np.mean(betas[:,2]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_e[contr, task_no] = np.mean(contrasts_mouse_e[task_no][contr])
    
# regression_mouse_f,  contrasts_mouse_f = mc.simulation.single_sub_ephys.reg_per_task_config(f_rewards_configs, f_locations, f_neurons, f_timings, contrast_matrix)
# # now generate the average beta value for each model
# mean_beta_clocks_f = list()
# mean_beta_midnight_f = list()
# mean_beta_locations_f = list()
# mean_contrasts_mouse_f = np.zeros((len(contrast_matrix), len(regression_mouse_f)))
# for task_no, betas in enumerate(regression_mouse_f):
#     mean_beta_clocks_f.append(np.mean(betas[:,0]))
#     mean_beta_midnight_f.append(np.mean(betas[:,1]))
#     mean_beta_locations_f.append(np.mean(betas[:,2]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_f[contr, task_no] = np.mean(contrasts_mouse_f[task_no][contr])
    
regression_mouse_g,  contrasts_mouse_g = mc.simulation.single_sub_ephys.reg_per_task_config(g_rewards_configs, g_locations, g_neurons, g_timings, contrast_matrix)
# now generate the average beta value for each model
mean_beta_clocks_g = list()
mean_beta_midnight_g = list()
mean_beta_locations_g = list()
mean_beta_phase_g = list()
mean_contrasts_mouse_g = np.zeros((len(contrast_matrix), len(regression_mouse_g)))
for task_no, betas in enumerate(regression_mouse_g):
    mean_beta_clocks_g.append(np.mean(betas[:,0]))
    mean_beta_midnight_g.append(np.mean(betas[:,1]))
    mean_beta_locations_g.append(np.mean(betas[:,2]))   
    mean_beta_phase_g.append(np.mean(betas[:,3]))
    for contr in range(len(contrast_matrix)):
        mean_contrasts_mouse_g[contr, task_no] = np.mean(contrasts_mouse_g[task_no][contr])
 


# regression_mouse_h, contrasts_mouse_h = mc.simulation.single_sub_ephys.reg_per_task_config(h_rewards_configs, h_locations, h_neurons, h_timings, contrast_matrix)
# # now generate the average beta value for each model
# mean_beta_clocks_h = list()
# mean_beta_midnight_h = list()
# mean_beta_locations_h = list()
# mean_contrasts_mouse_h = np.zeros((len(contrast_matrix), len(regression_mouse_h)))
# for task_no, betas in enumerate(regression_mouse_h):
#     mean_beta_clocks_h.append(np.mean(betas[:,0]))
#     mean_beta_midnight_h.append(np.mean(betas[:,1]))
#     mean_beta_locations_h.append(np.mean(betas[:,2]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_h[contr, task_no] = np.mean(contrasts_mouse_h[task_no][contr])


#
# Part 3: Plotting
# plot one violin plot per subject (i.e. 8 violins) where I visualize the variability across tasks per subject
# per contrast 

# mouse a.
# mean_contrasts_mouse_a[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_a[1,:], mean_contrasts_mouse_a[2,:]]
# ylabelstr = ["Only clocks", "Clocks-Midnight", "Clocks-Loc"]
# fig = plt.figure()
# axes = fig.add_axes([0,0,1,1])
# vp = axes.violinplot(set_of_columns)
# plt.show()

# data_contr_1 = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_b[0,:], mean_contrasts_mouse_c[0,:], mean_contrasts_mouse_d[0,:], mean_contrasts_mouse_e[0,:], mean_contrasts_mouse_f[0,:], mean_contrasts_mouse_g[0,:], mean_contrasts_mouse_h[0,:]]
# data_contr_2 = [mean_contrasts_mouse_a[1,:], mean_contrasts_mouse_b[1,:], mean_contrasts_mouse_c[1,:], mean_contrasts_mouse_d[1,:], mean_contrasts_mouse_e[1,:], mean_contrasts_mouse_f[1,:], mean_contrasts_mouse_g[1,:], mean_contrasts_mouse_h[1,:]]
# data_contr_3 = [mean_contrasts_mouse_a[2,:], mean_contrasts_mouse_b[2,:], mean_contrasts_mouse_c[2,:], mean_contrasts_mouse_d[2,:], mean_contrasts_mouse_e[2,:], mean_contrasts_mouse_f[2,:], mean_contrasts_mouse_g[2,:], mean_contrasts_mouse_h[2,:]]



contr_1_mean_clock = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_b[0,:],  mean_contrasts_mouse_g[0,:]]
contr_2_mean_midnight = [mean_contrasts_mouse_a[1,:], mean_contrasts_mouse_b[1,:], mean_contrasts_mouse_g[1,:]]
contr_3_mean_location = [mean_contrasts_mouse_a[2,:], mean_contrasts_mouse_b[2,:], mean_contrasts_mouse_g[2,:]]
contr_4_mean_phase = [mean_contrasts_mouse_a[3,:], mean_contrasts_mouse_b[3,:], mean_contrasts_mouse_g[3,:]]

contr_5_clocks_midnight = [mean_contrasts_mouse_a[4,:], mean_contrasts_mouse_b[4,:], mean_contrasts_mouse_g[4,:]]
contr_6_clocks_location = [mean_contrasts_mouse_a[5,:], mean_contrasts_mouse_b[5,:], mean_contrasts_mouse_g[5,:]]
contr_7_clocks_phase = [mean_contrasts_mouse_a[6,:], mean_contrasts_mouse_b[6,:], mean_contrasts_mouse_g[6,:]]


                                         
fig_one, ax_one = joypy.joyplot(contr_1_mean_clock, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
plt.title('Mean clocks, distr. across tasks per mouse [1 0 0 0]')
plt.show()

fig_two, ax_two = joypy.joyplot(contr_2_mean_midnight, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
plt.title('Mean midnight, distr. across tasks per mouse [0 1 0 0]')
plt.show()

fig_three, ax_three = joypy.joyplot(contr_3_mean_location, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
plt.title('Mean location, distr. across tasks per mouse [0 0 1 0]')
plt.show()

fig_four, ax_four = joypy.joyplot(contr_4_mean_phase, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
plt.title('Mean phase, distr. across tasks per mouse [0 0 0 1]')
plt.show()


#########

fig_contr_one, ax_contr_one = joypy.joyplot(contr_7_clocks_phase, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.3,0.3], fade = True)
plt.title('Clocks-phase, distr. across tasks per mouse [1 0 0 -1]')
plt.show()

fig_contr_two, axcontr__two = joypy.joyplot(contr_5_clocks_midnight, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.3,0.3], fade = True)
plt.title('Clocks-midnight contrast, distr. across tasks per mouse [1 -1 0 0]')
plt.show()

fig_contr_three, ax_contr_three = joypy.joyplot(contr_6_clocks_location, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.3,0.3], fade = True)
plt.title('Clocks-location contrast, distr. across tasks per mouse [1 0 -1 0]')
plt.show()





# plotting the mean across runs for each task for each beta separateyl (1 0 0 0,...)
x1 = np.array(mean_beta_clocks_a)
x2 = np.array(mean_beta_midnight_a)
x3 = np.array(mean_beta_locations_a)
x4 = np.array(mean_beta_phase_a)
hist_data = [x1, x2, x3, x4]

# group_labels = ['Clocks', 'Midnight', 'Location']
# colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

# # Create distplot with curve_type set to 'normal'
# fig = ff.create_distplot(hist_data, group_labels, colors=colors,
#                          bin_size=.2, show_rug=False)

# # Add title
# fig.update_layout(title_text='Mouse 1, mean beta per task configuration')
# fig.show()

fig, ax = plt.subplots()
ax.violinplot(hist_data, showmedians = True)
ax.set_title('Beta weights for a single mouse across task configs')
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(["Clocks", "Midnight", "Location", "Phase"])


contr_5_clocks_midnight = np.concatenate((mean_contrasts_mouse_a[4,:], mean_contrasts_mouse_b[4,:], mean_contrasts_mouse_g[4,:]), axis = None)
contr_6_clocks_location = np.concatenate((mean_contrasts_mouse_a[5,:], mean_contrasts_mouse_b[5,:], mean_contrasts_mouse_g[5,:]), axis = None)
contr_7_clocks_phase = np.concatenate((mean_contrasts_mouse_a[6,:], mean_contrasts_mouse_b[6,:], mean_contrasts_mouse_g[6,:]), axis = None)

# tstat of these values
t_stat_contr_5 = scipy.stats.ttest_1samp(contr_5_clocks_midnight, 0)


hist_data_contrasts = [contr_5_clocks_midnight, contr_6_clocks_location, contr_7_clocks_phase]
fig_con, ax_con = plt.subplots()
ax_con.violinplot(hist_data_contrasts, showmedians = True)
ax_con.set_title('Contrasts for 3 mice across tasks and runs')
ax_con.set_xticks([1,2,3])
ax_con.set_xticklabels(["Clocks - Midnight", "Clocks-Location", "Clocks-Phase"])



#
# Part 4: Group level analysis
#



    
# GROUP LEVEL ANALYSIS next.
# problem: there might be outliers!!
# coorperate a breakpoint if beta > 100 or so to check what is going wrong with the GLM in this case



    
    # 1. subject level.
    # for every mouse and every run, compute a GLM with my 3 regressors.
    # 2. compute contrasts:
    # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
    # (every regressor at its own, and the cotnrast between 2 betas (MRI: PEs) which is 
    # 1 minus the other)
    # take all of these values and average 
    #        1. across runs within one task config
    #       2. across task configs
    # Finally, you end up with 9 betas (MRI: COPEs) for every mouse (contrasts)
    # 3. Group level:
        # compute a random effects GLM for every of the contrasts, using
        # each mouse-beta as an input
    
    
    
    # next step: Stats! > group statistics?
    # > multiple runs? use the regressor model since there are different fields the mouse runs on? 
    # across tasks?? 
    # also check the correlation values, independent from the other regressors
    
    # before: within a mouse > fixed effects (+averaging betas) to compare runs 
    
    
    # last step: check ou FSL FEAT -> compare betas across mice with random effects if 
    # thats possible with only 8 mice, otherwise fixed effects 
    
    
    
    # potentially, also have a look at a second thing:
        # concatenate all trials across task cofngis
        # then reduce the size of the data to steps instead of ms
        # by using the step-regressors from the fMRI model (or the other way around)
    # afterwards, follow the same group-stats and contrasts
    # these contrasts are probably even more significant and will be more like my fmRI analyssi
    




