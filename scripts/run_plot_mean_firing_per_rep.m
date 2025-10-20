% script to plot a couple of basic things in task time.

clear all
do_plot = false;  % toggle plotting

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys'
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_08-Sep-2025.mat", source_dir));  
else
    abcd_data = load(sprintf("%s/derivatives/abcd_passed.mat", source_dir));
end

% subject_list = 1:length(abcd_data.abcd_data);
subject_list = 1:length(abcd_data.abcd_passed.abcd_data);
subject_list = 60:63;


% LOOP THROUGH SUBJECTS
for sub = 1:length(subject_list)
    subj = abcd_data.abcd_passed.abcd_data(sub);
    %plot_mean_firing_rate(subj, 10, 'zscore');  % or 'zscore'
    plot_FR_reward_windows_by_repeat(subj, 10, 2);
end

disp('DONE!');


% do rates anti-correlate with duration?
% -> not really!
% corr(duration, rate) = -0.014 (p=0.89)
% corr(duration, rate) = -0.044 (p=0.576)
% corr(duration, rate) = 0.021 (p=0.795)
% corr(duration, rate) = 0.115 (p=0.093)
% corr(duration, rate) = 0.054 (p=0.578)
% corr(duration, rate) = 0.156 (p=0.0217)
% corr(duration, rate) = -0.014 (p=0.833)
% corr(duration, rate) = -0.141 (p=0.0387)
% corr(duration, rate) = -0.041 (p=0.553)