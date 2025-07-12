% extracts all behavioural data from the ABCD per session
% such that it can be used for my LFP analysis
% saving one behavioural file per session that is structured as follows:
% behaviour_all(1,1) = curr_repeat
% behaviour_all(1,2) = found_A
% behaviour_all(1,3) = found_B
% behaviour_all(1,4) = found_C
% behaviour_all(1,5) = found_D
% behaviour_all(1,6) = loc_A
% behaviour_all(1,7) = loc_B
% behaviour_all(1,8) = loc_C
% behaviour_all(1,9) = loc_D
% behaviour_all(1,10) = new_grid_onset
% behaviour_all(1,11) = recording ??
% behaviour_all(1,12) = grid_no

clear all
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys'
    %abcd_data = load(sprintf("%s/beh_cells/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_24-Apr-2025.mat", source_dir));
    
else
    %abcd_data = load(sprintf("%s/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
    abcd_data = load(sprintf("%s/abcd_data_10-Jul-2025.mat", source_dir));
end

deriv_dir = sprintf("%s/derivatives/", source_dir);

subject_list = 1:length(abcd_data.abcd_data);

for sub = 1:length(subject_list)
    n = numel(abcd_data.abcd_data(sub).trial_vars); % Number of elements in the structured array
    for i = 1:n
        % Access the state_change_times for the current element
        current_times = abcd_data.abcd_data(sub).trial_vars(i).state_change_times;
        % Check if the data is in row format (1xN)
        if isrow(current_times)
            % Convert from row to column vector
            abcd_data.abcd_data(sub).trial_vars(i).state_change_times = current_times';
        end
    end
end


for sub = 1:length(subject_list)
    subj = abcd_data.abcd_data(sub);
    subject_folder = sprintf("%ss%02d/cells_and_beh", deriv_dir, sub);
    if ~exist(subject_folder, 'dir')
        mkdir(subject_folder); % Create the folder if it does not exist
        disp(['Folder created: ', subject_folder]);
    end

    t_found_reward = [subj.trial_vars.state_change_times]';
    configurations = [subj.trial_vars.sequence_locations]';
    grid_num = [subj.trial_vars.grid_num]';
    trial_rep_in_grid_correct = [subj.trial_vars.trial_num_in_grid_correct]';
    trial_rep_in_grid = [subj.trial_vars.trial_num_in_grid]';
    session_no = [subj.trial_vars.session_num]';
    end_trials = [subj.trial_vars.end_trial_timestamp]';
    
    new_trial_onset = [arrayfun(@(s) s.grid_onset_timestamp(1), subj.trial_vars)]';
    %new_trial_onset = [subj.trial_vars(1).rule_onset_timestamp;end_trials(1:end-1,:)];

    behaviour_all = [trial_rep_in_grid_correct, t_found_reward, configurations, trial_rep_in_grid, new_trial_onset, session_no, grid_num];


    % NOW FIGURE OUT HOW TO PUT IT ALL TOGETHER

    csvwrite(sprintf("%s/all_trial_times_%02d.csv", subject_folder, sub), behaviour_all);
    disp('now storing:')
    disp(sprintf("%s/all_trial_times_%02d.csv", subject_folder, sub))
end

