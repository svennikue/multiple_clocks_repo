% extract which subjects are the ones that have the same grids.
% then reshape the cells such that they are formatted as 
% firing rate per 50ms.
% then also recode the locations as location per 50ms bins.

clear all
abcd_data = load("abcd_data_09-Dec-2024.mat");
deriv_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/"

% some problem with subject 27 and 44 , I think because there are NaNs at some
% point
subject_list = 1:length(abcd_data.abcd_data);

conf_all_subs = cell(length(subject_list),1); 
unique_grids_all_subs = cell(length(subject_list),1);


% first find out which subjects share the same grids
for s = 1:length(subject_list)
    subj = abcd_data.abcd_data(s);
    configurations = [subj.trial_vars.sequence_locations]';
    trial_num_in_grid = [subj.trial_vars.trial_num_in_grid]';
    grid_num = [subj.trial_vars.grid_num]';
    new_grid_after = find(diff(grid_num) ~=0);
    new_grid_at = new_grid_after+1;
    all_configs(1,:) = [configurations(1,:)];
    % loop through subj and identify where grid_num changes
    for g = 1:length(new_grid_at)
        if trial_num_in_grid(new_grid_at(g)) == 1
            all_configs(g+1,:) = [configurations(new_grid_at(g),:)];
        end
    end
    % to later count grid occurences
    % grid_count = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    grids = {};
    for n_grid = 1:length(all_configs)
        grid_key = mat2str(reshape(all_configs(n_grid, :), 1, []));
        grids{end+1} = grid_key;
    end
    % this stores those grids that are unique per subject
    unique_grids_all_subs{s} = unique(grids);
    % this stores all configureations per subject
    conf_all_subs{s} = all_configs;
    clear all_configs
end

% I know that session 50 was one of the subjects that shared the same grids
common_grids = unique_grids_all_subs{50};
subjects_with_same_grids = 50;
for s = 1:(length(subject_list)-1)
    potential_common = intersect(common_grids, unique_grids_all_subs{s});
    if ~isempty(potential_common)
        %for now exclude s27 because there is an error.
        if s == 27
            continue
        end
        if s == 44 
            continue
        end
        subjects_with_same_grids = [subjects_with_same_grids, s];
    end
end


% script to extract the interesting cells and save them as csv
% Restructure them as well to analyse them analoguesly to the mouse data:
% save reward locations per task [n tasks x 4] and cell label per subject

% save cells in bins of 50ms -> encode firing rate per 50ms [n cells x time points]
% save location in bins of 50ms -> [location x  time points]
% save timings of state start + reward start for ABCD = [n repeats x 8
% timings]
% save all three of them in separate csv-s; split by task 

start_time = 0;
bin_size = 0.025; %play around with this! Mohamady has 0.025

for s = 1:length(subjects_with_same_grids)
    all_cells = [];
    region_labels_cells = {};
    sub = subjects_with_same_grids(s);
    subj = abcd_data.abcd_data(sub);
    subject_folder = sprintf("%ss%02d/LFP", deriv_dir, sub);
    t_found_reward = [subj.trial_vars.state_change_times]';
    configurations = [subj.trial_vars.sequence_locations]';
    grid_num = [subj.trial_vars.grid_num]';
    configs = [];


    % then extract all cells per subject and code them
    % as firing rate in a 50 ms timewindow
    % for the neural data, I am creating bins in one go from
    % 0 seconds to end_time
    end_time = subj.trial_vars(end).button_pressed_timestamp(end);
    for c = 1:length(subj.neural_data)
        curr_spike_times = subj.neural_data(c).spikeTimes;
        edges = start_time:bin_size:end_time;
        [firing_rate_curr_cell, edges] = histcounts(curr_spike_times, edges);
        region_labels_cells{c} = subj.neural_data(c).regionLabel;
        all_cells = [all_cells; firing_rate_curr_cell];
    end

    % per subject, also extract all grid configs and all cell labels
    filename = (sprintf("%s/all_cells_region_labels_sub%d.txt", subject_folder, sub));
    fid = fopen(filename, 'wt');  % Open a text file for writing
    % Write each label on a new line
    for i = 1:length(region_labels_cells)
        fprintf(fid, '%s\n', region_labels_cells{i});
    end
    fclose(fid);  % Close the file

    % concatenate the positions and position changes for each grid and repeat
    for c = 1:length(subj.trial_vars)
        curr_position_time = subj.trial_vars(c).grid_onset_timestamp;
        curr_position = subj.trial_vars(c).start_location;
        if c == 1
            all_positions_time = curr_position_time;
            all_positions = curr_position;
        else
            all_positions_time = [all_positions_time, curr_position_time];
            all_positions = [all_positions, curr_position];
        end
    end
    num_windows = length(all_cells);
    locations_per_50ms = zeros(1, num_windows);

    % finally, also store location data in the same format.
    % to do that, first identify from which to which bin they stayed in the
    % same location.
    for change_loc = 1:length(all_positions)
        current_location = all_positions(change_loc);
        if change_loc == 1
            first_bin_to_fill = 1;
        else 
            first_bin_to_fill = ceil(all_positions_time(change_loc)/bin_size);
        end
        if change_loc < length(all_positions)
            last_bin_to_fill = ceil(all_positions_time(change_loc+1)/bin_size);
        else
            last_bin_to_fill = length(locations_per_50ms);
        end
        locations_per_50ms(first_bin_to_fill:last_bin_to_fill) = current_location;
    end

    disp(sprintf("length cell recordings is %d and location bins is %d ", length(all_cells), length(locations_per_50ms)))

    % finally, split the cells, locations and timings by repeat.
    index_last_repeat = find(diff(grid_num) ~= 0); 
    index_last_repeat = [index_last_repeat; length(grid_num)];
    for g_i = 1:length(index_last_repeat) % loop through every repeat
        % first do the timings
        last_repeat=index_last_repeat(g_i);
        % also track the configurations to store later
        configs = [configs; configurations(last_repeat, :)];
        % the identify which bins correspond to timings of current repeat
        if g_i == 1
            timings_curr_grid = t_found_reward(1:last_repeat, :);
            % if first repeat, then this needs to be the start of the task
            start_idx = [subj.trial_vars(1).grid_onset_timestamp(1)];
            start_looking_for_A = arrayfun(@(x) x.grid_onset_timestamp(1), subj.trial_vars(1:last_repeat));
        else
            prev_last_repeat = index_last_repeat(g_i-1);
            %t_prev_end_trial = subj.trial_vars(prev_last_repeat).end_trial_timestamp;
            %start_idx = floor(t_prev_end_trial/ bin_size) + 1;
            timings_curr_grid = t_found_reward(prev_last_repeat+1:last_repeat, :);
            start_looking_for_A = arrayfun(@(x) x.grid_onset_timestamp(1), subj.trial_vars(prev_last_repeat+1:last_repeat));
            start_idx = [subj.trial_vars(prev_last_repeat+1).grid_onset_timestamp(1)];
        end

        % this is too simple!!
        % actually, it start A isn't equal to finding the previous D. It
        % should be when they are pressing the next button.
        %
        %
        % look in table where i can find this!!
         
        % Extract the last column and create the new first column
        %start_looking_for_A = [start_idx; timings_curr_grid(1:end-1, end)];
        % Prepend this new column to the existing matrix
        timings_curr_grid = [start_looking_for_A', timings_curr_grid];
        
        start_idx_bin = ceil(start_idx/bin_size);
        t_end_trial = subj.trial_vars(last_repeat).end_trial_timestamp;
        % note that because t_end_trial is always a bit later than finding
        % the last grid, location and cell arrays a slightly longer than
        % timings!
        end_idx_bin = ceil(t_end_trial/ bin_size);
        if end_idx_bin > length(all_cells)
            end_idx_bin = length(all_cells);
        end
       
        all_cells_curr_grid = all_cells(:, start_idx_bin:end_idx_bin);
        csvwrite(sprintf("%s/all_cells_firing_rate_grid%d_sub%d.csv", subject_folder, grid_num(last_repeat), sub), all_cells_curr_grid);
        locations_per_50ms_curr_grid = locations_per_50ms(start_idx_bin:end_idx_bin);
        csvwrite(sprintf("%s/locations_per_50ms_grid%d_sub%d.csv", subject_folder, grid_num(last_repeat), sub), locations_per_50ms_curr_grid);
        
        % change timings to bins that always start with the first bin that
        % I cut the timings and locations to 
        timings_curr_grid_in_bins = ceil(timings_curr_grid/bin_size)-start_idx_bin; 

        disp(sprintf("cell length in repeat %d is %d , loc length is %d and last timing is %d", g_i, size(all_cells_curr_grid,2), size(locations_per_50ms_curr_grid,2), timings_curr_grid_in_bins(end)))
        
        % save the timings in the binned format
        csvwrite(sprintf("%s/timings_rewards_grid%d_sub%d.csv", subject_folder, grid_num(last_repeat), sub), timings_curr_grid_in_bins);
    end
    % for each subject, save the configurations
    csvwrite(sprintf("%s/all_configs_sub%d.csv", subject_folder, sub), configs);
end


        % if g_i < length(index_last_repeat) % if not last repeat
        %     end_idx_bin = ceil(t_end_trial/ bin_size);
        % else
        %     end_idx_bin = num_windows;  % Go till the end of the time array
        % end

    % %
    % %
    % %
    % 
    % for c = 1:length(subj.trial_vars)
    %     curr_position_time = subj.trial_vars(c).grid_onset_timestamp;
    %     curr_position = subj.trial_vars(c).start_location;
    %     if c == 1
    %         all_positions_time = curr_position_time;
    %         all_positions = curr_position;
    %     else
    %         all_positions_time = [all_positions_time, curr_position_time];
    %         all_positions = [all_positions, curr_position];
    %     end
    % end
    % num_windows = length(all_cells);
    % locations_per_50ms = zeros(1, num_windows);
    % % then fill the location for each time bin
    % current_location = all_positions(1);  % Initialize with the first location
    % for i = 1:length(all_positions_time)
    %     if i == 1
    %         start_idx = 1;  % Start from the first index
    %     else
    %         start_idx = floor(all_positions_time(i-1) / bin_size) + 1;
    %     end
    % 
    %     if i < length(all_positions_time)
    %         end_idx = floor(all_positions_time(i) / bin_size);
    %     else
    %         end_idx = num_windows;  % Go till the end of the time array
    %     end
    %     locations_per_50ms(start_idx:end_idx) = current_location;
    %     if i < length(all_positions)
    %         current_location = all_positions(i);  % Update location for the next iteration
    %     end
    % end
    % 
    % % before saving them, split them by task.
    % for g_i = 1:length(index_last_repeat)
    %     % identify the corresponding bin
    %     if g_i == 1
    %         start_idx = 1;  % Start from the first bin
    %     else
    %         prev_last_repeat = index_last_repeat(g_i-1);
    %         t_prev_end_trial = subj.trial_vars(prev_last_repeat).end_trial_timestamp;
    %         start_idx = floor(t_prev_end_trial/ bin_size) + 1;
    %     end
    %     last_repeat=index_last_repeat(g_i);
    %     t_end_trial = subj.trial_vars(last_repeat).end_trial_timestamp;
    %     if g_i < length(index_last_repeat)
    %         end_idx = floor(t_end_trial/ bin_size);
    %     else
    %         end_idx = num_windows;  % Go till the end of the time array
    %     end
    %     all_cells_curr_grid = all_cells(:, start_idx:end_idx);
    %     csvwrite(sprintf("%s/all_cells_firing_rate_grid%d_sub%d.csv", subject_folder, grid_num(last_repeat), sub), all_cells_curr_grid);
    %     locations_per_50ms_curr_grid = locations_per_50ms(start_idx:end_idx);
    %     csvwrite(sprintf("%s/locations_per_50ms_grid%d_sub%d.csv", subject_folder, grid_num(last_repeat), sub), locations_per_50ms_curr_grid);
    % end

    %csvwrite(sprintf("%s/locations_per_50ms_sub_%d.csv", subject_folder, sub), locations_per_50ms);
    % for g_i = 1:length(index_last_repeat)
    % 
    %     last_repeat=index_last_repeat(g_i);
    %     configs = [configs; configurations(last_repeat, :)];
    %     if g_i == 1
    %         timings_curr_grid = t_found_reward(1:last_repeat, :);
    %         grid_onset = [subj.trial_vars(1).grid_onset_timestamp(1)];
    %     else
    %         prev_last_repeat=index_last_repeat(g_i-1);
    %         timings_curr_grid = t_found_reward(prev_last_repeat+1:last_repeat, :);
    %         grid_onset = [subj.trial_vars(prev_last_repeat+1).grid_onset_timestamp(1)];
    %     end
    % 
    %     % Extract the last column and create the new first column
    %     start_looking_for_A = [grid_onset; timings_curr_grid(1:end-1, end)];
    %     % Prepend this new column to the existing matrix
    %     timings_curr_grid = [start_looking_for_A, timings_curr_grid];
    % 
    %     % OK SECOND TRY.
    %     % MAYBE I NEED TO ADJUST THE TIMINGS WHEN I CUT THE NEURONS AND
    %     % TRAJECTORIES.
    %     % this will be something like
    %     timings_curr_grid_in_bins = floor(timings_curr_grid/bin_size)+1;
    %     % next, adjust them such that it doesn't refer to original bins 
    %     % but to the cut ones
    % 
    %     csvwrite(sprintf("%s/timings_rewards_grid%d_sub%d.csv", subject_folder, grid_num(last_repeat), sub), timings_curr_grid);
    % end
    % 
    % current_location = all_positions(1);  % Initialize with the first location
    % for i = 1:length(all_positions_time)
    %     if i == 1
    %         start_idx = 1;  % Start from the first index
    %     else
    %         start_idx = floor(all_positions_time(i-1) / bin_size) + 1;
    %     end
    % 
    %     if i < length(all_positions_time)
    %         end_idx = floor(all_positions_time(i) / bin_size);
    %     else
    %         end_idx = num_windows;  % Go till the end of the time array
    %     end
    %     locations_per_50ms(start_idx:end_idx) = current_location;
    %     if i < length(all_positions)
    %         current_location = all_positions(i);  % Update location for the next iteration
    %     end
    % end