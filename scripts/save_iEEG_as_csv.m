% extract which subjects are the ones that have the same grids.
% then reshape the cells such that they are formatted as 
% firing rate per 50ms.
% then also recode the locations as location per 50ms bins.

clear all
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys'
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
else
    abcd_data = load(sprintf("%s/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
end

deriv_dir = sprintf("%s/derivatives/", source_dir);

subject_list = 1:length(abcd_data.abcd_data);

% script to extract all cells and save them as csv
% Restructure them as well to analyse them analoguesly to the mouse data:
% save reward locations per task [n tasks x 4] and cell label per subject

% save cells in bins of 50ms -> encode firing rate per 50ms [n cells x time points]
% save location in bins of 50ms -> [location x  time points]
% save timings of state start + reward start for ABCD = [n repeats x 8
% timings]
% save all three of them in separate csv-s; split by task 
%%
% if something had gone wrong with the formatting, run this:
% Assume abcd_data.abcd_data is your structured array


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


bin_size = 0.025; %play around with this! Mohamady has 0.025


for sub = 1:length(subject_list)
    % if sub == 6 || sub == 9 || sub == 27 || sub == 44
    if sub == 9
        % 6 because it has every other grid_onset_timestamp transposed and that
        % annoying
        % 9 because it the timings start with negative values, so there
        % must have been something wrong
        % because 27 has the subj.trial_vars.state_change_times encoded in
        % a transposed way (?) and that creates problems
        continue;
    end
    subj = abcd_data.abcd_data(sub);
    all_cells = [];
    region_labels_cells = {};
    subject_folder = sprintf("%ss%02d/cells_and_beh", deriv_dir, sub);
    if ~exist(subject_folder, 'dir')
        mkdir(subject_folder); % Create the folder if it does not exist
        disp(['Folder created: ', subject_folder]);
    end
    % if sub == 27
    %     t_found_reward = subj.trial_vars.state_change_times;
    % else
    %     t_found_reward = [subj.trial_vars.state_change_times]';
    % end
    t_found_reward = [subj.trial_vars.state_change_times]';
    configurations = [subj.trial_vars.sequence_locations]';
    grid_num = [subj.trial_vars.grid_num]';
    configs = [];

    % then extract all cells per subject and code them
    % as firing rate in a 50 ms timewindow
    % for the neural data, I am creating bins in one go from
    % 0 seconds to end_time
    onset_time_behaviour = subj.trial_vars(1).rule_onset_timestamp;
    if sub > 51
        end_time_behaviour = subj.trial_vars(end).DONOTUSE_button_pressed_timestamp(end);
    else
        end_time_behaviour = subj.trial_vars(end).button_pressed_timestamp(end);
    end

    for c = 1:length(subj.neural_data)
        curr_spike_times = subj.neural_data(c).spikeTimes;
        end_time_neural_recs = curr_spike_times(end);
        % creating a file which runs from start behaviour to end behaviour
        % as these are the times that I will actually analyse.
        edges = 0:bin_size:end_time_behaviour;
        % counting how many spikes are within 
        [firing_rate_curr_cell, edges] = histcounts(curr_spike_times, edges);
        region_labels_cells{c} = subj.neural_data(c).regionLabel;
        all_cells = [all_cells; firing_rate_curr_cell];
    end

    % per subject, also extract all grid configs and all cell labels
    filename = (sprintf("%s/all_cells_region_labels_sub%02d.txt", subject_folder, sub));
    fid = fopen(filename, 'wt');  % Open a text file for writing
    % Write each label on a new line
    for i = 1:length(region_labels_cells)
        fprintf(fid, '%s\n', region_labels_cells{i});
    end
    fclose(fid);  % Close the file

    % concatenate the positions and position changes for each grid and repeat
    for c = 1:length(subj.trial_vars)
        if sub > 51
            curr_button_t = subj.trial_vars(c).DONOTUSE_button_pressed_timestamp;
        else
            curr_button_t = subj.trial_vars(c).button_pressed_timestamp;
        end
        curr_button = subj.trial_vars(c).button_pressed;

        curr_position_time = subj.trial_vars(c).visit_begin_timestamp_c;
        curr_position = subj.trial_vars(c).visited_locations_c;

        if c == 1
            all_button_t = curr_button_t;
            all_buttons = curr_button;

            all_positions_time = curr_position_time;
            all_positions = curr_position;
        else
            all_button_t = [all_button_t, curr_button_t];
            all_buttons = [all_buttons, curr_button];

            all_positions_time = [all_positions_time, curr_position_time];
            all_positions = [all_positions, curr_position];
        end
    end
    % in the end of this check if they are the same length!
    if length(all_positions_time) ~= length(all_positions)
        display("careful! there is a mistake in the match of positions and timings!!")
        break
    end

    % store location data in the binned format.
    % to do that, first identify from which to which bin they stayed in the
    % same location.
    num_bins = length(all_cells);
    locations_per_50ms = zeros(1, num_bins);
    for new_loc_t_idx = 1:length(all_positions_time)
        curr_t_loc_change = all_positions_time(new_loc_t_idx);
        if new_loc_t_idx == 1
            first_bin_to_fill = 1;
        else
            % this rounds down: makes them be at a location slightly
            % earlier.
            first_bin_to_fill = floor(all_positions_time(new_loc_t_idx)/bin_size);
        end
        if new_loc_t_idx < length(all_positions_time)
            last_bin_to_fill = floor(all_positions_time(new_loc_t_idx+1)/bin_size);
        else
            last_bin_to_fill = length(locations_per_50ms);
        end
        curr_location = all_positions(new_loc_t_idx);
        locations_per_50ms(first_bin_to_fill:last_bin_to_fill) = curr_location;
    end

    timings_rewards_bins = ceil(t_found_reward/bin_size);



    % do the same for buttons
    % to do that, first identify from which to which bin they stayed in the
    % same location.
    buttons_per_ms_bin = cell(1,num_bins);
    for press_button = 1:length(all_button_t)
        if press_button == 1
            first_bin_to_fill = 1;
        else 
            first_bin_to_fill = floor(all_button_t(press_button)/bin_size);
        end
        if press_button < length(all_button_t)
            last_bin_to_fill = floor(all_button_t(press_button+1)/bin_size);
        else
            last_bin_to_fill = length(buttons_per_ms_bin);
        end
        current_but = all_buttons(press_button);
        buttons_per_ms_bin(first_bin_to_fill:last_bin_to_fill) = current_but;
    end


    % disp(sprintf("length cell recordings is %d and location bins is %d ", length(all_cells), length(locations_per_50ms)))

    % finally, extract the relevant timings.
    index_last_repeat = find(diff(grid_num) ~= 0); 
    index_last_repeat = [index_last_repeat; length(grid_num)];
    for g_i = 1:length(index_last_repeat) % loop through every grid
        % first do the timings
        last_repeat=index_last_repeat(g_i);
        % also track the configurations to store later
        configs = [configs; configurations(last_repeat, :)];
        % then identify which bins correspond to timings of current repeat
        if g_i == 1
            timings_curr_grid = t_found_reward(1:last_repeat, :);
            % if first repeat, then this needs to be the start of the task
            start_idx = [subj.trial_vars(1).grid_onset_timestamp(1)];
            start_looking_for_A = arrayfun(@(x) x.grid_onset_timestamp(1), subj.trial_vars(1:last_repeat));
            % put end_trial_timestamp
            %start_looking_for_A = arrayfun(@(x) x.grid_onset_timestamp(1), subj.trial_vars(1:last_repeat));
        else
            prev_last_repeat = index_last_repeat(g_i-1);
            %t_prev_end_trial = subj.trial_vars(prev_last_repeat).end_trial_timestamp;
            %start_idx = floor(t_prev_end_trial/ bin_size) + 1;
            %start_idx = [subj.trial_vars(prev_last_repeat+1).grid_onset_timestamp(1)];
            timings_curr_grid = t_found_reward(prev_last_repeat+1:last_repeat, :);
            start_looking_for_A = arrayfun(@(x) x.end_trial_timestamp, subj.trial_vars(prev_last_repeat:last_repeat-1));
            start_idx = t_found_reward(prev_last_repeat, end);
            test_idx = 0;
            while isnan(start_idx)
                start_idx = subj.trial_vars(prev_last_repeat).grid_onset_timestamp(end-test_idx);
                test_idx = test_idx +1;
            end
        end

        % Extract the last column and create the new first column
        % Add this new column to the existing matrix
        timings_curr_grid = [start_looking_for_A', timings_curr_grid];
        
        start_idx_bin = floor(start_idx/bin_size);
        t_end_trial = subj.trial_vars(last_repeat).end_trial_timestamp;
        if isnan(t_end_trial)
           if sub > 51
                t_end_trial = subj.trial_vars(last_repeat).DONOTUSE_button_pressed_timestamp(end);
            else
                t_end_trial = subj.trial_vars(last_repeat).button_pressed_timestamp(end);
           end
        end

        % note that because t_end_trial is always a bit later than finding
        % the last grid, location and cell arrays a slightly longer than
        % timings!
        end_idx_bin = ceil(t_end_trial/ bin_size);
        if end_idx_bin > length(all_cells)
            end_idx_bin = length(all_cells);
        end
       
        % finally, use the extracted timings to split the cells, locations and timings by repeat.
        all_cells_curr_grid = all_cells(:, start_idx_bin:end_idx_bin);
        csvwrite(sprintf("%s/all_cells_firing_rate_grid%d_sub%02d.csv", subject_folder, grid_num(last_repeat), sub), all_cells_curr_grid);
        locations_per_50ms_curr_grid = locations_per_50ms(start_idx_bin:end_idx_bin);
        csvwrite(sprintf("%s/locations_per_25ms_grid%d_sub%02d.csv", subject_folder, grid_num(last_repeat), sub), locations_per_50ms_curr_grid);
        all_buttons_curr_grid = buttons_per_ms_bin(start_idx_bin:end_idx_bin);
        
        writecell(all_buttons_curr_grid,sprintf("%s/buttons_per_25ms_grid%d_sub%02d.csv", subject_folder, grid_num(last_repeat), sub));

        % change timings to bins that always start with the first bin that
        % I cut the timings and locations to 
        timings_curr_grid_in_bins = ceil(timings_curr_grid/bin_size)-start_idx_bin; 

        % do a little test-loop!
        if timings_curr_grid_in_bins(1,1) < 0
            display('something off with first timing.')
        end

        for i = 1:(size(timings_curr_grid_in_bins,1))
            for j = 1:(size(timings_curr_grid_in_bins,2)-1)
                add_idx = 0;
                % Find index for the time to mark
                curr_rew = configurations(last_repeat, j);
                idx_location_where_reward_should_be = timings_curr_grid_in_bins(i, j+1);
                if ~isnan(idx_location_where_reward_should_be)
                    location_where_reward_shoud_be = locations_per_50ms_curr_grid(idx_location_where_reward_should_be);
                    if curr_rew ~= location_where_reward_shoud_be
                        while curr_rew ~= location_where_reward_shoud_be
                            add_idx = add_idx + 1;
                            location_where_reward_shoud_be = locations_per_50ms_curr_grid(idx_location_where_reward_should_be+ add_idx);
                        end
                        timings_curr_grid_in_bins(i, j+1) = idx_location_where_reward_should_be+ add_idx;
                        
                        display("Careful!!! reward is not where it's supposed to be!")
                        display("grid, then repeat, then reward, then how much added to index")
                        g_i
                        i
                        j
                        add_idx
                    end
                end
            end
        end


            

        % % Constants
        % num_columns = size(timings_curr_grid, 2);
        % 
        % % Create the figure
        % figure;
        % hold on;
        % 
        % % Plot each segment
        % for i = 1:num_columns
        %     % Find indices for start and end times
        %     start_time = timings_curr_grid(1, i);
        %     end_time = timings_curr_grid(end, i);
        % 
        %     start_idx = find(all_positions_time >= start_time, 1, 'first');
        %     end_idx = find(all_positions_time <= end_time, 1, 'last');
        % 
        %     % Extract and plot the segment
        %     if ~isempty(start_idx) && ~isempty(end_idx)
        %         segment_times = all_positions_time(start_idx:end_idx);
        %         segment_positions = all_positions(start_idx:end_idx);
        %         plot(segment_times, segment_positions, '-o'); % Change '-o' to another marker if preferred
        %     end
        % end
        % 
        % % Mark specific times
        % for i = 1:(size(timings_curr_grid,1))
        %     for j = 1:(size(timings_curr_grid,2)-1)
        %         % Find index for the time to mark
        %         mark_time = timings_curr_grid(i,j+1);
        %         all_locations_before_reward = find(all_positions_time <= mark_time);
        % 
        %         if all_locations_before_reward
        %             mark_idx = all_locations_before_reward(end);
        %             plot(all_positions_time(mark_idx), configurations(last_repeat, j), 'ko', 'MarkerFaceColor', 'k'); % Black filled dot
        %             plot(all_positions_time(mark_idx), all_positions(mark_idx), 'rx', 'MarkerSize', 20); % Red 'x' marker
        %         end
        %     end
        % end
        % 
        % % Label the plot
        % xlabel('Time');
        % ylabel('Position');
        % 
        % title(sprintf('Position Plot with rewards for grid%d and sub%02d', grid_num(last_repeat), sub));
        % legend('Positions', 'Reward Times');
        % hold off;



        %disp(sprintf("cell length in repeat %d is %d , loc length is %d and last timing is %d", g_i, size(all_cells_curr_grid,2), size(locations_per_50ms_curr_grid,2), timings_curr_grid_in_bins(end)))
        
        % save the timings in the binned format
        csvwrite(sprintf("%s/timings_rewards_grid%d_sub%02d.csv", subject_folder, grid_num(last_repeat), sub), timings_curr_grid_in_bins);
    end
    % for each subject, save the configurations
    disp(sprintf("%s/all_configs_sub%02d.csv", subject_folder, sub))
    csvwrite(sprintf("%s/all_configs_sub%02d.csv", subject_folder, sub), configs);
end




