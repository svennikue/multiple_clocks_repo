% script to extract all cells, normalise them into 360 bins per repeat 
% and save them as csv
% save location timings in the equivalent 360-> [location x  time points] 
%%

clear all
do_plot = false;  % toggle plotting

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys'
    %abcd_data = load(sprintf("%s/beh_cells/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_08-Sep-2025.mat", source_dir));
    
else
    %abcd_data = load(sprintf("%s/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
    %abcd_data = load(sprintf("%s/abcd_data_10-Jul-2025.mat", source_dir));
    abcd_data = load(sprintf("%s/derivatives/abcd_passed.mat", source_dir));
end

deriv_dir = sprintf("%s/derivatives/", source_dir);

% subject_list = 1:length(abcd_data.abcd_data);
subject_list = 1:length(abcd_data.abcd_passed.abcd_data);
% subject_list = 1:3;

% PARAMETERS
n_states = 4;
bins_per_state = 90;
n_bins_total = bins_per_state * n_states;


% LOOP THROUGH SUBJECTS
for sub = 1:length(subject_list)
 %for sub = 59
    subj = abcd_data.abcd_passed.abcd_data(sub);
    %subj = abcd_data.abcd_data(sub);
    subject_folder = sprintf("%ss%02d/cells_and_beh", deriv_dir, sub);
    if ~exist(subject_folder, 'dir')
        mkdir(subject_folder);
        disp(['Folder created: ', subject_folder]);
    end

    if do_plot
        figure('Name', ['Subject ', num2str(sub)], 'Position', [100 100 1000 800]);
    end 

    % LOOP THROUGH NEURONS
    for cell_idx = 1:length(subj.neural_data)
        curr_cell_label = subj.neural_data(cell_idx).electrodeLabel;
        if isempty(subj.neural_data(cell_idx).roi) || isempty(subj.neural_data(cell_idx).roi{1})
            curr_roi_label = subj.neural_data(cell_idx).regionLabel;
        else
            curr_roi_label = subj.neural_data(cell_idx).roi{1};
        end
        combo_cell_label = sprintf('%02d-%s-%s', cell_idx, curr_cell_label, curr_roi_label);
        curr_spike_times = subj.neural_data(cell_idx).spikeTimes;
        all_reps_curr_cell = [];

        % locations and states only have to be done once
        if cell_idx == 1
            locations_all_reps = [];
            state_bounds_all_reps = [];
        end


        % loop through each repeat
        for t = 1:length(subj.trial_vars)
            
            % --- default rows as NaNs (so we can always append one row per trial) ---
            spike_rates_row   = nan(1, n_bins_total);
            location_bins_row = nan(1, n_bins_total);
            state_bounds_row  = nan(1, n_states+2); % trial_start + 3 state changes + trial_end = 5; here n_states+1 pairs with your build below

            % Trial timing
            trial_start = subj.trial_vars(t).grid_onset_timestamp(1);
            trial_end = subj.trial_vars(t).end_trial_timestamp;

            % State boundaries (A, B, C, D)
            state_times = subj.trial_vars(t).state_change_times;

            % Only compute edges if state_times exist and finite
            if ~isempty(state_times) && ~any(isnan(state_times))
                state_boundaries = [trial_start; state_times(:); trial_end];
                state_bounds_row = state_boundaries(:)';                       % store as 1 x (n_states+1)

                % Build bin edges and centers (same for spikes and location)
                bin_edges_total = [];
                for s = 1:n_states
                    t_start = state_boundaries(s);
                    t_end = state_boundaries(s+1);
                    bin_edges_state = linspace(t_start, t_end, bins_per_state + 1);
                    if s == 1
                        bin_edges_total = [bin_edges_total, bin_edges_state];
                    else
                        bin_edges_total = [bin_edges_total, bin_edges_state(2:end)];  % avoid duplicate edge
                    end
                end
            
                    
                % If bin edges look right, compute spike rates and locations
                if ~isempty(bin_edges_total) && numel(bin_edges_total) == (n_bins_total + 1)
                    % === Firing rate binning ===
                    % Empty spike list is fine: histcounts -> zeros.
                    spike_counts = histcounts(curr_spike_times, bin_edges_total);
                    bin_durations = diff(bin_edges_total);
                    spike_rates = spike_counts ./ bin_durations;
                    spike_rates_row = spike_rates;
                    
    
                    % === Location binning (aligned with same bin centers) ===
                    % only do this once for the first cell
                    if cell_idx == 1
                        locations = subj.trial_vars(t).start_location;
                        loc_timings = subj.trial_vars(t).grid_onset_timestamp;
                        if ~isempty(locations) && ~isempty(loc_timings) ...
                                && numel(locations) == numel(loc_timings)

                            bin_centers = (bin_edges_total(1:end-1) + bin_edges_total(2:end)) / 2;
                            location_bins = nan(1, n_bins_total);
                            for b = 1:n_bins_total
                                t_bin = bin_centers(b);
                                idx = find(loc_timings <= t_bin, 1, 'last');
                                location_bins(b) = locations(idx);
                            end
                            location_bins_row = location_bins;
                        end
                    end
                end
            else
                % keep spike_rates_row/location_bins_row/state_bounds_row as NaNs
                % (no keyboard, no continue)
                disp('careful! state_times missing -> inserting NaN row (not a problem, just to know for analysis later)');
            end
            % Append rows (always one per trial)

            all_reps_curr_cell   = [all_reps_curr_cell;   spike_rates_row];
            if cell_idx == 1
                locations_all_reps   = [locations_all_reps;   location_bins_row];
                state_bounds_all_reps = [state_bounds_all_reps; state_bounds_row];
            end

        end


        % === SAVE PER CELL ===
        if ~isempty(all_reps_curr_cell)
            % if cell_idx == 5
            %     keyboard
            % end
            csvwrite(fullfile(subject_folder, sprintf('cell-%02d-%s-360_bins_passed.csv', cell_idx, combo_cell_label)), all_reps_curr_cell);
            if cell_idx == 1
                csvwrite(fullfile(subject_folder, 'locations.csv'), locations_all_reps);
                csvwrite(fullfile(subject_folder, 'state_boundaries.csv'), state_bounds_all_reps);
            end
        end

        % === OPTIONAL: PLOT ===
        if do_plot && ~isempty(all_reps_curr_cell)
            mean_firing = mean(all_reps_curr_cell, 1);
            subplot_rows = ceil(sqrt(length(subj.neural_data)));
            subplot_cols = ceil(length(subj.neural_data) / subplot_rows);
            subplot(subplot_rows, subplot_cols, cell_idx);
            plot(mean_firing, 'k', 'LineWidth', 1.5)
            title(sprintf('Cell %d', cell_idx));
            xlabel('Bin (1–360)');
            ylabel('Firing rate (Hz)');
        end
    end
end

disp('DONE!');