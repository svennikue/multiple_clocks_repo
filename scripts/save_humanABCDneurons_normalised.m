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
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_24-Apr-2025.mat", source_dir));
    
else
    %abcd_data = load(sprintf("%s/abcd_data_FIXED_19-Feb-2025.mat", source_dir));
    abcd_data = load(sprintf("%s/abcd_data_10-Jul-2025.mat", source_dir));
end

deriv_dir = sprintf("%s/derivatives/", source_dir);

subject_list = 1:length(abcd_data.abcd_data);
% subject_list = 1:3;

% if something had gone wrong with the formatting, run this:
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


% PARAMETERS
n_states = 4;
bins_per_state = 90;
n_bins_total = bins_per_state * n_states;


% LOOP THROUGH SUBJECTS
for sub = 1:length(subject_list)
    subj = abcd_data.abcd_data(sub);
    subject_folder = sprintf("%ss%02d/cells_and_beh", deriv_dir, sub);
    if ~exist(subject_folder, 'dir')
        mkdir(subject_folder);
        disp(['Folder created: ', subject_folder]);
    end

    % OPTIONAL: figure setup
    if do_plot
        figure('Name', ['Subject ', num2str(sub)], 'Position', [100 100 1000 800]);
    end

    % LOOP THROUGH NEURONS
    for cell_idx = 1:length(subj.neural_data)
        curr_cell_label = subj.neural_data(cell_idx).electrodeLabel;
        curr_spike_times = subj.neural_data(cell_idx).spikeTimes;
        all_reps_curr_cell = [];

        % locations and states only have to be done once
        if cell_idx == 1
            locations_all_reps = [];
            state_bounds_all_reps = [];
        end


        % loop through each repeat
        for t = 1:length(subj.trial_vars)
            % Skip if no valid spike time
            if isempty(curr_spike_times)
                continue
            end

            % Trial timing
            trial_start = subj.trial_vars(t).grid_onset_timestamp(1);
            trial_end = subj.trial_vars(t).end_trial_timestamp;

            % State boundaries (A, B, C, D)
            state_times = subj.trial_vars(t).state_change_times;
            if any(isnan(state_times)) || isempty(state_times)
                continue
            end
            state_boundaries = [trial_start; state_times(:); trial_end];

            % Store this trial's boundaries
            state_bounds_all_reps = [state_bounds_all_reps; state_boundaries'];

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
            bin_centers = (bin_edges_total(1:end-1) + bin_edges_total(2:end)) / 2;

            % === Firing rate binning ===
            spike_counts = histcounts(curr_spike_times, bin_edges_total);
            bin_durations = diff(bin_edges_total);
            spike_rates = spike_counts ./ bin_durations;

            all_reps_curr_cell = [all_reps_curr_cell; spike_rates];

            % === Location binning (aligned with same bin centers) ===
            % only do this once for the first cell
            if cell_idx == 1
                locations = subj.trial_vars(t).start_location;
                loc_timings = subj.trial_vars(t).grid_onset_timestamp;
                if length(locations) ~= length(loc_timings)
                    continue
                end

                location_bins = zeros(1, n_bins_total);
                for b = 1:n_bins_total
                    t_bin = bin_centers(b);
                    idx = find(loc_timings <= t_bin, 1, 'last');
                    location_bins(b) = locations(idx);
                end
                locations_all_reps = [locations_all_reps; location_bins];
            end
        end

        % === SAVE PER CELL ===
        if ~isempty(all_reps_curr_cell)
            csvwrite(fullfile(subject_folder, sprintf('cell_%03d_%s_360_bins.csv', cell_idx, curr_cell_label)), all_reps_curr_cell);
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