% step-aligned binning into 400 bins with reward alignment

clear all
do_plot = true;

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans";
if ~exist(source_dir, 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys';
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_08-Sep-2025.mat", source_dir));
else
    abcd_data = load(sprintf("%s/derivatives/abcd_passed.mat", source_dir));
end

deriv_dir = sprintf("%s/derivatives/", source_dir);
n_bins_total = 400;

for sub = 59
    subj = abcd_data.abcd_passed.abcd_data(sub);

    subject_folder = sprintf("%ss%02d/cells_and_beh", deriv_dir, sub);
    if ~exist(subject_folder, 'dir')
        mkdir(subject_folder);
    end

    % preallocate 
    n_trials = length(subj.trial_vars);
    locations_all = nan(n_trials, n_bins_total);
    buttons_all   = nan(n_trials, n_bins_total);

    for cell_idx = 1:length(subj.neural_data)

        curr_spike_times = subj.neural_data(cell_idx).spikeTimes;
        all_reps = nan(n_trials, n_bins_total);

        for t = 1:n_trials

            trial = subj.trial_vars(t);

            locs = trial.start_location;
            loc_times = trial.grid_onset_timestamp;
            buttons = trial.button_pressed;
            reward_times = trial.state_change_times;
            reward_locs = trial.sequence_locations;

            % === REJECTION of incomplete runs ===
            if isempty(locs) || isempty(loc_times) || sum(~isnan(reward_times)) < 4
                disp('skipping in line 46 for trial n = ')
                disp(t)
                continue
            end

            % === STEP DETECTION ===
            % defined as 'location onset to new location'
            step_idx = [1; find(diff(locs') ~= 0) + 1];

            step_times = loc_times(step_idx);
            step_locs = locs(step_idx);

            % add end
            step_times = [step_times, trial.end_trial_timestamp];
            step_locs = [step_locs, locs(end)];

            % === REWARD STEP INDEX ===
            reward_step_idx = nan(1,4);
            valid = true;

            for r = 1:4
                idx_loc = find(loc_times <= reward_times(r), 1, 'last');
                if isempty(idx_loc)
                    valid = false; break
                end

                curr_loc = locs(idx_loc);

                % ok i don't get this. I dont think this is right.
                % i think the initial idx should be the correct one!
                % double check thi.s
                idx_step = find(step_locs == curr_loc, 1);

                if isempty(idx_step)
                    valid = false; break
                end

                reward_step_idx(r) = idx_step;
            end

            if ~valid
                disp('skipping in line 81')
                continue
            end

            % === BUILD BIN EDGES ===
            all_edges = [];

            for seg = 1:4
                if seg == 1
                    s0 = 1;
                else
                    s0 = reward_step_idx(seg-1);
                end
                s1 = reward_step_idx(seg);

                steps = s0:s1;
                n_steps = length(steps);

                bins = floor(100 / n_steps);
                rem  = mod(100, n_steps);

                seg_edges = [];

                for i = 1:n_steps
                    si = steps(i);

                    n_b = bins + (i <= rem);
                    edges = linspace(step_times(si), step_times(si+1), n_b+1);

                    if isempty(seg_edges)
                        seg_edges = edges;
                    else
                        seg_edges = [seg_edges edges(2:end)];
                    end
                end

                if isempty(all_edges)
                    all_edges = seg_edges;
                else
                    all_edges = [all_edges seg_edges(2:end)];
                end
            end

            if numel(all_edges) ~= n_bins_total+1
                continue
            end

            % === SPIKES ===
            counts = histcounts(curr_spike_times, all_edges);
            rates = counts ./ diff(all_edges);

            all_reps(t,:) = rates;

            % === LOCATION + BUTTON (once) ===
            if cell_idx == 1
                centers = (all_edges(1:end-1) + all_edges(2:end)) / 2;

                for b = 1:n_bins_total
                    idx = find(loc_times <= centers(b), 1, 'last');
                    if ~isempty(idx)
                        locations_all(t,b) = locs(idx);
                        buttons_all(t,b) = buttons(idx);
                    end
                end
            end
        end

        % === PLOT ===
        if do_plot
            n_cells = length(subj.neural_data);
            n_rows = ceil(sqrt(n_cells));
            n_cols = ceil(n_cells / n_rows);
        
            subplot(n_rows, n_cols, cell_idx);
            plot(mean(all_reps,1,'omitnan'),'k')
            title(sprintf('Cell %d', cell_idx));
        end
    end

    % % === SAVE ===
    % % === SAVE ===
    % if ~isempty(all_reps_curr_cell)
    %     csvwrite(fullfile(subject_folder, sprintf('cell-%02d-%s-400_bins_steps.csv', cell_idx, combo_cell_label)), all_reps_curr_cell);
    % 
    %     if cell_idx == 1
    %         csvwrite(fullfile(subject_folder, 'locations_steps_400.csv'), locations_all_reps);
    %     end
    % end
    % 
    % csvwrite(fullfile(subject_folder,'locations_steps_400.csv'), locations_all);
    % csvwrite(fullfile(subject_folder,'buttons_steps_400.csv'), buttons_all);

end

disp('DONE!');