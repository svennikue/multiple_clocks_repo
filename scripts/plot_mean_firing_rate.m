function [cell_FR_by_rep, mean_FR_raw, mean_FR_std, mean_dur_by_rep, mean_spkcount_by_rep, corr_stats] = ...
    plot_FR_correct_repeats_onefig(subj, nRepeats, norm_method)

% Inputs:
%   nRepeats    : number of correct repeats (default 10)
%   norm_method : 'minmax' (default) or 'zscore' for per-cell standardization
%
% Uses ONLY correct trials (trial_correct==1) and bins by trial_num_in_grid_correct (1..nRepeats).
% Counts spikes in [start, end) to avoid boundary double-counting.

if nargin < 2 || isempty(nRepeats), nRepeats = 10; end
if nargin < 3 || isempty(norm_method), norm_method = 'minmax'; end

% Colors/styles (FRs vs duration/count distinctly styled)
frColorRaw = [0 0.4470 0.7410];   % blue
frColorStd = [0.4940 0.1840 0.5560]; % purple
durColor   = [0.8500 0.3250 0.0980]; % orange
cntColor   = [0.4660 0.6740 0.1880]; % green

% Subject label
if isfield(subj,'subject_ID') && ~isempty(subj.subject_ID)
    sid = string(subj.subject_ID);
else
    sid = "UnknownSubject";
end

nCells  = numel(subj.neural_data);
nTrials = numel(subj.trial_vars);

% --------- Gather valid trials (correct & repeat in range) ---------
valid = false(1, nTrials);
rep_idx = nan(1, nTrials);
tStart  = nan(1, nTrials);
tEnd    = nan(1, nTrials);
dur     = nan(1, nTrials);

for t = 1:nTrials
    tv = subj.trial_vars(t);
    if ~isfield(tv,'trial_correct') || ~isfield(tv,'trial_num_in_grid_correct'), continue; end
    if tv.trial_correct ~= 1, continue; end
    r = tv.trial_num_in_grid_correct;
    if isempty(r) || isnan(r) || r < 1 || r > nRepeats, continue; end

    s = tv.grid_onset_timestamp(1);
    e = tv.end_trial_timestamp;
    d = e - s;
    if ~isfinite(d) || d <= 0, continue; end

    valid(t) = true;
    rep_idx(t) = r;
    tStart(t) = s;
    tEnd(t)   = e;
    dur(t)    = d;
end

valid_trials = find(valid);

% Mean duration per repeat (trial-level)
sum_dur   = zeros(1, nRepeats);
count_dur = zeros(1, nRepeats);
for tt = valid_trials
    r = rep_idx(tt);
    sum_dur(r)   = sum_dur(r)   + dur(tt);
    count_dur(r) = count_dur(r) + 1;
end
mean_dur_by_rep = sum_dur ./ max(count_dur, 1);
mean_dur_by_rep(count_dur == 0) = NaN;

% --------- Per-cell FR & spike counts by repeat ---------
sum_FR  = zeros(nCells, nRepeats);
cnt_FR  = zeros(nCells, nRepeats);
sum_cnt = zeros(nCells, nRepeats);
cnt_cnt = zeros(nCells, nRepeats);

% For correlation across all (cell, trial) pairs
all_dur = [];  % durations
all_fr  = [];  % firing rates (Hz)

for ci = 1:nCells
    spk = subj.neural_data(ci).spikeTimes;  % absolute timestamps
    for tt = valid_trials
        r = rep_idx(tt);
        s = tStart(tt);
        e = tEnd(tt);
        d = e - s;

        % Count spikes in [start, end)
        c = sum(spk >= s & spk < e);
        fr = c / d;

        sum_FR(ci, r)  = sum_FR(ci, r)  + fr;
        cnt_FR(ci, r)  = cnt_FR(ci, r)  + 1;

        sum_cnt(ci, r) = sum_cnt(ci, r) + c;
        cnt_cnt(ci, r) = cnt_cnt(ci, r) + 1;

        % store for correlation
        all_dur(end+1) = d; %#ok<AGROW>
        all_fr(end+1)  = fr; %#ok<AGROW>
    end
end

cell_FR_by_rep = sum_FR ./ max(cnt_FR, 1);
cell_FR_by_rep(cnt_FR == 0) = NaN;

cell_cnt_by_rep = sum_cnt ./ max(cnt_cnt, 1);
cell_cnt_by_rep(cnt_cnt == 0) = NaN;

% Across-cells means
mean_FR_raw = nanmean(cell_FR_by_rep, 1);

% Per-cell normalization across repeats
norm_mat = cell_FR_by_rep;
switch lower(norm_method)
    case 'minmax'
        for ci = 1:nCells
            x = norm_mat(ci,:);
            a = nanmin(x); b = nanmax(x);
            if isfinite(a) && isfinite(b) && b > a
                norm_mat(ci,:) = (x - a) ./ (b - a);
            else
                norm_mat(ci,:) = NaN;
            end
        end
        ylab_std = 'Min–max norm (0–1)';
    case 'zscore'
        for ci = 1:nCells
            x = norm_mat(ci,:);
            mu = nanmean(x); sg = nanstd(x);
            if isfinite(mu) && sg > 0
                norm_mat(ci,:) = (x - mu) ./ sg;
            else
                norm_mat(ci,:) = NaN;
            end
        end
        ylab_std = 'Z-scored FR';
    otherwise
        error('norm_method must be ''minmax'' or ''zscore''.');
end
mean_FR_std = nanmean(norm_mat, 1);

% Mean spike count across cells by repeat
mean_spkcount_by_rep = nanmean(cell_cnt_by_rep, 1);

% Correlation: duration vs firing rate across all (cell, trial) pairs
valid_corr = isfinite(all_dur) & isfinite(all_fr);
if any(valid_corr)
    [rho, p] = corr(all_dur(valid_corr).', all_fr(valid_corr).', 'rows','complete');
    Ncorr = sum(valid_corr);
else
    rho = NaN; p = NaN; Ncorr = 0;
end
corr_stats = struct('r', rho, 'p', p, 'N', Ncorr);

% -------------------- Single Figure with All Subplots --------------------
x = 1:nRepeats;
fig = figure('Name','FR by repeat (correct only)');
tl = tiledlayout(fig, 3, 2, 'Padding','compact','TileSpacing','compact');
title(tl, sprintf('Subject %s — Correct trials only', sid));

% 1) Per-cell mean FR
nexttile; hold on;
for ci = 1:nCells
    plot(x, cell_FR_by_rep(ci,:), '-', 'Marker','o', 'LineWidth', 1.0);
end
xlim([1 nRepeats]); xticks(1:nRepeats);
xlabel('Repeat (1..10)'); ylabel('Hz'); box on;
title('Per-cell mean firing rate');

% 2) Across-cells RAW mean FR (distinct color)
nexttile; hold on;
plot(x, mean_FR_raw, '-o', 'LineWidth', 2.0, 'Color', frColorRaw, 'MarkerFaceColor', frColorRaw);
xlim([1 nRepeats]); xticks(1:nRepeats);
xlabel('Repeat'); ylabel('Hz'); box on;
title('Across-cells mean FR (RAW)');

% 3) Across-cells STANDARDIZED mean FR (distinct color/marker)
nexttile; hold on;
plot(x, mean_FR_std, '-^', 'LineWidth', 2.0, 'Color', frColorStd, 'MarkerFaceColor', frColorStd);
xlim([1 nRepeats]); xticks(1:nRepeats);
xlabel('Repeat'); ylabel(ylab_std); box on;
title(sprintf('Across-cells mean FR (per-cell %s)', lower(norm_method)));

% 4) Mean trial duration (separate subplot, dashed style)
nexttile; hold on;
plot(x, mean_dur_by_rep, '--d', 'LineWidth', 2.0, 'Color', durColor, 'MarkerFaceColor', durColor);
xlim([1 nRepeats]); xticks(1:nRepeats);
xlabel('Repeat'); ylabel('Mean duration (s)'); box on;
title('Mean trial duration');

% 5) Mean spike count (separate subplot, dotted style)
nexttile; hold on;
plot(x, mean_spkcount_by_rep, ':s', 'LineWidth', 2.2, 'Color', cntColor, 'MarkerFaceColor', cntColor);
xlim([1 nRepeats]); xticks(1:nRepeats);
xlabel('Repeat'); ylabel('Mean spike count'); box on;
title('Mean spike count (per cell)');

% 6) Notes / correlation tile — use real newlines via sprintf
nexttile; axis off;
msg = sprintf(['Correlation between duration and firing rate\n' ...
               '(across all cell–trial pairs)\n' ...
               'r = %.3f,  p = %.3g,  N = %d'], rho, p, Ncorr);
text(0.02, 0.95, msg, 'Units','normalized', ...
     'HorizontalAlignment','left', 'VerticalAlignment','top', ...
     'FontWeight','bold', 'FontSize', 10);

end
