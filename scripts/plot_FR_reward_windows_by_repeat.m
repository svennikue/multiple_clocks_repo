function [cell_mean_by_rep, across_mean_by_rep, across_reward_mean_by_rep, cell_reward_by_rep] = ...
    plot_FR_reward_windows_by_repeat(subj, nRepeats, windowSec)
% plot_FR_reward_windows_by_repeat(subj, nRepeats, windowSec)
% Counts spikes in EXACT ±windowSec-second windows around each of the 4
% state_change_times per correct trial. No merging; double counting OK.
%
% Subplots (one figure):
%   1) Per-neuron curves: mean FR across the 4 windows vs repeat (1..nRepeats)
%   2) Across-neuron mean of (1)
%   3) 4 separate curves (A,B,C,D): per-window FR vs repeat, averaged across neurons
%
% Inputs:
%   subj.neural_data(ci).spikeTimes              : absolute spike times (s)
%   subj.trial_vars(t).trial_correct            : 1 if correct
%   subj.trial_vars(t).trial_num_in_grid_correct: repeat index (1..nRepeats)
%   subj.trial_vars(t).state_change_times       : [1x4] absolute times (s)
%   nRepeats   : default 10
%   windowSec  : default 2  (so each window is 4 s total)
%
% Outputs:
%   cell_mean_by_rep           : [nCells x nRepeats], per-cell mean (avg over A..D & trials)
%   across_mean_by_rep         : [1 x nRepeats], across-neuron mean of cell_mean_by_rep
%   across_reward_mean_by_rep  : [4 x nRepeats], A..D per repeat, averaged across neurons
%   cell_reward_by_rep         : [nCells x nRepeats x 4], per-cell, per-repeat, per-window means

if nargin < 2 || isempty(nRepeats), nRepeats = 10; end
if nargin < 3 || isempty(windowSec), windowSec = 2; end
L = 2*windowSec;  % each window length in seconds (e.g., 4 s)

% Subject label
if isfield(subj,'subject_ID') && ~isempty(subj.subject_ID)
    sid = string(subj.subject_ID);
else
    sid = "UnknownSubject";
end

nCells  = numel(subj.neural_data);
nTrials = numel(subj.trial_vars);

% ---- collect valid trials: correct & repeat in range & 4 state times ----
valid_trials = false(1, nTrials);
rep_idx      = nan(1, nTrials);
state_mat    = nan(nTrials, 4);

for t = 1:nTrials
    tv = subj.trial_vars(t);
    if ~isfield(tv,'trial_correct') || tv.trial_correct ~= 1, continue; end
    if ~isfield(tv,'trial_num_in_grid_correct'), continue; end
    r = tv.trial_num_in_grid_correct;
    if isempty(r) || isnan(r) || r < 1 || r > nRepeats, continue; end
    if ~isfield(tv,'state_change_times') || numel(tv.state_change_times) ~= 4, continue; end
    if any(~isfinite(tv.state_change_times)), continue; end

    valid_trials(t) = true;
    rep_idx(t)      = r;
    state_mat(t,:)  = tv.state_change_times(:).';
end
VT = find(valid_trials);

% ---- accumulators ----
% per-cell, per-repeat, per-window (A..D) sum of FRs and counts
sum_fr_reward = zeros(nCells, nRepeats, 4);
cnt_fr_reward = zeros(nCells, nRepeats, 4);

% per-cell, per-repeat sum of the mean FR across 4 windows + counts
sum_fr_mean = zeros(nCells, nRepeats);
cnt_fr_mean = zeros(nCells, nRepeats);

% ---- compute rates ----
for ci = 1:nCells
    spk = subj.neural_data(ci).spikeTimes;  % absolute timestamps
    if isempty(spk), spk = []; end

    for tt = VT
        r = rep_idx(tt);
        times = state_mat(tt,:);  % [1x4], A..D (keep given order)

        fr4 = nan(1,4);
        for k = 1:4
            a = times(k) - windowSec;   % start of window
            b = times(k) + windowSec;   % end of window (right-open)
            if ~isfinite(a) || ~isfinite(b) || b <= a, continue; end

            % EXACT ±windowSec around state change; no clipping, double count allowed
            c = sum(spk >= a & spk < b);
            fr4(k) = c / L;  % Hz in that 4-s window
            sum_fr_reward(ci, r, k) = sum_fr_reward(ci, r, k) + fr4(k);
            cnt_fr_reward(ci, r, k) = cnt_fr_reward(ci, r, k) + 1;
        end

        % mean across the four windows (equivalent to total counts / 16 s)
        mfr = mean(fr4, 'omitnan');
        if isfinite(mfr)
            sum_fr_mean(ci, r) = sum_fr_mean(ci, r) + mfr;
            cnt_fr_mean(ci, r) = cnt_fr_mean(ci, r) + 1;
        end
    end
end

% Per-cell means
cell_reward_by_rep = sum_fr_reward ./ max(cnt_fr_reward, 1);      % [nCells x nRepeats x 4]
cell_reward_by_rep(cnt_fr_reward == 0) = NaN;

cell_mean_by_rep   = sum_fr_mean   ./ max(cnt_fr_mean, 1);        % [nCells x nRepeats]
cell_mean_by_rep(cnt_fr_mean == 0) = NaN;

% Across-neuron means
across_mean_by_rep = nanmean(cell_mean_by_rep, 1);                % [1 x nRepeats]
tmp = squeeze(nanmean(cell_reward_by_rep, 1));                    % [nRepeats x 4]
across_reward_mean_by_rep = permute(tmp, [2 1]);                  % [4 x nRepeats]

% ---- plotting ----
x = 1:nRepeats;

% some colors for A,B,C,D
cA = [0.0000 0.4470 0.7410]; % blue
cB = [0.8500 0.3250 0.0980]; % orange
cC = [0.4660 0.6740 0.1880]; % green
cD = [0.4940 0.1840 0.5560]; % purple

fig = figure('Name','Reward-centered FR by repeat (correct only)');
tl = tiledlayout(fig, 3, 1, 'Padding','compact','TileSpacing','compact');
title(tl, sprintf('Subject %s — FR in \\pm%.1fs around A–D (correct trials)', sid, windowSec));

% (1) Per-neuron: avg across A–D vs repeat
nexttile; hold on;
for ci = 1:nCells
    plot(x, cell_mean_by_rep(ci,:), '-o', 'LineWidth', 1.0);
end
xlim([1 nRepeats]); xticks(1:nRepeats);
ylabel('FR (Hz)'); xlabel('Repeat (1..10)'); box on;
title('Per-neuron mean FR (avg of A–D)');

% (2) Across-neuron mean: avg across A–D vs repeat
nexttile; hold on;
plot(x, across_mean_by_rep, '-s', 'LineWidth', 2.0, 'MarkerFaceColor', [0.3010 0.7450 0.9330], ...
     'Color', [0.3010 0.7450 0.9330]);
xlim([1 nRepeats]); xticks(1:nRepeats);
ylabel('FR (Hz)'); xlabel('Repeat (1..10)'); box on;
title('Across-neuron mean FR (avg of A–D)');

% (3) A–D separately, averaged across neurons
nexttile; hold on;
plot(x, across_reward_mean_by_rep(1,:), '-o', 'LineWidth', 2.0, 'Color', cA, 'MarkerFaceColor', cA);
plot(x, across_reward_mean_by_rep(2,:), '-s', 'LineWidth', 2.0, 'Color', cB, 'MarkerFaceColor', cB);
plot(x, across_reward_mean_by_rep(3,:), '-^', 'LineWidth', 2.0, 'Color', cC, 'MarkerFaceColor', cC);
plot(x, across_reward_mean_by_rep(4,:), '-d', 'LineWidth', 2.0, 'Color', cD, 'MarkerFaceColor', cD);
xlim([1 nRepeats]); xticks(1:nRepeats);
ylabel('FR (Hz)'); xlabel('Repeat (1..10)'); box on;
legend({'A','B','C','D'}, 'Location','best');
title('Across-neuron FR per window (A–D)');

end
