function qc = qc_single_session(subj, varargin)
% QC_SINGLE_SESSION  Per-session QC with correlation-based deduplication.
% Computes, per cell: spikes, RPV, per-grid FR stability, and cell-vs-cell
% zero-lag correlations from binned spike counts over the task window.
%
% Accept/Reject:
%   - Base acceptance: spikes >= MinSpikes AND RPV_frac < RPV_Frac_Thresh
%   - If two (or more) accepted cells have corr >= CorrThresh:
%       keep the "better" one (more spikes, lower RPV, more stable FR),
%       reject others as "high-corr duplicate".
%
% Returns:
%   qc (struct array): fields listed below (see "Save" block).

% ----------------- Parameters -----------------
p = inputParser;
addParameter(p, 'MinSpikes', 300, @(x) isnumeric(x) && x>=0);
addParameter(p, 'RefracMS', 1.5, @(x) isnumeric(x) && x>0);
addParameter(p, 'RPV_Frac_Thresh', 0.01, @(x) isnumeric(x) && x>=0 && x<1);

% Correlation settings
addParameter(p, 'BinSizeS', 0.10, @(x) isnumeric(x) && x>0);     % 100 ms bins
addParameter(p, 'CorrWindow', 'task', @(s) any(strcmpi(s,{'task','overlap'})));
addParameter(p, 'CorrThresh', 0.50, @(x) isnumeric(x) && x>=0 && x<=1);

% Per-grid stability helper (for reporting / tie-breakers only)
addParameter(p, 'LowFRFrac', 0.20, @(x) isnumeric(x) && x>0 && x<1);
addParameter(p, 'MinTrialDur', 0.050, @(x) isnumeric(x) && x>=0);

% Session metadata for the header
addParameter(p, 'SessionID', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'SessionIndex', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
parse(p, varargin{:});

MinSpikes = p.Results.MinSpikes;
RefracS   = p.Results.RefracMS/1000;
bin       = p.Results.BinSizeS;
corrThr   = p.Results.CorrThresh;
min_dur   = p.Results.MinTrialDur;

% ----------------- Trials → task window -----------------
if ~isfield(subj,'trial_vars') || isempty(subj.trial_vars)
    error('No trial_vars in this session.');
end
tv = subj.trial_vars(:);
T  = numel(tv);
starts = nan(T,1); ends = nan(T,1); grids = nan(T,1);
for t = 1:T
    try s = tv(t).grid_onset_timestamp(1); catch, s = NaN; end
    try e = tv(t).end_trial_timestamp;     catch, e = NaN; end
    try g = tv(t).grid_num;                catch, g = NaN; end
    starts(t) = s; ends(t) = e; grids(t) = g;
end
durs = ends - starts;
valid_trials = isfinite(starts) & isfinite(ends) & isfinite(grids) & (durs > min_dur);
if ~any(valid_trials), error('No valid trials in this session.'); end

task_t0 = min(starts(valid_trials));
task_t1 = max(ends(valid_trials));
task_dur = task_t1 - task_t0;  if task_dur <= 0, error('Task duration not positive.'); end

% Optionally compute grid list for stability measures
ug = unique(grids(valid_trials));

% ----------------- Cells: spikes & labels -----------------
if ~isfield(subj,'neural_data') || isempty(subj.neural_data)
    error('No neural_data in this session.');
end
nC = numel(subj.neural_data);

spikeTimes_all = cell(nC,1);
elecLabels = cell(nC,1);
regionLabels = cell(nC,1);

for c = 1:nC
    nd = subj.neural_data(c);
    % labels
    if isfield(nd,'electrodeLabel') && ~isempty(nd.electrodeLabel)
        elecLabels{c} = char(string(nd.electrodeLabel));
    else
        elecLabels{c} = sprintf('Electrode_%d', c);
    end
    if isfield(nd,'regionLabel') && ~isempty(nd.regionLabel)
        regionLabels{c} = char(string(nd.regionLabel));
    else
        regionLabels{c} = 'Region_?';
    end
    % spikes
    if isfield(nd,'spikeTimes') && ~isempty(nd.spikeTimes)
        st = sort(nd.spikeTimes(:));
        st = st(isfinite(st));
    else
        st = [];
    end
    spikeTimes_all{c} = st;
end

% ----------------- Helpers -----------------
% Per-grid FR (for stability reporting / tie-breakers)
function [gridFR, gridDur] = per_grid_fr(spikes)
    gridFR  = nan(numel(ug),1);
    gridDur = zeros(numel(ug),1);
    for gi = 1:numel(ug)
        g = ug(gi);
        mask = valid_trials & (grids==g);
        if ~any(mask), continue; end
        idx = find(mask);
        tot_d = 0; tot_n = 0;
        for k = 1:numel(idx)
            tt = idx(k);
            t0 = starts(tt); t1 = ends(tt);
            tot_d = tot_d + (t1 - t0);
            if ~isempty(spikes)
                tot_n = tot_n + sum(spikes >= t0 & spikes < t1);
            end
        end
        gridDur(gi) = tot_d;
        if tot_d > 0
            gridFR(gi) = tot_n / tot_d;
        end
    end
end

% ----------------- First pass: per-cell metrics (no correlations yet) -----------------
qc = struct('electrodeLabel', elecLabels, ...
            'regionLabel',   regionLabels, ...
            'n_spikes', [], 'overall_FR_Hz', [], ...
            'RPV_frac', [], 'RPV_count', [], ...
            'grid_FR_mean', [], 'grid_FR_CV', [], ...
            'lowFR_grid_frac', [], ...
            'corr_max', [], 'corr_max_partner_idx', [], ...
            'base_accept', [], ...     % spikes & RPV threshold only
            'is_reliable', [], 'fail_reasons', []);

for c = 1:nC
    st = spikeTimes_all{c};
    nsp = numel(st);
    overall_FR = nsp / task_dur;

    % ISI / RPV
    if nsp >= 2
        isi = diff(st);
        RPV = sum(isi < RefracS);
    else
        RPV = 0;
    end
    RPV_frac = RPV / max(1,nsp);

    % Per-grid FR stability (reporting / tie-breakers)
    [gridFR, ~] = per_grid_fr(st);
    gridFR_mean = nanmean(gridFR);
    gridFR_CV   = nanstd(gridFR) / max(eps, gridFR_mean);
    lowFR_mask  = gridFR < (p.Results.LowFRFrac * gridFR_mean);
    lowFR_mask(isnan(gridFR)) = false;
    lowFR_grid_frac = mean(lowFR_mask);

    % Base acceptance (no correlation yet)
    base_ok = (nsp >= MinSpikes) && (RPV_frac < p.Results.RPV_Frac_Thresh);

    % Save now; we'll fill correlation fields later
    qc(c).n_spikes         = nsp;
    qc(c).overall_FR_Hz    = overall_FR;
    qc(c).RPV_frac         = RPV_frac;
    qc(c).RPV_count        = RPV;
    qc(c).grid_FR_mean     = gridFR_mean;
    qc(c).grid_FR_CV       = gridFR_CV;
    qc(c).lowFR_grid_frac  = lowFR_grid_frac;
    qc(c).corr_max         = NaN;
    qc(c).corr_max_partner_idx = NaN;
    qc(c).base_accept      = base_ok;
    qc(c).is_reliable      = base_ok;     % temporary; correlation dedup may demote
    qc(c).fail_reasons     = {};
    if ~base_ok
        if nsp < MinSpikes
            qc(c).fail_reasons{end+1} = sprintf('few spikes (%d < %d)', nsp, MinSpikes);
        end
        if RPV_frac >= p.Results.RPV_Frac_Thresh
            qc(c).fail_reasons{end+1} = sprintf('RPV %.2f%% ≥ %.2f%%', 100*RPV_frac, 100*p.Results.RPV_Frac_Thresh);
        end
    end
end

% ----------------- Correlation matrix over task/overlap window -----------------
% Choose correlation window
switch lower(p.Results.CorrWindow)
    case 'task'
        t0 = task_t0; t1 = task_t1;
    case 'overlap'
        % use overlap across "base accepted" cells; fall back to task if empty
        first_spk = inf(nC,1); last_spk = -inf(nC,1);
        for c = 1:nC
            s = spikeTimes_all{c};
            if ~isempty(s), first_spk(c)=s(1); last_spk(c)=s(end); end
        end
        mask = [qc.base_accept]' & isfinite(first_spk) & isfinite(last_spk);
        if any(mask)
            t0 = max([task_t0; first_spk(mask)]);
            t1 = min([task_t1;  last_spk(mask)]);
            if ~(t1 > t0), t0 = task_t0; t1 = task_t1; end
        else
            t0 = task_t0; t1 = task_t1;
        end
end
edges = t0:bin:t1; if edges(end) < t1, edges = [edges, t1]; end

% Build counts for ALL cells (so we can compute corr with "all others")
has_counts = false(nC,1);
counts_all = cell(nC,1);
for c = 1:nC
    s = spikeTimes_all{c};
    if isempty(s), continue
    end
    s = s(s>=t0 & s<t1);
    if isempty(s), continue
    end
    cnt = histcounts(s, edges);
    if var(cnt) <= 0, continue
    end
    counts_all{c} = cnt;
    has_counts(c) = true;
end

% Assemble matrix for cells with counts
idx_counts = find(has_counts);
if numel(idx_counts) >= 2
    M = cell2mat(cellfun(@(x) x(:).', counts_all(idx_counts), 'UniformOutput', false));
    R = corrcoef(M.');  % rows=cells-with-counts
else
    R = [];  % nothing to correlate
end

% Helper to fetch corr(i,j) by original indices
function r = get_corr(i,j)
    r = NaN;
    if isempty(R) || i==j, return; end
    ii = find(idx_counts==i, 1);
    jj = find(idx_counts==j, 1);
    if isempty(ii) || isempty(jj), return; end
    r = R(ii,jj);
end

% Fill max correlation per cell (against ANY other cell)
for c = 1:nC
    rmax = -Inf; rmax_j = NaN;
    if ~isempty(R)
        for j = 1:nC
            if j==c, continue; end
            rij = get_corr(c,j);
            if isfinite(rij) && rij > rmax
                rmax = rij; rmax_j = j;
            end
        end
    end
    if isfinite(rmax)
        qc(c).corr_max = rmax;
        qc(c).corr_max_partner_idx = rmax_j;
    else
        qc(c).corr_max = NaN;
        qc(c).corr_max_partner_idx = NaN;
    end
end

% ----------------- Correlation-based dedup among base-accepted cells -----------------
% Build graph on base-accepted cells: edge if corr >= CorrThresh
acc_idx = find([qc.base_accept]);
if numel(acc_idx) >= 2 && ~isempty(R)
    % adjacency among accepted cells
    A = false(numel(acc_idx));
    for a = 1:numel(acc_idx)-1
        i = acc_idx(a);
        for b = a+1:numel(acc_idx)
            j = acc_idx(b);
            rij = get_corr(i,j);
            if isfinite(rij) && (rij >= corrThr)
                A(a,b) = true; A(b,a) = true;
            end
        end
    end

    % connected components among accepted cells
    visited = false(numel(acc_idx),1);
    for a = 1:numel(acc_idx)
        if visited(a), continue; end
        % BFS
        queue = a; comp = a; visited(a)=true;
        while ~isempty(queue)
            u = queue(1); queue(1) = [];
            nbrs = find(A(u,:));
            for v = nbrs
                if ~visited(v)
                    visited(v) = true;
                    queue(end+1) = v; %#ok<AGROW>
                    comp(end+1)  = v; %#ok<AGROW>
                end
            end
        end
        if numel(comp) <= 1
            continue % no conflict
        end
        group_idx = acc_idx(comp); % original cell indices in this corr-clique

        % --- Choose winner by priority:
        % (1) more spikes, (2) lower RPV, (3) lower lowFR_grid_frac,
        % (4) lower grid_FR_CV, (5) higher overall_FR_Hz
        Spk  = arrayfun(@(k) qc(k).n_spikes,        group_idx);
        Rpv  = arrayfun(@(k) qc(k).RPV_frac,        group_idx);
        LowF = arrayfun(@(k) qc(k).lowFR_grid_frac, group_idx);
        CV   = arrayfun(@(k) qc(k).grid_FR_CV,      group_idx);
        FR   = arrayfun(@(k) qc(k).overall_FR_Hz,   group_idx);

        rankMat = [-Spk(:), Rpv(:), LowF(:), CV(:), -FR(:)];
        [~, ord] = sortrows(rankMat, [1 2 3 4 5], {'ascend','ascend','ascend','ascend','ascend'});
        winner_local = ord(1);
        winner = group_idx(winner_local);

        % Demote others
        losers = setdiff(group_idx, winner);
        for k = losers
            if qc(k).is_reliable
                qc(k).is_reliable = false;
                qc(k).fail_reasons{end+1} = sprintf('high-corr (>=%.2f) duplicate of %s|%s', ...
                    corrThr, qc(winner).electrodeLabel, qc(winner).regionLabel);
            else
                qc(k).fail_reasons{end+1} = sprintf('high-corr (>=%.2f) duplicate of %s|%s', ...
                    corrThr, qc(winner).electrodeLabel, qc(winner).regionLabel);
            end
        end
    end
end

% ----------------- Print summary -----------------
% Session meta
if isempty(p.Results.SessionID)
    if isfield(subj,'subject_ID'), sessID = char(string(subj.subject_ID));
    else, sessID = 'unknown_session'; end
else
    sessID = char(string(p.Results.SessionID));
end
idxStr = 'n/a'; if ~isempty(p.Results.SessionIndex), idxStr = num2str(p.Results.SessionIndex); end
task_dur_in_mins = task_dur/60;

fprintf('\nQC summary — Session %s (idx %s) — task %.1f mins [%.3f → %.3f s]\n', ...
    sessID, idxStr, task_dur_in_mins, task_t0, task_t1);
fprintf('Rules: spikes≥%d, RPV<%.2f%%, corr≥%.2f → keep better (spikes, RPV, stability)\n', ...
    MinSpikes, 100*p.Results.RPV_Frac_Thresh, corrThr);

n_pass = sum([qc.is_reliable]);
for c = 1:nC
    tag = 'PASS'; if ~qc(c).is_reliable, tag='FAIL'; end
    extra = '';
    if isfinite(qc(c).corr_max)
        partner = qc(c).corr_max_partner_idx;
        if ~isnan(partner)
            extra = sprintf(' | maxCorr=%.2f vs %s|%s', qc(c).corr_max, qc(partner).electrodeLabel, qc(partner).regionLabel);
        else
            extra = sprintf(' | maxCorr=%.2f', qc(c).corr_max);
        end
    end
    fprintf('%s | %s: %s | spikes=%5d | RPV=%.2f%% | lowFR=%.1f%% | CV=%.2f%s', ...
        qc(c).electrodeLabel, qc(c).regionLabel, tag, ...
        qc(c).n_spikes, 100*qc(c).RPV_frac, 100*qc(c).lowFR_grid_frac, qc(c).grid_FR_CV, extra);
    if ~qc(c).is_reliable && ~isempty(qc(c).fail_reasons)
        fprintf(' | reasons: %s', strjoin(qc(c).fail_reasons,'; '));
    end
    fprintf('\n');
end
fprintf('Total: %d cells | Passed: %d | Excluded: %d\n\n', nC, n_pass, nC - n_pass);
end
