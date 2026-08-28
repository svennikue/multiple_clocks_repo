% RUN_CELL_QC  Reproduce the single-unit quality control and abcd_passed.mat.
%
% This is the script of record for which units enter the analyses. It reads the
% raw sorted dataset, applies three acceptance criteria, and writes both the
% full per-cell QC record and the filtered dataset that every downstream
% analysis consumes.
%
%   inputs   <source>/abcd_data_08-Sep-2025.mat
%   outputs  <source>/derivatives/qc_all_sessions<SUF>.mat     per-cell metrics + decisions
%            <source>/derivatives/accepted_cells_index<SUF>.mat flat index of accepted cells
%            <source>/derivatives/abcd_passed<SUF>.mat          dataset filtered to accepted cells
%            <source>/derivatives/qc_pass_mask<SUF>.csv         one row per cell, for cross-checking
%
% Criteria (in order):
%   1. n_spikes >= MinSpikes within the session's task window
%   2. refractory-period violations (ISI < RefracMS) < RPV_Frac_Thresh
%   3. among cells passing 1-2, any pair correlating at r >= CorrThresh
%      (zero-lag, BinSizeS bins) is treated as one neuron recorded on two
%      microwires of the same bundle; one representative is kept, ranked by
%      spike count, then RPV, then firing stability, then rate.
%
% NOTE ON PROVENANCE. The run that produced the canonical abcd_passed.mat
% (2026-04-16) recorded two further settings, MinOverallFR_Hz = 0.1 and
% SessionLowFR_Hz = 0.1. Neither was ever applied: 13 accepted units fire below
% 0.1 Hz (minimum 0.049 Hz). They are therefore deliberately NOT implemented
% here -- adding them would change the accepted set. The three criteria above
% reproduce the canonical 1042 -> 984 split exactly.
%
% Usage:
%   matlab -batch "run('scripts/run_cell_qc.m')"
%
% @author: Svenja Kuchenhoff

clear; clc;

% ----------------------------- CONFIG --------------------------------------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans";
if ~exist(char(source_dir), 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys';
    in_file = sprintf("%s/beh_cells/abcd_data_08-Sep-2025.mat", source_dir);
else
    in_file = sprintf("%s/abcd_data_08-Sep-2025.mat", source_dir);
end

% Suffix for the outputs. Leave as "_rebuild" to write alongside the canonical
% files without touching them; set to "" only once the rebuild has been verified.
OUT_SUFFIX  = "_rebuild";
SAVE_PASSED = true;    % abcd_passed is ~6 GB and slow to write; skip while testing

qc_settings.MinSpikes       = 300;     % spikes in the task window
qc_settings.RefracMS        = 1.5;     % refractory window for RPV
qc_settings.RPV_Frac_Thresh = 0.01;    % < 1% of ISIs in refractory
qc_settings.BinSizeS        = 0.10;    % 100 ms bins for correlations
qc_settings.CorrWindow      = 'task';
qc_settings.CorrThresh      = 0.50;    % duplicate threshold
qc_settings.LowFRFrac       = 0.20;    % stability metric / tie-breaker only
qc_settings.MinTrialDur     = 0.050;

deriv_dir = fullfile(char(source_dir), 'derivatives');
if ~exist(deriv_dir, 'dir'); mkdir(deriv_dir); end

fprintf('Loading %s ...\n', in_file);
abcd_data = load(char(in_file));
subject_list = 1:numel(abcd_data.abcd_data);

% ----------------------------- MAIN LOOP -----------------------------------
qc_all = struct();
qc_all.meta = struct( ...
    'timestamp',   datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
    'source_file', char(in_file), ...
    'deriv_dir',   deriv_dir, ...
    'qc_settings', qc_settings, ...
    'qc_function', 'local: qc_one_session (in this file)');

qc_all.sessions = repmat(struct('subject_id','', 'session_index',[], ...
    'n_cells',0, 'n_pass',0, 'pass_mask',[], 'pass_idx',[], 'qc',[]), ...
    numel(subject_list), 1);

for sub = subject_list
    subj = abcd_data.abcd_data(sub);
    if isfield(subj,'subject_ID') && ~isempty(subj.subject_ID)
        subject_id = char(string(subj.subject_ID));
    else
        subject_id = sprintf('s%02d', sub);
    end

    qc = qc_one_session(subj, qc_settings);

    tmp       = [qc.is_reliable];
    pass_mask = tmp(:);
    qc_all.sessions(sub).subject_id    = subject_id;
    qc_all.sessions(sub).session_index = sub;
    qc_all.sessions(sub).n_cells       = numel(qc);
    qc_all.sessions(sub).n_pass        = sum(pass_mask);
    qc_all.sessions(sub).pass_mask     = pass_mask;
    qc_all.sessions(sub).pass_idx      = find(pass_mask);
    qc_all.sessions(sub).qc            = qc;

    fprintf('  %-14s %3d cells -> %3d pass\n', subject_id, numel(qc), sum(pass_mask));
end

n_cells_vec = arrayfun(@(s) s.n_cells, qc_all.sessions);
n_pass_vec  = arrayfun(@(s) s.n_pass,  qc_all.sessions);
total_cells = sum(n_cells_vec);
total_pass  = sum(n_pass_vec);

% ----------------------------- SAVE ----------------------------------------
save(fullfile(deriv_dir, sprintf('qc_all_sessions%s.mat', OUT_SUFFIX)), 'qc_all', '-v7');

% flat index of accepted cells
accepted = struct('session',{},'subject_id',{},'cell_idx',{}, ...
                  'electrodeLabel',{},'regionLabel',{});
k = 0;
for sub = subject_list
    sess = qc_all.sessions(sub);
    for ci = sess.pass_idx(:)'
        k = k + 1;
        accepted(k).session        = sub;
        accepted(k).subject_id     = sess.subject_id;
        accepted(k).cell_idx       = ci;
        accepted(k).electrodeLabel = sess.qc(ci).electrodeLabel;
        accepted(k).regionLabel    = sess.qc(ci).regionLabel;
    end
end
save(fullfile(deriv_dir, sprintf('accepted_cells_index%s.mat', OUT_SUFFIX)), 'accepted', '-v7');

% flat CSV so the decision can be diffed outside MATLAB
fid = fopen(fullfile(deriv_dir, sprintf('qc_pass_mask%s.csv', OUT_SUFFIX)), 'w');
fprintf(fid, 'session,subject_id,cell_idx,electrodeLabel,regionLabel,n_spikes,FR_Hz,RPV_frac,corr_max,is_reliable,fail_reasons\n');
for sub = subject_list
    sess = qc_all.sessions(sub);
    for ci = 1:sess.n_cells
        c = sess.qc(ci);
        fr = ''; if ~isempty(c.fail_reasons); fr = strjoin(c.fail_reasons, '; '); end
        cm = c.corr_max; if ~isfinite(cm); cm = NaN; end
        fprintf(fid, '%d,%s,%d,%s,%s,%d,%.6f,%.6f,%.6f,%d,"%s"\n', ...
            sub, sess.subject_id, ci, c.electrodeLabel, c.regionLabel, ...
            c.n_spikes, c.overall_FR_Hz, c.RPV_frac, cm, c.is_reliable, fr);
    end
end
fclose(fid);

if SAVE_PASSED
    abcd_passed = abcd_data;
    for sub = subject_list
        keep = qc_all.sessions(sub).pass_mask(:)';
        nd   = abcd_data.abcd_data(sub).neural_data;
        keep = keep(1:min(numel(keep), numel(nd)));
        abcd_passed.abcd_data(sub).neural_data = nd(keep);
    end
    fprintf('Writing abcd_passed%s.mat (this is large and slow) ...\n', OUT_SUFFIX);
    save(fullfile(deriv_dir, sprintf('abcd_passed%s.mat', OUT_SUFFIX)), 'abcd_passed', '-v7.3');
end

fprintf('\nTotal cells: %d | Passed: %d (%.1f%%) | Excluded: %d\n', ...
    total_cells, total_pass, 100*total_pass/max(1,total_cells), total_cells-total_pass);
fprintf('Outputs written to %s with suffix "%s"\n', deriv_dir, OUT_SUFFIX);


%% ====================== LOCAL FUNCTION ======================
function qc = qc_one_session(subj, S)
% Per-session QC: spike count, refractory violations, within-bundle duplicates.

MinSpikes = S.MinSpikes;
RefracS   = S.RefracMS/1000;
bin       = S.BinSizeS;
corrThr   = S.CorrThresh;
min_dur   = S.MinTrialDur;

% ---- task window from the trials ----
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
task_dur = task_t1 - task_t0;
ug = unique(grids(valid_trials));

% ---- cells ----
nC = numel(subj.neural_data);
spikeTimes_all = cell(nC,1); elecLabels = cell(nC,1); regionLabels = cell(nC,1);
for c = 1:nC
    nd = subj.neural_data(c);
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
    if isfield(nd,'spikeTimes') && ~isempty(nd.spikeTimes)
        st = sort(nd.spikeTimes(:)); st = st(isfinite(st));
    else
        st = [];
    end
    spikeTimes_all{c} = st;
end

    function gridFR = per_grid_fr(spikes)
        gridFR = nan(numel(ug),1);
        for gi = 1:numel(ug)
            idx = find(valid_trials & (grids==ug(gi)));
            tot_d = 0; tot_n = 0;
            for kk = 1:numel(idx)
                t0i = starts(idx(kk)); t1i = ends(idx(kk));
                tot_d = tot_d + (t1i - t0i);
                if ~isempty(spikes)
                    tot_n = tot_n + sum(spikes >= t0i & spikes < t1i);
                end
            end
            if tot_d > 0, gridFR(gi) = tot_n / tot_d; end
        end
    end

% ---- criteria 1 and 2 ----
qc = struct('electrodeLabel', elecLabels, 'regionLabel', regionLabels, ...
            'n_spikes',[], 'overall_FR_Hz',[], 'RPV_frac',[], 'RPV_count',[], ...
            'grid_FR_mean',[], 'grid_FR_CV',[], 'lowFR_grid_frac',[], ...
            'corr_max',[], 'corr_max_partner_idx',[], 'base_accept',[], ...
            'is_reliable',[], 'fail_reasons',[]);

for c = 1:nC
    st = spikeTimes_all{c};
    nsp = numel(st);
    if nsp >= 2, RPV = sum(diff(st) < RefracS); else, RPV = 0; end
    RPV_frac = RPV / max(1,nsp);

    gridFR = per_grid_fr(st);
    gridFR_mean = mean(gridFR, 'omitnan');
    gridFR_CV   = std(gridFR, 'omitnan') / max(eps, gridFR_mean);
    lowFR_mask  = gridFR < (S.LowFRFrac * gridFR_mean);
    lowFR_mask(isnan(gridFR)) = false;

    base_ok = (nsp >= MinSpikes) && (RPV_frac < S.RPV_Frac_Thresh);

    qc(c).n_spikes        = nsp;
    qc(c).overall_FR_Hz   = nsp / task_dur;
    qc(c).RPV_frac        = RPV_frac;
    qc(c).RPV_count       = RPV;
    qc(c).grid_FR_mean    = gridFR_mean;
    qc(c).grid_FR_CV      = gridFR_CV;
    qc(c).lowFR_grid_frac = mean(lowFR_mask);
    qc(c).corr_max        = NaN;
    qc(c).corr_max_partner_idx = NaN;
    qc(c).base_accept     = base_ok;
    qc(c).is_reliable     = base_ok;
    qc(c).fail_reasons    = {};
    if ~base_ok
        if nsp < MinSpikes
            qc(c).fail_reasons{end+1} = sprintf('few spikes (%d < %d)', nsp, MinSpikes);
        end
        if RPV_frac >= S.RPV_Frac_Thresh
            qc(c).fail_reasons{end+1} = sprintf('RPV %.2f%% >= %.2f%%', 100*RPV_frac, 100*S.RPV_Frac_Thresh);
        end
    end
end

% ---- criterion 3: zero-lag correlation over the task window ----
t0 = task_t0; t1 = task_t1;
edges = t0:bin:t1; if edges(end) < t1, edges = [edges, t1]; end

has_counts = false(nC,1); counts_all = cell(nC,1);
for c = 1:nC
    s = spikeTimes_all{c};
    if isempty(s), continue; end
    s = s(s>=t0 & s<t1);
    if isempty(s), continue; end
    cnt = histcounts(s, edges);
    if var(cnt) <= 0, continue; end
    counts_all{c} = cnt; has_counts(c) = true;
end
idx_counts = find(has_counts);
if numel(idx_counts) >= 2
    M = cell2mat(cellfun(@(x) x(:).', counts_all(idx_counts), 'UniformOutput', false));
    R = corrcoef(M.');
else
    R = [];
end

    function r = get_corr(i,j)
        r = NaN;
        if isempty(R) || i==j, return; end
        ii = find(idx_counts==i,1); jj = find(idx_counts==j,1);
        if isempty(ii) || isempty(jj), return; end
        r = R(ii,jj);
    end

for c = 1:nC
    rmax = -Inf; rmax_j = NaN;
    if ~isempty(R)
        for j = 1:nC
            if j==c, continue; end
            rij = get_corr(c,j);
            if isfinite(rij) && rij > rmax, rmax = rij; rmax_j = j; end
        end
    end
    if isfinite(rmax)
        qc(c).corr_max = rmax; qc(c).corr_max_partner_idx = rmax_j;
    end
end

% dedup among base-accepted cells
acc_idx = find([qc.base_accept]);
if numel(acc_idx) >= 2 && ~isempty(R)
    A = false(numel(acc_idx));
    for a = 1:numel(acc_idx)-1
        for b = a+1:numel(acc_idx)
            rij = get_corr(acc_idx(a), acc_idx(b));
            if isfinite(rij) && (rij >= corrThr), A(a,b) = true; A(b,a) = true; end
        end
    end
    visited = false(numel(acc_idx),1);
    for a = 1:numel(acc_idx)
        if visited(a), continue; end
        queue = a; comp = a; visited(a) = true;
        while ~isempty(queue)
            u = queue(1); queue(1) = [];
            for v = find(A(u,:))
                if ~visited(v)
                    visited(v) = true; queue(end+1) = v; comp(end+1) = v; %#ok<AGROW>
                end
            end
        end
        if numel(comp) <= 1, continue; end
        group_idx = acc_idx(comp);
        Spk  = arrayfun(@(k) qc(k).n_spikes,        group_idx);
        Rpv  = arrayfun(@(k) qc(k).RPV_frac,        group_idx);
        LowF = arrayfun(@(k) qc(k).lowFR_grid_frac, group_idx);
        CV   = arrayfun(@(k) qc(k).grid_FR_CV,      group_idx);
        FR   = arrayfun(@(k) qc(k).overall_FR_Hz,   group_idx);
        [~, ord] = sortrows([-Spk(:), Rpv(:), LowF(:), CV(:), -FR(:)]);
        winner = group_idx(ord(1));
        for k = setdiff(group_idx, winner)
            qc(k).is_reliable = false;
            qc(k).fail_reasons{end+1} = sprintf('high-corr (>=%.2f) duplicate of %s|%s', ...
                corrThr, qc(winner).electrodeLabel, qc(winner).regionLabel);
        end
    end
end
end
