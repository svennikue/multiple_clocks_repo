function qc = qc_single_session(subj, varargin)
% QC_SINGLE_SESSION  Quality control for one session (one subject).
% Prints a concise summary and returns a struct array 'qc' (one entry per unit).
%
% Expected fields:
%   subj.neural_data(c).spikeTimes         (seconds)
%   subj.neural_data(c).electrodeLabel     (char/string)
%   subj.neural_data(c).regionLabel        (char/string)
%   subj.trial_vars(t).grid_onset_timestamp(1)
%   subj.trial_vars(t).end_trial_timestamp
%   subj.trial_vars(t).grid_num

% ----------------- Parameters -----------------
p = inputParser;
addParameter(p, 'RefracMS', 1.5, @(x) isnumeric(x) && x>0);
addParameter(p, 'RPV_Frac_Thresh', 0.01, @(x) isnumeric(x) && x>=0 && x<1);
addParameter(p, 'LowFRFrac', 0.20, @(x) isnumeric(x) && x>0 && x<1);
addParameter(p, 'LowFR_GridFrac_Thresh', 0.30, @(x) isnumeric(x) && x>=0 && x<=1);
addParameter(p, 'MinSpikes', 300, @(x) isnumeric(x) && x>=0);
% ---- lenient dupe controls ----
addParameter(p, 'DupeCompareWithin', 'bundle', @(s) any(strcmpi(s,{'bundle','all'})));
addParameter(p, 'DupeWindowMS', 0.10, @(x) isnumeric(x) && x>0);     % ±0.10 ms window
addParameter(p, 'DupeExcessFactor', 4.0, @(x) isnumeric(x) && x>=0); % >3× chance
addParameter(p, 'DupeFrac_Thresh', 0.3, @(x) isnumeric(x) && x>=0 && x<1);

% session metadata for printing
addParameter(p, 'SessionID', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'SessionIndex', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
addParameter(p, 'MinTrialDur', 0.050, @(x) isnumeric(x) && x>=0);
parse(p, varargin{:});

RefracS   = p.Results.RefracMS/1000;
DupeWinS  = p.Results.DupeWindowMS/1000;
min_dur   = p.Results.MinTrialDur;

% ----------------- Helpers -----------------
safe_label = @(val,def) get_safe_label(val,def);
    function s = get_safe_label(val,def)
        if nargin<2, def=''; end
        try
            if isempty(val)
                s = def;
            elseif ischar(val)
                s = val;
            elseif isstring(val)
                s = char(val);
            elseif iscell(val)
                s = safe_label(val{1},def);
            else
                s = char(string(val));
            end
        catch
            s = def;
        end
    end

% Extract a "bundle" label from electrodeLabel by stripping trailing digits
    function b = bundle_of(label)
        if isempty(label), b = ''; return; end
        s = char(string(label));
        b = regexprep(s,'\d+$','');  % e.g., 'LA1' -> 'LA'
    end

% ----------------- Trials -----------------
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
task_dur = task_t1 - task_t0;
if task_dur <= 0, error('Task duration not positive.'); end

ug = unique(grids(valid_trials));  % unique grids

% ----------------- Cells -----------------
if ~isfield(subj,'neural_data') || isempty(subj.neural_data)
    error('No neural_data in this session.');
end
nC = numel(subj.neural_data);

spikeTimes_all = cell(nC,1);
elecLabels = cell(nC,1);
regionLabels = cell(nC,1);

for c = 1:nC
    % Labels for printing
    if isfield(subj.neural_data(c),'electrodeLabel')
        elecLabels{c} = safe_label(subj.neural_data(c).electrodeLabel, sprintf('Electrode_%d', c));
    else
        elecLabels{c} = sprintf('Electrode_%d', c);
    end
    if isfield(subj.neural_data(c),'regionLabel')
        regionLabels{c} = safe_label(subj.neural_data(c).regionLabel, 'Region_?');
    else
        regionLabels{c} = 'Region_?';
    end

    % Spikes
    if isfield(subj.neural_data(c),'spikeTimes') && ~isempty(subj.neural_data(c).spikeTimes)
        st = subj.neural_data(c).spikeTimes(:);
        st = st(isfinite(st));
        st = sort(st);
    else
        st = [];
    end
    spikeTimes_all{c} = st;
end

% ---- per-grid FR helper ----
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
                t = idx(k);
                t0 = starts(t); t1 = ends(t);
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

% ---- lenient dupe function ----
    function dupe_frac = dupe_fraction_lenient(c_idx, compareWithin, window_s, excess_factor)
        st = spikeTimes_all{c_idx};
        nsp = numel(st);
        if nsp==0, dupe_frac = 0; return; end

        if strcmpi(compareWithin,'bundle')
            base = bundle_of(elecLabels{c_idx});
            same = cellfun(@(lab) strcmp(bundle_of(lab), base), elecLabels);
            hasSpikes = cellfun(@(x) ~isempty(x), spikeTimes_all);
            compare_idx = find(hasSpikes & (1:numel(spikeTimes_all))' ~= c_idx & same);
        else
            hasSpikes = cellfun(@(x) ~isempty(x), spikeTimes_all);
            compare_idx = find(hasSpikes & (1:numel(spikeTimes_all))' ~= c_idx);
        end
        if isempty(compare_idx), dupe_frac = 0; return; end

        Tdur = max(task_t1 - task_t0, eps); % seconds
        matched_c = false(size(st));
        excess_matches = 0;

        for j = 1:numel(compare_idx)
            k = compare_idx(j);
            st2 = spikeTimes_all{k};
            if isempty(st2), continue; end

            % two-pointer coincidence count
            i=1; m=1; obs = 0;
            while i<=numel(st) && m<=numel(st2)
                dt = st(i) - st2(m);
                if abs(dt) <= window_s
                    if ~matched_c(i)
                        matched_c(i) = true;
                        obs = obs + 1;
                    end
                    if st(i) <= st2(m), i = i+1; else, m = m+1; end
                elseif dt > window_s
                    m = m+1;
                else
                    i = i+1;
                end
            end

            % chance expectation
            r1 = numel(st)/Tdur; r2 = numel(st2)/Tdur;
            expected = (2*window_s) * r1 * r2 * Tdur;

            if obs > excess_factor * expected
                excess_matches = excess_matches + (obs - excess_factor * expected);
            end
        end

        dupe_frac = max(0, excess_matches) / max(1, nsp);
    end

% ---- Pairwise duplicate test (lenient, chance-corrected, within bundle/all) ----
function is_dup = is_duplicate_pair(c1, c2, compareWithin, window_s, excess_factor)
    % Only compare if same bundle when requested
    if strcmpi(compareWithin,'bundle')
        if ~strcmp(bundle_of(elecLabels{c1}), bundle_of(elecLabels{c2}))
            is_dup = false; return
        end
    end
    st1 = spikeTimes_all{c1}; st2 = spikeTimes_all{c2};
    if isempty(st1) || isempty(st2), is_dup = false; return; end

    % two-pointer coincidences, count each spike in st1 at most once
    i=1; j=1; obs=0; matched1=false(size(st1));
    while i<=numel(st1) && j<=numel(st2)
        dt = st1(i) - st2(j);
        if abs(dt) <= window_s
            if ~matched1(i)
                matched1(i) = true; obs = obs + 1;
            end
            if st1(i) <= st2(j), i=i+1; else, j=j+1; end
        elseif dt > window_s
            j=j+1;
        else
            i=i+1;
        end
    end

    Tdur = max(task_t1 - task_t0, eps);
    r1 = numel(st1)/Tdur; r2 = numel(st2)/Tdur;
    expected = (2*window_s) * r1 * r2 * Tdur;  % Poisson chance

    is_dup = obs > excess_factor * expected;
end

% ---- Build duplicate groups (graph of pairs -> connected components) ----
function groups = build_duplicate_groups(compareWithin, window_s, excess_factor)
    N = numel(spikeTimes_all);
    adj = false(N,N);
    for a = 1:N-1
        for b = a+1:N
            if is_duplicate_pair(a,b,compareWithin,window_s,excess_factor)
                adj(a,b)=true; adj(b,a)=true;
            end
        end
    end
    groups = {}; visited = false(N,1);
    for n = 1:N
        if ~visited(n)
            % BFS/DFS
            stack = n; visited(n)=true; comp = n;
            while ~isempty(stack)
                u = stack(end); stack(end) = [];
                nbrs = find(adj(u,:));
                for v = nbrs
                    if ~visited(v)
                        visited(v)=true;
                        stack(end+1)=v; %#ok<AGROW>
                        comp(end+1)=v; %#ok<AGROW>
                    end
                end
            end
            if numel(comp)>1
                groups{end+1} = sort(comp); %#ok<AGROW>
            end
        end
    end
end

% ---- Choose winner in a group (lower is better on first two; higher on last two) ----
function winner = choose_winner(idx_vec)
    % assemble ranking table
    RPV   = arrayfun(@(k) qc(k).RPV_frac,   idx_vec);
    lowFR = arrayfun(@(k) qc(k).lowFR_grid_frac, idx_vec);
    nsp   = arrayfun(@(k) qc(k).n_spikes,   idx_vec);
    FR    = arrayfun(@(k) qc(k).overall_FR_Hz, idx_vec);
    rankMat = [RPV(:), lowFR(:), -nsp(:), -FR(:)];
    [~, ord] = sortrows(rankMat, [1 2 3 4]);
    winner = idx_vec(ord(1));
end




% ----------------- QC per cell -----------------
qc = struct('electrodeLabel', elecLabels, ...
            'regionLabel',   regionLabels, ...
            'n_spikes', [], 'overall_FR_Hz', [], ...
            'RPV_frac', [], 'RPV_count', [], ...
            'grid_FR_mean', [], 'grid_FR_CV', [], ...
            'lowFR_grid_frac', [], ...
            'dupe_frac', [], 'dupe_count', [], ...
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

    % Per-grid FRs & stability
    [gridFR, ~] = per_grid_fr(st);
    gridFR_mean = nanmean(gridFR);
    gridFR_CV   = nanstd(gridFR) / max(eps, gridFR_mean);
    lowFR_mask  = gridFR < (p.Results.LowFRFrac * gridFR_mean);
    lowFR_mask(isnan(gridFR)) = false;
    lowFR_grid_frac = mean(lowFR_mask);

    % Lenient duplicates
    dupe_frac = dupe_fraction_lenient(c, p.Results.DupeCompareWithin, DupeWinS, p.Results.DupeExcessFactor);
    dupe_hits = round(dupe_frac * max(1,numel(st)));

    % Decision
    fail = {};
    if RPV_frac >= p.Results.RPV_Frac_Thresh
        fail{end+1} = sprintf('RPV %.2f%% ≥ %.2f%%', 100*RPV_frac, 100*p.Results.RPV_Frac_Thresh);
    end
    if dupe_frac >= p.Results.DupeFrac_Thresh
        fail{end+1} = sprintf('dupe %.2f%% ≥ %.2f%%', 100*dupe_frac, 100*p.Results.DupeFrac_Thresh);
    end
    if lowFR_grid_frac >= p.Results.LowFR_GridFrac_Thresh
        fail{end+1} = sprintf('lowFR grids %.1f%% ≥ %.1f%%', 100*lowFR_grid_frac, 100*p.Results.LowFR_GridFrac_Thresh);
    end
    if nsp < p.Results.MinSpikes
        fail{end+1} = sprintf('few spikes (%d < %d)', nsp, p.Results.MinSpikes);
    end

    is_reliable = isempty(fail);

    % Save
    qc(c).n_spikes         = nsp;
    qc(c).overall_FR_Hz    = overall_FR;
    qc(c).RPV_frac         = RPV_frac;
    qc(c).RPV_count        = RPV;
    qc(c).grid_FR_mean     = gridFR_mean;
    qc(c).grid_FR_CV       = gridFR_CV;
    qc(c).lowFR_grid_frac  = lowFR_grid_frac;
    qc(c).dupe_frac        = dupe_frac;
    qc(c).dupe_count       = dupe_hits;
    qc(c).is_reliable      = is_reliable;
    qc(c).fail_reasons     = fail;
end


% ----------------- Deduplicate: keep the best unit per duplicate group -----------------
groups = build_duplicate_groups(p.Results.DupeCompareWithin, DupeWinS, p.Results.DupeExcessFactor);

% annotate defaults
for k = 1:numel(qc)
    qc(k).dupe_group = 0;        % 0 = not in any dup group
    qc(k).dupe_primary = true;   % assume primary unless demoted below
    qc(k).dupe_primary_idx = k;
    qc(k).dupe_primary_label = qc(k).electrodeLabel;
end

for gi = 1:numel(groups)
    idx_vec = groups{gi};
    win = choose_winner(idx_vec);
    % mark group id
    for k = idx_vec
        qc(k).dupe_group = gi;
        qc(k).dupe_primary_idx = win;
        qc(k).dupe_primary_label = sprintf('%s|%s', qc(win).electrodeLabel, qc(win).regionLabel);
    end
    % demote losers
    losers = setdiff(idx_vec, win);
    for k = losers
        qc(k).dupe_primary = false;
        % force fail regardless of dupe_frac threshold
        if qc(k).is_reliable
            qc(k).is_reliable = false;
            qc(k).fail_reasons{end+1} = sprintf('duplicate of %s', qc(k).dupe_primary_label);
        else
            % already failed for other reasons; still record duplicate
            qc(k).fail_reasons{end+1} = sprintf('duplicate of %s', qc(k).dupe_primary_label);
        end
    end
end



% ----------------- Print summary -----------------
% Session ID/index for header
if isempty(p.Results.SessionID)
    if isfield(subj,'subject_ID'), sessID = safe_label(subj.subject_ID,'unknown_session');
    else, sessID = 'unknown_session';
    end
else
    sessID = safe_label(p.Results.SessionID,'unknown_session');
end

idxStr = 'n/a';
if ~isempty(p.Results.SessionIndex), idxStr = num2str(p.Results.SessionIndex); end
task_dur_in_mins = task_dur/60;

fprintf('\nQC summary — Session %s (idx %s) — task %.1f mins [%.3f → %.3f s]\n', ...
    sessID, idxStr, task_dur_in_mins, task_t0, task_t1);
fprintf('Criteria: RPV<%.2f%%, dupes<%.1f%%, lowFR grids<%.1f%%, spikes≥%d\n', ...
    100*p.Results.RPV_Frac_Thresh, 100*p.Results.DupeFrac_Thresh, ...
    100*p.Results.LowFR_GridFrac_Thresh, p.Results.MinSpikes);

for c = 1:nC
    tag = 'PASS'; if ~qc(c).is_reliable, tag = 'FAIL'; end
    fprintf('%s | %s: %s | spikes=%5d | FR=%.3f Hz | RPV=%.2f%% | dupes=%.2f%% | lowFRgrids=%.1f%%', ...
        qc(c).electrodeLabel, qc(c).regionLabel, tag, ...
        qc(c).n_spikes, qc(c).overall_FR_Hz, ...
        100*qc(c).RPV_frac, 100*qc(c).dupe_frac, 100*qc(c).lowFR_grid_frac);
    if ~qc(c).is_reliable
        fprintf(' | reasons: %s', strjoin(qc(c).fail_reasons, '; '));
    end
    fprintf('\n');
end
fprintf('\n');
end
