function [R, labels, used_idx] = plot_session_correlogram(subj, varargin)
% Quick-and-dirty cell-by-cell correlations for one session.
% Bins spikes, computes Pearson corr of binned counts, and plots a heatmap with r values.
%
% Inputs:
%   subj.neural_data(c).spikeTimes (sec)
%   subj.neural_data(c).electrodeLabel, .regionLabel (optional but nice)
%   subj.trial_vars(t).grid_onset_timestamp(1), .end_trial_timestamp
%
% Name-Value (all optional):
%   'BinSizeS'      (default 0.05)   % 50 ms bins
%   'Window'        'task'|'overlap' % task: [min onset, max end]; overlap: intersection across cells (clipped to task)
%   'MinSpikes'     (default 5)      % require at least this many spikes in the window
%   'UsePassedOnly' (default true)   % use PassMask if provided
%   'PassMask'      (default [])     % logical vector same length as subj.neural_data
%   'SessionID'     (default '')
%   'SessionIndex'  (default [])
%
% Returns:
%   R        NxN correlation matrix (N = included cells)
%   labels   cellstr of "electrode|region"
%   used_idx indices into subj.neural_data

% ----------------- args -----------------
p = inputParser;
addParameter(p, 'BinSizeS', 0.05, @(x) isnumeric(x) && x>0);
addParameter(p, 'Window', 'task', @(s) any(strcmpi(s,{'task','overlap'})));
addParameter(p, 'MinSpikes', 5, @(x) isnumeric(x) && x>=0);
addParameter(p, 'UsePassedOnly', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'PassMask', [], @(x) islogical(x) || isempty(x));
addParameter(p, 'SessionID', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'SessionIndex', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});
bin = p.Results.BinSizeS;

% ----------------- task window from trials -----------------
tv = subj.trial_vars(:);
T = numel(tv);
starts = nan(T,1); ends = nan(T,1);
for t = 1:T
    s = NaN; e = NaN;
    if isfield(tv(t),'grid_onset_timestamp') && ~isempty(tv(t).grid_onset_timestamp)
        s = tv(t).grid_onset_timestamp(1);
    end
    if isfield(tv(t),'end_trial_timestamp') && ~isempty(tv(t).end_trial_timestamp)
        e = tv(t).end_trial_timestamp;
    end
    starts(t) = s; ends(t) = e;
end
valid = isfinite(starts) & isfinite(ends) & (ends > starts);
if ~any(valid), error('No valid trial timing in this session.'); end
task_t0 = min(starts(valid));
task_t1 = max(ends(valid));

% ----------------- collect spikes & labels -----------------
N = numel(subj.neural_data);
labels_full = cell(N,1);
first_spk = inf(N,1); last_spk = -inf(N,1);
spk = cell(N,1);

for c = 1:N
    nd = subj.neural_data(c);
    % labels
    elab = 'electrode'; rlab = 'region';
    if isfield(nd,'electrodeLabel') && ~isempty(nd.electrodeLabel)
        elab = char(string(nd.electrodeLabel));
    end
    if isfield(nd,'regionLabel') && ~isempty(nd.regionLabel)
        rlab = char(string(nd.regionLabel));
    end
    labels_full{c} = sprintf('%s|%s', elab, rlab);
    % spikes
    if isfield(nd,'spikeTimes') && ~isempty(nd.spikeTimes)
        s = nd.spikeTimes(:);
        s = s(isfinite(s));
        s = sort(s);
    else
        s = [];
    end
    spk{c} = s;
    if ~isempty(s)
        first_spk(c) = s(1);
        last_spk(c)  = s(end);
    end
end

% ----------------- choose which cells to use -----------------
use = true(N,1);

% gate by PassMask if provided
if p.Results.UsePassedOnly && ~isempty(p.Results.PassMask)
    pm = p.Results.PassMask(:);
    if numel(pm) ~= N
        warning('PassMask length mismatch (got %d, expected %d). Ignoring pass mask.', numel(pm), N);
    else
        use = use & pm;
    end
end
% exclude empties
use = use & cellfun(@(x) ~isempty(x), spk);

% ----------------- choose time window -----------------
switch lower(p.Results.Window)
    case 'task'
        t0 = task_t0; t1 = task_t1;
    case 'overlap'
        t0 = max([task_t0; first_spk(use)]);
        t1 = min([task_t1; last_spk(use)]);
        if ~(t1 > t0)
            warning('No positive overlap across included cells; falling back to task window.');
            t0 = task_t0; t1 = task_t1;
        end
end
Tdur = t1 - t0;
if Tdur <= 0, error('Chosen window has non-positive duration.'); end

% ----------------- bin spikes -----------------
edges = t0:bin:t1;
if edges(end) < t1, edges = [edges, t1]; end

M = []; kept = []; kept_labels = {};
for c = 1:N
    if ~use(c), continue; end
    s = spk{c};
    s = s(s>=t0 & s<t1);
    if numel(s) < p.Results.MinSpikes, continue; end
    counts = histcounts(s, edges);
    if var(counts) <= 0, continue; end        % drop zero-variance series (corr=NaN)
    M(end+1, :) = counts; %#ok<AGROW>
    kept(end+1,1) = c; %#ok<AGROW>
    kept_labels{end+1,1} = labels_full{c}; %#ok<AGROW>
end

if isempty(M)
    warning('No cells survived inclusion for correlation in this session.');
    R = []; labels = {}; used_idx = [];
    return
end

% ----------------- correlation -----------------
R = corrcoef(M.');   % rows=cells, columns=time-bins

% ----------------- plot -----------------
figure('Color','w');
imagesc(R, [-1 1]); axis square
colormap(parula); colorbar
title(sprintf('Cell–cell corr | %s (idx %s) | bin=%.0f ms | window=%s | N=%d', ...
    string(p.Results.SessionID), local_idx_str(p.Results.SessionIndex), bin*1000, upper(p.Results.Window), size(R,1)));

xticks(1:numel(kept_labels)); yticks(1:numel(kept_labels));
xticklabels(kept_labels); yticklabels(kept_labels);
xtickangle(45);

% overlay r values
for i = 1:size(R,1)
    for j = 1:size(R,2)
        if ~isnan(R(i,j))
            txt = sprintf('%.2f', R(i,j));
            tc = 'k'; if abs(R(i,j)) > 0.6, tc = 'w'; end
            text(j, i, txt, 'HorizontalAlignment','center', 'FontSize', 7, 'Color', tc);
        end
    end
end
box on

labels   = kept_labels;
used_idx = kept;

% ------------- tiny helper -------------
function s = local_idx_str(k)
    if isempty(k), s = 'n/a'; else, s = num2str(k); end
end
end
