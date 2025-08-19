%% Script to analyse spike-sorting reliability and save one MAT with all sessions
%% 18 Aug 2025

% ----------- CONFIG -----------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans";
if ~exist(char(source_dir), 'dir')
    source_dir = '/ceph/behrens/svenja/human_ABCD_ephys';
    abcd_data = load(sprintf("%s/beh_cells/abcd_data_24-Apr-2025.mat", source_dir));
else
    abcd_data = load(sprintf("%s/abcd_data_10-Jul-2025.mat", source_dir));
end

deriv_dir = fullfile(char(source_dir), 'derivatives');
if ~exist(deriv_dir,'dir'); mkdir(deriv_dir); end

min_cells_to_pass = 1;                     % session considered usable if >= this many units pass
subject_list = 1:length(abcd_data.abcd_data);

% ---- QC settings you want applied to all sessions (recorded into qc_all.meta) ----
dupe_settings.compareWithin = 'bundle';    % 'bundle' or 'all'
dupe_settings.windowMS      = 0.10;        % ±0.10 ms window
dupe_settings.excessFactor  = 3.0;         % >3× chance coincidences
dupe_settings.fracThresh    = 0.20;        % per-cell dupe fraction threshold (pre-dedup)

% ---- Preallocate results container ----
qc_all = struct();
qc_all.meta = struct( ...
    'timestamp', datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
    'source_dir', char(source_dir), ...
    'deriv_dir',  deriv_dir, ...
    'min_cells_to_pass', min_cells_to_pass, ...
    'dupe', dupe_settings, ...
    'qc_function', which('qc_single_session') ...
    );
qc_all.sessions = repmat(struct( ...
    'subject_id','', ...
    'session_index',[], ...
    'n_cells',0, ...
    'n_pass',0, ...
    'pass_mask',[], ...
    'pass_idx',[], ...
    'pass_labels',struct('electrodeLabel',{},'regionLabel',{}), ...
    'qc',[] ...
    ), numel(subject_list), 1);

% ----------- MAIN LOOP -----------
for sub = subject_list
    subj = abcd_data.abcd_data(sub);

    % Robust session ID
    if isfield(subj,'subject_ID') && ~isempty(subj.subject_ID)
        subject_id = char(string(subj.subject_ID));
    else
        subject_id = sprintf('s%02d', sub);
    end

    % ---- Run QC for this session ----
    qc = qc_single_session(subj, ...
        'SessionID', subject_id, 'SessionIndex', sub, ...
        'DupeCompareWithin', dupe_settings.compareWithin, ...
        'DupeWindowMS',      dupe_settings.windowMS, ...
        'DupeExcessFactor',  dupe_settings.excessFactor, ...
        'DupeFrac_Thresh',   dupe_settings.fracThresh ...
        );

    % ---- Build pass mask for downstream gating ----
    if isempty(qc)
        pass_mask   = false(0,1);
        pass_idx    = [];
        pass_labels = struct('electrodeLabel', {}, 'regionLabel', {});
        n_cells = 0; n_pass = 0;
    else
        tmp = [qc.is_reliable];          % concatenate logicals from struct array
        pass_mask = tmp(:);              % column vector
        pass_idx  = find(pass_mask);
        n_cells   = numel(qc);
        n_pass    = numel(pass_idx);

        pass_labels = struct( ...
            'electrodeLabel', {qc(pass_idx).electrodeLabel}, ...
            'regionLabel',    {qc(pass_idx).regionLabel} );
    end

    % ---- Store everything for this session ----
    qc_all.sessions(sub).subject_id    = subject_id;
    qc_all.sessions(sub).session_index = sub;
    qc_all.sessions(sub).n_cells       = n_cells;
    qc_all.sessions(sub).n_pass        = n_pass;
    qc_all.sessions(sub).pass_mask     = pass_mask;
    qc_all.sessions(sub).pass_idx      = pass_idx;
    qc_all.sessions(sub).pass_labels   = pass_labels;
    qc_all.sessions(sub).qc            = qc;           % full per-cell QC struct (includes dedup info)
end

% ----------- GLOBAL SUMMARY & SAVE -----------
n_cells_vec = arrayfun(@(s) s.n_cells, qc_all.sessions);
n_pass_vec  = arrayfun(@(s) s.n_pass,  qc_all.sessions);
total_cells = sum(n_cells_vec);
total_pass  = sum(n_pass_vec);
total_fail  = total_cells - total_pass;

% Session pass boolean (>= min_cells_to_pass)
qc_all.session_pass = n_pass_vec >= min_cells_to_pass;

% Simple per-session overview table (optional)
qc_all.overview = table( ...
    (1:numel(subject_list))', ...
    string({qc_all.sessions.subject_id}'), ...
    n_cells_vec(:), n_pass_vec(:), qc_all.session_pass(:), ...
    'VariableNames', {'session_idx','subject_id','n_cells','n_pass','session_pass'} );

% Save ONE file with everything
one_mat = fullfile(deriv_dir, 'qc_all_sessions.mat');
save(one_mat, 'qc_all', '-v7');

% Print final tally
fprintf('\nSaved QC for %d sessions to:\n  %s\n', numel(subject_list), one_mat);
fprintf('Total cells: %d | Passed: %d | Excluded: %d\n\n', total_cells, total_pass, total_fail);
